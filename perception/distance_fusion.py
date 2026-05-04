import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

import numpy as np

from utils.utils import EMAFilter

logger = logging.getLogger(__name__)

_DEPTH_MIN_M = 0.05
_DEPTH_MAX_M = 20.0

@dataclass
class DistanceResult:
    distance_m: float
    nearest_m: float
    method: str
    confidence: str
    variance: float = 0.0
    pixel_count: int = 0
    geometric_m: Optional[float] = None
    raw_depth_m: Optional[float] = None
    scale_used: float = 1.0

@dataclass
class FusionDiagnostics:
    method_counts: Dict[str, int] = field(default_factory=lambda: {
        "geometric": 0,
        "calibrated_depth": 0,
        "fallback_depth": 0,
        "none": 0,
    })
    global_scale: float = 1.0
    estimated_fx: Optional[float] = None
    last_geom_depth_ratio: Optional[float] = None
    samples_seen: int = 0

_PERSON_LABELS = {
    "person", "pedestrian", "man", "woman", "child", "boy", "girl",
    "human", "human face", "face", "head",
}

class HybridDistanceEstimator:
    METHOD_GEOMETRIC = "geometric"
    METHOD_CALIB_DEPTH = "calibrated_depth"
    METHOD_FALLBACK_DEPTH = "fallback_depth"
    METHOD_NONE = "none"

    def __init__(self, config, calib):
        self.config = config
        self.calib = calib
        self._global_scale = EMAFilter(alpha=config.distance_scale_alpha)
        if abs(config.initial_depth_scale_factor - 1.0) > 1e-6:
            self._global_scale.update(config.initial_depth_scale_factor)
        self._class_scale: Dict[str, EMAFilter] = {}
        self._fx_estimator = EMAFilter(alpha=config.online_fx_alpha)
        self._fy_estimator = EMAFilter(alpha=config.online_fx_alpha)
        self._diag = FusionDiagnostics()
        self._last_log_ts: float = 0.0

    def fuse(self, box, label, depth_map, frame_shape, det_conf, track_age=0):
        cfg = self.config
        H, W = frame_shape[:2]

        geom = self._compute_geometric(box, label, frame_shape) if cfg.use_geometric_distance else None

        if depth_map is not None:
            depth_med, depth_near, depth_var, depth_pix = self._compute_depth_roi(depth_map, box)
        else:
            depth_med = depth_near = None
            depth_var = float("inf")
            depth_pix = 0

        if geom is not None and depth_med is not None and depth_med > 0.05:
            ratio = geom / depth_med
            if 0.20 < ratio < 4.0 and (track_age >= 2 or det_conf >= cfg.det_conf_high_threshold):
                self._global_scale.update(ratio)
                if label not in self._class_scale:
                    self._class_scale[label] = EMAFilter(alpha=min(0.25, cfg.distance_scale_alpha * 2.0))
                self._class_scale[label].update(ratio)
                self._diag.last_geom_depth_ratio = ratio
                self._diag.samples_seen += 1

        if not self.calib.is_calibrated and cfg.enable_online_calibration:
            self._refine_focal_length_online(box, label, depth_med)

        scale_used = 1.0

        if geom is not None:
            final = geom * float(cfg.user_scale_factor)
            method = self.METHOD_GEOMETRIC
            reproj = getattr(self.calib, "reprojection_error", float("inf"))
            if self.calib.is_calibrated and reproj <= cfg.calibration_quality_warn_px:
                confidence = "high"
            else:
                confidence = "medium"
        elif depth_med is not None:
            scale = self._effective_scale(label) * float(cfg.user_scale_factor)
            scale_used = scale
            if abs(scale - 1.0) > 0.02 or self._diag.samples_seen >= 3:
                final = depth_med * scale
                method = self.METHOD_CALIB_DEPTH
                confidence = self._classify_depth_confidence(
                    depth_var, depth_pix, det_conf, track_age, calibrated=True
                )
            else:
                final = depth_med
                method = self.METHOD_FALLBACK_DEPTH
                confidence = self._classify_depth_confidence(
                    depth_var, depth_pix, det_conf, track_age, calibrated=False
                )
        else:
            self._diag.method_counts[self.METHOD_NONE] += 1
            return None

        if method == self.METHOD_FALLBACK_DEPTH:
            final = self._near_field_correction(final)

        if method == self.METHOD_GEOMETRIC:
            nearest = max(_DEPTH_MIN_M, final * 0.93)
        elif depth_near is not None:
            corrected = depth_near * scale_used
            if method == self.METHOD_FALLBACK_DEPTH:
                corrected = self._near_field_correction(corrected)
            nearest = corrected
        else:
            nearest = final

        self._diag.method_counts[method] = self._diag.method_counts.get(method, 0) + 1
        self._diag.global_scale = float(self._global_scale.value or 1.0)
        self._diag.estimated_fx = self._fx_estimator.value
        self._maybe_log_diag()

        return DistanceResult(
            distance_m=float(final),
            nearest_m=float(nearest),
            method=method,
            confidence=confidence,
            variance=depth_var,
            pixel_count=depth_pix,
            geometric_m=float(geom) if geom is not None else None,
            raw_depth_m=float(depth_med) if depth_med is not None else None,
            scale_used=float(scale_used),
        )

    @property
    def diagnostics(self) -> FusionDiagnostics:
        return self._diag

    def effective_fx(self, frame_width: int) -> float:
        if self.calib.is_calibrated and self.calib.fx:
            base = float(self.calib.fx)
        else:
            base = (frame_width / 2.0) / max(
                1e-6, math.tan(math.radians(self.config.default_h_fov_deg / 2.0))
            )
        if (
            self.config.enable_online_calibration
            and not self.calib.is_calibrated
            and self._fx_estimator.value is not None
        ):
            base = 0.5 * float(self._fx_estimator.value) + 0.5 * base
        return base

    def effective_fy(self, frame_height: int) -> float:
        if self.calib.is_calibrated and self.calib.fy:
            base = float(self.calib.fy)
        else:
            base = (frame_height / 2.0) / max(
                1e-6, math.tan(math.radians(self.config.default_v_fov_deg / 2.0))
            )
        if (
            self.config.enable_online_calibration
            and not self.calib.is_calibrated
            and self._fy_estimator.value is not None
        ):
            base = 0.5 * float(self._fy_estimator.value) + 0.5 * base
        return base

    def _label_is_person(self, label: str) -> bool:
        if not label:
            return False
        l = label.strip().lower()
        if l in _PERSON_LABELS:
            return True
        for k in _PERSON_LABELS:
            if k in l or l in k:
                return True
        return False

    def _compute_geometric(self, box, label, frame_shape):
        cfg = self.config
        H, W = frame_shape[:2]
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        bw = x2 - x1
        bh = y2 - y1
        if bw < cfg.geometric_min_box_dim_px or bh < cfg.geometric_min_box_dim_px:
            return None

        margin = cfg.geometric_partial_clip_margin_px
        clip_l = x1 < margin
        clip_t = y1 < margin
        clip_r = x2 > (W - margin)
        clip_b = y2 > (H - margin)

        real_h = cfg.known_height_for(label)
        real_w = cfg.known_width_for(label)
        fx = self.effective_fx(W)
        fy = self.effective_fy(H)

        is_person = self._label_is_person(label)
        bbox_area_ratio = (bw * bh) / max(1.0, float(W * H))
        is_closeup = is_person and bbox_area_ratio >= cfg.closeup_bbox_area_ratio

        candidates: List[Tuple[float, float]] = []

        if real_h is not None and not clip_t and not clip_b:
            d_h = (real_h * fy) / bh
            if 0.10 < d_h < cfg.depth_max_range_m * 1.5:
                candidates.append((d_h, cfg.geometric_height_weight))

        if real_w is not None and not clip_l and not clip_r:
            d_w = (real_w * fx) / bw
            if 0.10 < d_w < cfg.depth_max_range_m * 1.5:
                weight = cfg.geometric_width_weight
                if clip_t or clip_b:
                    weight = max(weight, 1.1)
                candidates.append((d_w, weight))

        if not candidates and is_closeup and real_w is not None:
            tight_margin = cfg.closeup_clip_margin_px
            cl_l = x1 < tight_margin
            cl_r = x2 > (W - tight_margin)
            if not (cl_l and cl_r):
                effective_bw = bw
                if cl_l:
                    effective_bw = (W - x1) * 2.0
                elif cl_r:
                    effective_bw = (x2 - 0.0) * 2.0 - (x2 - 0.0)
                    effective_bw = (x2 * 2.0) - W if x2 > W / 2 else bw
                effective_bw = max(effective_bw, bw)
                d_w = (real_w * fx) / max(1.0, effective_bw)
                if 0.10 < d_w < cfg.depth_max_range_m * 1.5:
                    candidates.append((d_w, 0.7))

        if not candidates and is_closeup:
            face_w = 0.16
            face_height_ratio_in_person_bbox = 0.20
            est_face_bw = bw * 0.45
            d_face = (face_w * fx) / max(1.0, est_face_bw)
            if 0.10 < d_face < 4.0:
                candidates.append((d_face, 0.4))

        if not candidates:
            return None

        if len(candidates) == 2:
            d1, w1 = candidates[0]
            d2, w2 = candidates[1]
            ratio = max(d1, d2) / max(1e-6, min(d1, d2))
            if ratio > 1.8:
                candidates = [max(candidates, key=lambda c: c[1])]

        weight_sum = sum(w for _, w in candidates)
        if weight_sum <= 0:
            return None
        d = sum(d_ * w for d_, w in candidates) / weight_sum
        if not (0.10 < d < cfg.depth_max_range_m * 1.5):
            return None
        return float(d)

    def _compute_depth_roi(self, depth_map, box):
        cfg = self.config
        h_map, w_map = depth_map.shape[:2]
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(w_map - 1, int(box[2]))
        y2 = min(h_map - 1, int(box[3]))
        if x2 <= x1 or y2 <= y1:
            return None, None, float("inf"), 0

        roi = depth_map[y1:y2, x1:x2]
        valid_mask = (roi > _DEPTH_MIN_M) & (roi < _DEPTH_MAX_M) & np.isfinite(roi)
        depths = roi[valid_mask]
        n = int(valid_mask.sum())
        if n < cfg.depth_min_valid_pixels:
            return None, None, float("inf"), n

        lo = np.percentile(depths, cfg.depth_roi_trim_low)
        hi = np.percentile(depths, cfg.depth_roi_trim_high)
        trimmed = depths[(depths >= lo) & (depths <= hi)]
        if len(trimmed) < 8:
            return None, None, float("inf"), n

        median = float(np.median(trimmed))
        var = float(np.var(trimmed))

        bh = y2 - y1
        bw = x2 - x1
        lc_y1 = y1 + int(0.55 * bh)
        lc_x1 = x1 + int(0.20 * bw)
        lc_x2 = x2 - int(0.20 * bw)
        lc_y2 = y2
        nearest = median
        if lc_y2 > lc_y1 and lc_x2 > lc_x1:
            lc_roi = depth_map[lc_y1:lc_y2, lc_x1:lc_x2]
            lc_valid = lc_roi[
                (lc_roi > _DEPTH_MIN_M) & (lc_roi < _DEPTH_MAX_M) & np.isfinite(lc_roi)
            ]
            if len(lc_valid) >= 8:
                nearest = float(np.percentile(lc_valid, 15))

        return median, nearest, var, n

    def _effective_scale(self, label: str) -> float:
        per_class = self._class_scale.get(label)
        class_value = float(per_class.value) if (per_class is not None and per_class.value is not None) else None
        global_value = float(self._global_scale.value) if self._global_scale.value is not None else 1.0
        if class_value is None:
            return global_value
        return 0.7 * class_value + 0.3 * global_value

    def _near_field_correction(self, d: float) -> float:
        cfg = self.config
        if d <= 0 or d > cfg.near_field_threshold_m:
            return d
        anchor = 0.20
        if d <= anchor:
            return d * cfg.near_field_min_factor
        t = (d - anchor) / max(1e-6, cfg.near_field_threshold_m - anchor)
        t = max(0.0, min(1.0, t))
        s = t * t * (3.0 - 2.0 * t)
        factor = cfg.near_field_min_factor + (1.0 - cfg.near_field_min_factor) * s
        return d * factor

    def _refine_focal_length_online(self, box, label, depth_med):
        if depth_med is None or depth_med <= 0:
            return
        scale = float(self._global_scale.value) if self._global_scale.value is not None else 1.0
        anchored = depth_med * scale
        if not (0.3 < anchored < 8.0):
            return
        real_h = self.config.known_height_for(label)
        real_w = self.config.known_width_for(label)
        bw = float(box[2]) - float(box[0])
        bh = float(box[3]) - float(box[1])
        if real_h is not None and bh > 12:
            fy_est = bh * anchored / real_h
            if 200.0 < fy_est < 4500.0:
                self._fy_estimator.update(fy_est)
        if real_w is not None and bw > 12:
            fx_est = bw * anchored / real_w
            if 200.0 < fx_est < 4500.0:
                self._fx_estimator.update(fx_est)

    def _classify_depth_confidence(self, variance, pixel_count, det_conf, track_age, calibrated):
        cfg = self.config
        score = 0
        if variance < cfg.depth_variance_medium_threshold:
            score += 2
        elif variance < cfg.depth_variance_high_threshold:
            score += 1
        if pixel_count >= cfg.roi_size_high_threshold:
            score += 2
        elif pixel_count >= cfg.roi_size_medium_threshold:
            score += 1
        if det_conf >= cfg.det_conf_high_threshold:
            score += 2
        elif det_conf >= cfg.det_conf_medium_threshold:
            score += 1
        if track_age >= 8:
            score += 2
        elif track_age >= 3:
            score += 1
        if calibrated:
            score += 1
        if score >= 7:
            return "high"
        if score >= 4:
            return "medium"
        return "low"

    def _maybe_log_diag(self) -> None:
        now = time.time()
        if now - self._last_log_ts < self.config.fusion_log_interval_s:
            return
        self._last_log_ts = now
        logger.info(
            "Fusion diag: counts=%s global_scale=%.3f user_scale=%.3f est_fx=%s est_fy=%s samples=%d",
            self._diag.method_counts,
            self._diag.global_scale,
            float(self.config.user_scale_factor),
            f"{self._fx_estimator.value:.0f}" if self._fx_estimator.value else "-",
            f"{self._fy_estimator.value:.0f}" if self._fy_estimator.value else "-",
            self._diag.samples_seen,
        )

def colorize_depth(depth_map, max_range_m: float):
    if depth_map is None:
        return None
    import cv2
    norm = np.clip(depth_map / max(1e-6, max_range_m), 0.0, 1.0)
    norm_u8 = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(norm_u8, cv2.COLORMAP_INFERNO)
