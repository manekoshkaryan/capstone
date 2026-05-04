import logging
import time
import numpy as np
import torch
import cv2
from typing import Optional, Tuple
from PIL import Image
from config import AppConfig
from calibration import CalibrationData

logger = logging.getLogger(__name__)

_DEPTH_MIN_M = 0.05
_DEPTH_MAX_M = 20.0
_STATS_LOG_INTERVAL_S = 30.0


class DepthEstimator:
    def __init__(self, config: AppConfig):
        self._config = config
        self._model = None
        self._processor = None
        self._device = config.device
        self._use_half = config.effective_half()
        self._loaded = False
        self._last_stats_ts: float = 0.0
        self._active_model_id: str = ""

    def load(self):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        dtype = torch.float16 if self._use_half else torch.float32

        for model_id in (self._config.depth_model_id, self._config.depth_model_id_fallback):
            try:
                logger.info(f"Loading depth model: {model_id}")
                proc = AutoImageProcessor.from_pretrained(model_id)
                model = AutoModelForDepthEstimation.from_pretrained(model_id, dtype=dtype)
                model.eval()
                model.to(self._device)
                self._processor = proc
                self._model = model
                self._active_model_id = model_id
                self._loaded = True
                logger.info(f"Depth model ready: {model_id} | {self._device} | {'fp16' if self._use_half else 'fp32'}")
                return
            except Exception as e:
                logger.warning(f"Depth model {model_id} failed: {e}")

        logger.error("All depth models failed to load — depth estimation disabled")
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def estimate(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        if not self._loaded:
            return None
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            inputs = self._processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            if self._use_half:
                inputs = {
                    k: v.to(torch.float16) if v.is_floating_point() else v
                    for k, v in inputs.items()
                }

            with torch.no_grad():
                if self._use_half and self._device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self._model(**inputs)
                else:
                    outputs = self._model(**inputs)

            pred_depth = outputs.predicted_depth

            h, w = frame_bgr.shape[:2]
            depth_resized = torch.nn.functional.interpolate(
                pred_depth.unsqueeze(1).float(),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            depth_np = depth_resized.cpu().numpy().astype(np.float32)
            depth_np = np.clip(depth_np, _DEPTH_MIN_M, _DEPTH_MAX_M)

            now = time.time()
            if now - self._last_stats_ts > _STATS_LOG_INTERVAL_S:
                self._last_stats_ts = now
                valid = depth_np[(depth_np > _DEPTH_MIN_M) & (depth_np < _DEPTH_MAX_M)]
                if len(valid) > 0:
                    logger.info(
                        f"Depth stats [{self._active_model_id.split('/')[-1]}]: "
                        f"min={valid.min():.2f}m  mean={valid.mean():.2f}m  "
                        f"max={valid.max():.2f}m  p25={np.percentile(valid,25):.2f}m  "
                        f"p75={np.percentile(valid,75):.2f}m"
                    )
            return depth_np

        except Exception as e:
            logger.warning(f"Depth estimation error: {e}")
            return None

    def extract_roi_depth(
        self,
        depth_map: np.ndarray,
        box: np.ndarray,
        calib: CalibrationData,
        det_conf: float,
        track_age: int,
    ) -> Tuple[Optional[float], Optional[float], float, int, str]:
        cfg = self._config
        h_map, w_map = depth_map.shape[:2]

        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(w_map - 1, int(box[2]))
        y2 = min(h_map - 1, int(box[3]))

        if x2 <= x1 or y2 <= y1:
            return None, None, float("inf"), 0, "low"

        roi = depth_map[y1:y2, x1:x2]

        valid_mask = (roi > _DEPTH_MIN_M) & (roi < _DEPTH_MAX_M) & np.isfinite(roi)
        valid_depths = roi[valid_mask]
        pixel_count = int(valid_mask.sum())

        if pixel_count < cfg.depth_min_valid_pixels:
            return None, None, float("inf"), pixel_count, "low"

        low_p = np.percentile(valid_depths, cfg.depth_roi_trim_low)
        high_p = np.percentile(valid_depths, cfg.depth_roi_trim_high)
        trimmed = valid_depths[(valid_depths >= low_p) & (valid_depths <= high_p)]

        if len(trimmed) < 10:
            return None, None, float("inf"), pixel_count, "low"

        estimate_m = float(np.median(trimmed))
        variance = float(np.var(trimmed))

        bh = y2 - y1
        bw = x2 - x1
        lc_y1 = y1 + int(0.55 * bh)
        lc_x1 = x1 + int(0.20 * bw)
        lc_x2 = x2 - int(0.20 * bw)
        lc_y2 = y2

        nearest_m = estimate_m
        if lc_y1 < lc_y2 and lc_x1 < lc_x2:
            lc_roi = depth_map[lc_y1:lc_y2, lc_x1:lc_x2]
            lc_valid = lc_roi[(lc_roi > _DEPTH_MIN_M) & (lc_roi < _DEPTH_MAX_M) & np.isfinite(lc_roi)]
            if len(lc_valid) >= 10:
                nearest_m = float(np.percentile(lc_valid, 15))

        conf_class = _score_confidence(
            variance=variance,
            pixel_count=pixel_count,
            det_conf=det_conf,
            track_age=track_age,
            is_calibrated=calib.is_calibrated,
            cfg=cfg,
        )

        return estimate_m, nearest_m, variance, pixel_count, conf_class

    def colorize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        if depth_map is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        norm = np.clip(depth_map / self._config.depth_max_range_m, 0.0, 1.0)
        norm_u8 = (norm * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_INFERNO)
        return colored


def _score_confidence(
    variance: float,
    pixel_count: int,
    det_conf: float,
    track_age: int,
    is_calibrated: bool,
    cfg,
) -> str:
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

    if is_calibrated:
        score += 1

    if score >= 7:
        return "high"
    elif score >= 4:
        return "medium"
    return "low"
