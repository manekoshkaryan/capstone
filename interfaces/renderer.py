
import cv2
import numpy as np
from typing import List, Optional, Dict, Any

from perception.tracker import TrackedObject
from utils.utils import confidence_class_to_color, confidence_class_to_dots

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE_LABEL = 0.52
_FONT_SCALE_SMALL = 0.42
_FONT_SCALE_PANEL = 0.48
_THICKNESS = 1
_BOX_THICKNESS = 2
_TOP_PANEL_H_VERBOSE = 76
_TOP_PANEL_H_MIN = 28
_BOTTOM_PANEL_H = 36
_GUIDANCE_BAR_H_MIN = 64
_GUIDANCE_BAR_H_VERBOSE = 38
_PANEL_BG = (20, 20, 20)
_PANEL_ALPHA = 0.82
_OBSTACLE_BOX_COLOR = (0, 50, 255)
_NO_DIST_COLOR = (120, 120, 120)
_LABEL_BG_ALPHA = 0.65
_DIAG_PANEL_W = 300

_METHOD_BADGE = {
    "geometric": ("G", (60, 220, 140)),
    "calibrated_depth": ("CD", (90, 200, 250)),
    "fallback_depth": ("FD", (140, 140, 200)),
    "none": ("—", (130, 130, 130)),
}

_GUIDANCE_COLOR = {
    "STOP": (40, 40, 240),
    "GO_STRAIGHT": (60, 220, 140),
    "BEAR_LEFT": (60, 200, 230),
    "BEAR_RIGHT": (60, 200, 230),
    "GO_LEFT": (90, 200, 250),
    "GO_RIGHT": (90, 200, 250),
    "REVERSE": (100, 100, 220),
}

_GUIDANCE_ARROW = {
    "STOP": "X",
    "GO_STRAIGHT": "^",
    "BEAR_LEFT": "<^",
    "BEAR_RIGHT": "^>",
    "GO_LEFT": "<",
    "GO_RIGHT": ">",
    "REVERSE": "v",
}

class Renderer:
    def __init__(self, config):
        self._config = config

    def draw_frame(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        depth_map: Optional[np.ndarray],
        depth_colorized: Optional[np.ndarray],
        metrics: Dict[str, Any],
        calib_mode_label: str,
    ) -> np.ndarray:
        out = frame.copy()

        if self._config.show_depth_view and depth_colorized is not None:
            if depth_colorized.shape[:2] != out.shape[:2]:
                depth_colorized = cv2.resize(depth_colorized, (out.shape[1], out.shape[0]))
            out = cv2.addWeighted(out, 0.55, depth_colorized, 0.45, 0)

        floor_mask = metrics.get("floor_mask")
        if floor_mask is not None and getattr(self._config, "show_floor_overlay", True):
            self._draw_floor(out, floor_mask, metrics.get("floor_horizon_y", 0))

        free_space = metrics.get("free_space")
        if free_space is not None and getattr(self._config, "show_sector_bars", False):
            self._draw_sectors(out, free_space)

        for obj in tracked_objects:
            if not obj.is_stable and not self._config.verbose_diagnostics:
                continue
            self._draw_object(out, obj)

        self._draw_top_panel(out, metrics, calib_mode_label)
        self._draw_bottom_panel(out, tracked_objects, metrics)
        self._draw_guidance_banner(out, metrics)
        self._draw_protocol_panel(out, metrics)

        if self._config.verbose_diagnostics:
            self._draw_diagnostics_panel(out, tracked_objects, metrics)

        return out

    def _draw_floor(self, frame, floor_mask, horizon_y):
        if floor_mask is None or floor_mask.size == 0:
            return
        h, w = frame.shape[:2]
        if floor_mask.shape[:2] != (h, w):
            floor_mask = cv2.resize(floor_mask.astype(np.uint8), (w, h)).astype(bool)
        overlay = frame.copy()
        overlay[floor_mask] = (60, 200, 80)
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
        if horizon_y and 0 <= horizon_y < h:
            cv2.line(frame, (0, horizon_y), (w, horizon_y), (200, 200, 0), 1, cv2.LINE_AA)

    def _draw_sectors(self, frame, fs):
        h, w = frame.shape[:2]
        n = len(fs.sectors)
        if n == 0:
            return
        bar_h = 12
        y0 = h - 36 - bar_h - 6
        for i, s in enumerate(fs.sectors):
            x1 = int(i * w / n) + 4
            x2 = int((i + 1) * w / n) - 4
            d = max(0.0, min(s.free_distance_m, 5.0))
            fill = int((d / 5.0) * (x2 - x1))
            color = (40, 40, 200) if d < 1.2 else ((40, 180, 220) if d < 2.0 else (60, 200, 80))
            if i == fs.best_sector_idx:
                cv2.rectangle(frame, (x1 - 2, y0 - 2), (x2 + 2, y0 + bar_h + 2), (255, 230, 130), 1)
            cv2.rectangle(frame, (x1, y0), (x2, y0 + bar_h), (40, 40, 40), -1)
            cv2.rectangle(frame, (x1, y0), (x1 + fill, y0 + bar_h), color, -1)
            label = f"{s.name} {d:.1f}m"
            cv2.putText(frame, label, (x1 + 2, y0 - 4), _FONT, 0.36, (210, 210, 210), 1, cv2.LINE_AA)

    def _draw_guidance_banner(self, frame, metrics):
        action = metrics.get("guidance_action", "")
        text = metrics.get("guidance_text", "")
        if not action and not text:
            return
        h, w = frame.shape[:2]
        verbose = getattr(self._config, "verbose_diagnostics", False)
        color = _GUIDANCE_COLOR.get(action, (180, 180, 180))
        arrow = _GUIDANCE_ARROW.get(action, "")
        target = metrics.get("target_landmark", "")
        msg = text if text else action

        if verbose:
            bar_h = _GUIDANCE_BAR_H_VERBOSE
            y0 = _TOP_PANEL_H_VERBOSE + 4
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, y0), (w - 10, y0 + bar_h), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
            cv2.rectangle(frame, (10, y0), (10 + 56, y0 + bar_h), color, -1)
            cv2.putText(frame, arrow, (18, y0 + 28), _FONT, 0.95, (15, 15, 15), 2, cv2.LINE_AA)
            if target:
                msg = f"[target: {target}]  {msg}"
            cv2.putText(frame, msg, (78, y0 + 25), _FONT, 0.55, color, 2, cv2.LINE_AA)
            return

        bar_h = _GUIDANCE_BAR_H_MIN
        y0 = _TOP_PANEL_H_MIN + 6
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, y0), (w - 10, y0 + bar_h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (10, y0), (10 + 90, y0 + bar_h), color, -1)
        ar_size, _ = cv2.getTextSize(arrow, _FONT, 1.7, 3)
        ax = 10 + (90 - ar_size[0]) // 2
        ay = y0 + (bar_h + ar_size[1]) // 2 - 4
        cv2.putText(frame, arrow, (ax, ay), _FONT, 1.7, (15, 15, 15), 3, cv2.LINE_AA)
        primary = msg.split(".")[0].strip() if msg else ""
        rest = msg[len(primary) + 1:].strip(" .") if msg and len(msg) > len(primary) else ""
        if target:
            rest = f"to {target}" if not rest else f"{rest}  ({target})"
        cv2.putText(frame, primary, (110, y0 + 32), _FONT, 0.95, (240, 240, 240), 2, cv2.LINE_AA)
        if rest:
            cv2.putText(frame, rest, (110, y0 + 54), _FONT, 0.6, color, 1, cv2.LINE_AA)

    def _draw_object(self, frame: np.ndarray, obj: TrackedObject):
        x1, y1, x2, y2 = map(int, obj.box)
        h, w = frame.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w - 1, x2); y2 = min(h - 1, y2)

        verbose = getattr(self._config, "verbose_diagnostics", False)
        method = getattr(obj, "distance_method", "none")
        badge_text, badge_color = _METHOD_BADGE.get(method, _METHOD_BADGE["none"])

        if obj.is_obstacle:
            box_color = _OBSTACLE_BOX_COLOR
        elif obj.distance_m is not None:
            box_color = confidence_class_to_color(obj.dist_conf_class)
        else:
            box_color = _NO_DIST_COLOR

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, _BOX_THICKNESS)

        if verbose:
            line1 = f"{obj.label}  {obj.det_conf:.2f}"
            if obj.distance_m is not None:
                dots = confidence_class_to_dots(obj.dist_conf_class)
                line2 = f"{obj.distance_m:.2f}m ({obj.distance_cm}cm)  {dots}"
            else:
                line2 = "depth: ---"
            self._draw_label_box(frame, x1, y1, line1, line2, box_color, badge_text, badge_color)
        else:
            if obj.distance_m is not None:
                line1 = f"{obj.label} {obj.distance_m:.1f}m"
            else:
                line1 = obj.label
            self._draw_label_simple(frame, x1, y1, line1, box_color)

        if obj.is_obstacle and obj.nearest_dist_m is not None and verbose:
            warn_txt = f"! OBSTACLE {obj.nearest_dist_m:.2f}m"
            tw, th = _text_size(warn_txt, _FONT_SCALE_SMALL, _THICKNESS)
            wx = x1
            wy = y2 + th + 6
            if wy + th < frame.shape[0]:
                cv2.putText(frame, warn_txt, (wx, wy), _FONT, _FONT_SCALE_SMALL, (0, 50, 255), 2)

        if self._config.show_tracking:
            tid_txt = f"#{obj.track_id}"
            cv2.putText(frame, tid_txt, (x2 - 30, y1 - 5), _FONT, 0.38, (200, 200, 200), 1)

    def _draw_label_simple(self, frame, x1, y1, text, color):
        pad = 4
        tw, th = _text_size(text, _FONT_SCALE_LABEL, _THICKNESS)
        bg_x1 = x1
        bg_y1 = max(0, y1 - th - pad * 2 - 1)
        bg_x2 = min(frame.shape[1] - 1, x1 + tw + pad * 2)
        bg_y2 = y1
        roi = frame[bg_y1:bg_y2, bg_x1:bg_x2]
        if roi.size > 0:
            bg = np.full_like(roi, 15)
            blended = cv2.addWeighted(roi, 1.0 - _LABEL_BG_ALPHA, bg, _LABEL_BG_ALPHA, 0)
            frame[bg_y1:bg_y2, bg_x1:bg_x2] = blended
        cv2.putText(frame, text, (bg_x1 + pad, bg_y2 - pad), _FONT, _FONT_SCALE_LABEL, color, _THICKNESS, cv2.LINE_AA)

    def _draw_label_box(
        self,
        frame: np.ndarray,
        x1: int, y1: int,
        line1: str, line2: str,
        color,
        badge_text: str,
        badge_color,
    ):
        pad = 4
        tw1, th1 = _text_size(line1, _FONT_SCALE_LABEL, _THICKNESS)
        tw2, th2 = _text_size(line2, _FONT_SCALE_SMALL, _THICKNESS)
        badge_w = max(_text_size(badge_text, _FONT_SCALE_SMALL, _THICKNESS)[0] + 8, 22)
        box_w = max(tw1, tw2) + pad * 2 + badge_w + 4
        box_h = th1 + th2 + pad * 3

        bg_x1 = x1
        bg_y1 = max(0, y1 - box_h - 2)
        bg_x2 = x1 + box_w
        bg_y2 = y1

        h, w = frame.shape[:2]
        bg_x2 = min(bg_x2, w - 1)
        bg_y1 = max(bg_y1, 0)

        roi = frame[bg_y1:bg_y2, bg_x1:bg_x2]
        if roi.size > 0:
            bg_rect = np.full_like(roi, 15)
            blended = cv2.addWeighted(roi, 1.0 - _LABEL_BG_ALPHA, bg_rect, _LABEL_BG_ALPHA, 0)
            frame[bg_y1:bg_y2, bg_x1:bg_x2] = blended

        badge_x1 = max(bg_x1, bg_x2 - badge_w - 2)
        badge_y1 = bg_y1 + 2
        badge_y2 = badge_y1 + th1 + 4
        cv2.rectangle(frame, (badge_x1, badge_y1), (bg_x2 - 2, badge_y2), badge_color, -1)
        cv2.putText(
            frame, badge_text,
            (badge_x1 + 4, badge_y2 - 3),
            _FONT, _FONT_SCALE_SMALL, (15, 15, 15), 1, cv2.LINE_AA,
        )

        text_y1 = bg_y1 + th1 + pad
        text_y2 = text_y1 + th2 + pad
        cv2.putText(frame, line1, (bg_x1 + pad, text_y1), _FONT, _FONT_SCALE_LABEL, color, _THICKNESS, cv2.LINE_AA)
        cv2.putText(frame, line2, (bg_x1 + pad, text_y2), _FONT, _FONT_SCALE_SMALL, (220, 220, 220), _THICKNESS, cv2.LINE_AA)

    def _draw_top_panel(self, frame: np.ndarray, metrics: Dict[str, Any], calib_label: str):
        h, w = frame.shape[:2]
        verbose = getattr(self._config, "verbose_diagnostics", False)
        panel_h = _TOP_PANEL_H_VERBOSE if verbose else _TOP_PANEL_H_MIN
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), _PANEL_BG, -1)
        cv2.addWeighted(overlay, _PANEL_ALPHA, frame, 1 - _PANEL_ALPHA, 0, frame)

        title = self._config.system_name
        fps = metrics.get("fps", 0.0)

        if not verbose:
            cv2.putText(frame, title, (10, 20), _FONT, 0.5, (60, 220, 140), 1, cv2.LINE_AA)
            fps_txt = f"{fps:.0f} fps"
            tw, _ = _text_size(fps_txt, 0.45, 1)
            cv2.putText(frame, fps_txt, (w - tw - 10, 20), _FONT, 0.45, (160, 200, 160), 1, cv2.LINE_AA)
            return

        cv2.putText(frame, title, (10, 22), _FONT, 0.7, (60, 220, 140), 2, cv2.LINE_AA)

        det_lat = metrics.get("det_ms", 0.0)
        dep_lat = metrics.get("dep_ms", 0.0)
        cap_lat = metrics.get("cap_ms", 0.0)
        pipe_lat = metrics.get("pipe_ms", 0.0)
        device_str = metrics.get("device", "CPU")
        det_model = metrics.get("det_model", "?")
        dep_model = metrics.get("dep_model", "?")
        n_objs = metrics.get("n_objects", 0)

        col1 = f"FPS:{fps:5.1f}  Cap:{cap_lat:.0f}  Det:{det_lat:.0f}  Dep:{dep_lat:.0f}  Total:{pipe_lat:.0f}ms"
        col2 = f"{device_str} | {det_model} | {dep_model} | objects:{n_objs}"

        cv2.putText(frame, col1, (170, 22), _FONT, _FONT_SCALE_PANEL, (180, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, col2, (10, 44), _FONT, _FONT_SCALE_PANEL, (180, 220, 255), 1, cv2.LINE_AA)
        if calib_label:
            cv2.putText(frame, calib_label, (10, 64), _FONT, _FONT_SCALE_SMALL, (80, 200, 255), 1, cv2.LINE_AA)

    def _draw_bottom_panel(self, frame: np.ndarray, objects: List[TrackedObject], metrics: Dict[str, Any]):
        h, w = frame.shape[:2]
        verbose = getattr(self._config, "verbose_diagnostics", False)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - _BOTTOM_PANEL_H), (w, h), _PANEL_BG, -1)
        cv2.addWeighted(overlay, _PANEL_ALPHA, frame, 1 - _PANEL_ALPHA, 0, frame)

        nearest_obj = None
        nearest_d = float("inf")
        for o in objects:
            if not o.is_stable:
                continue
            d = o.nearest_dist_m if o.nearest_dist_m is not None else o.distance_m
            if d is None:
                continue
            if d < nearest_d:
                nearest_d = d
                nearest_obj = o

        if nearest_obj is not None:
            cx = (float(nearest_obj.box[0]) + float(nearest_obj.box[2])) / 2.0
            rel = cx / max(w, 1)
            if rel < 0.35:
                direction = "left"
            elif rel > 0.65:
                direction = "right"
            else:
                direction = "ahead"
            urgency_color = (0, 80, 255) if nearest_d < 1.0 else ((0, 200, 255) if nearest_d < 2.5 else (180, 255, 180))
            if verbose:
                method = getattr(nearest_obj, "distance_method", "none")
                badge_text, _ = _METHOD_BADGE.get(method, _METHOD_BADGE["none"])
                text = f"NEAREST: {nearest_obj.label} {direction}  {nearest_d:.2f}m  [{badge_text}]"
            else:
                text = f"{nearest_obj.label} {direction}  {nearest_d:.1f}m"
            cv2.putText(frame, text, (10, h - 12), _FONT, 0.55, urgency_color, 2, cv2.LINE_AA)
        else:
            cv2.putText(
                frame, "Path clear." if not verbose else "No stable obstacles in range.",
                (10, h - 12), _FONT, 0.5, (160, 200, 160), 1, cv2.LINE_AA,
            )

        if not verbose:
            return

        counts = metrics.get("fusion_method_counts") or {}
        total = max(1, sum(counts.values()))
        order = ["geometric", "calibrated_depth", "fallback_depth"]
        x = w - 380
        for key in order:
            c = counts.get(key, 0)
            pct = (c / total) * 100.0
            badge_text, badge_color = _METHOD_BADGE[key]
            cv2.putText(
                frame, f"{badge_text} {pct:4.0f}%",
                (x, h - 12), _FONT, 0.5, badge_color, 1, cv2.LINE_AA,
            )
            x += 95

    def _draw_protocol_panel(self, frame: np.ndarray, metrics: Dict[str, Any]):
        mode = str(metrics.get("protocol_mode") or "")
        if not mode:
            return
        h, w = frame.shape[:2]
        msg = str(metrics.get("protocol_last_message") or "")
        wc = int(metrics.get("protocol_last_word_count") or 0)
        action = str(metrics.get("protocol_last_action") or "")
        resp_ms = metrics.get("protocol_last_response_ms")
        speech_ms = metrics.get("protocol_last_speech_ms")
        overlap = int(metrics.get("protocol_overlap_count") or 0)
        dup = int(metrics.get("protocol_duplicate_count") or 0)

        panel_w = 460
        panel_h = 88
        panel_x = w - panel_w - 12
        panel_y = h - _BOTTOM_PANEL_H - panel_h - 10
        if panel_y < 0:
            panel_y = 10
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (10, 10, 10),
            -1,
        )
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (60, 220, 140),
            1,
        )
        title = f"Protocol: {mode.upper()}  action={action or '-'}  wc={wc}"
        cv2.putText(
            frame, title, (panel_x + 10, panel_y + 20),
            _FONT, 0.5, (60, 220, 140), 1, cv2.LINE_AA,
        )
        latency_line = (
            f"resp={_fmt_ms(resp_ms)}  tts={_fmt_ms(speech_ms)}  "
            f"overlaps={overlap}  dup={dup}"
        )
        cv2.putText(
            frame, latency_line, (panel_x + 10, panel_y + 40),
            _FONT, 0.42, (200, 220, 200), 1, cv2.LINE_AA,
        )
        if msg:
            display = msg if len(msg) <= 56 else (msg[:53] + "...")
            cv2.putText(
                frame, display, (panel_x + 10, panel_y + 64),
                _FONT, 0.46, (240, 240, 240), 1, cv2.LINE_AA,
            )
        cv2.putText(
            frame, "press 5/6/7/8 to switch | e=run eval",
            (panel_x + 10, panel_y + panel_h - 6),
            _FONT, 0.36, (140, 180, 140), 1, cv2.LINE_AA,
        )

    def _draw_diagnostics_panel(self, frame: np.ndarray, objects: List[TrackedObject], metrics: Dict[str, Any]):
        h, w = frame.shape[:2]
        panel_x = w - _DIAG_PANEL_W
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, _TOP_PANEL_H_VERBOSE), (w, h - _BOTTOM_PANEL_H), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        y = _TOP_PANEL_H_VERBOSE + 18
        line_h = 16

        def put(txt, color=(200, 200, 200)):
            nonlocal y
            cv2.putText(frame, txt, (panel_x + 8, y), _FONT, 0.40, color, 1, cv2.LINE_AA)
            y += line_h

        put("=== DIAGNOSTICS ===", (255, 220, 80))
        put(f"Tracked: {len(objects)}")
        scale = metrics.get("fusion_global_scale", 1.0)
        est_fx = metrics.get("fusion_estimated_fx")
        eff_fx = metrics.get("effective_fx", 0.0)
        eff_fy = metrics.get("effective_fy", 0.0)
        samples = metrics.get("fusion_samples", 0)
        user_scale = metrics.get("user_scale_factor", 1.0)
        put(f"depth scale: {scale:.3f}  (samples {samples})", (180, 220, 255))
        put(f"user scale (+/-/0): {user_scale:.2f}", (255, 230, 130))
        put(f"effective fx={eff_fx:.0f}  fy={eff_fy:.0f}", (180, 220, 255))
        if est_fx is not None:
            put(f"online fx est: {est_fx:.0f} px", (180, 220, 255))
        imu_valid = metrics.get("imu_valid", False)
        if imu_valid:
            put(
                f"IMU pitch={metrics.get('imu_pitch_deg', 0.0):+.1f} yaw={metrics.get('imu_yaw_deg', 0.0):+.1f}",
                (180, 220, 255),
            )
        else:
            put("IMU: -", (130, 130, 130))
        cov = metrics.get("floor_coverage", 0.0)
        put(f"floor coverage: {cov*100:.1f}%", (180, 220, 255))
        put(f"landmarks: {metrics.get('landmark_count', 0)}", (180, 220, 255))
        target = metrics.get("target_landmark", "")
        if target:
            put(f"target: {target}", (255, 230, 130))
        reproj = metrics.get("calib_reproj_px")
        if reproj is not None:
            warn = metrics.get("calib_quality_warn_px", 1.0) or 1.0
            color = (180, 255, 180) if reproj <= warn else (0, 200, 255)
            put(f"calib err: {reproj:.3f}px", color)
        else:
            put("calib err: --- (uncalibrated)", (130, 130, 130))

        put("")
        put("--- objects ---", (255, 220, 80))
        for obj in objects[:10]:
            method = getattr(obj, "distance_method", "none")
            badge, _ = _METHOD_BADGE.get(method, _METHOD_BADGE["none"])
            d_str = f"{obj.distance_m:.2f}m" if obj.distance_m is not None else "---"
            put(
                f"#{obj.track_id} {obj.label[:9]:9s} {d_str} [{badge}/{obj.dist_conf_class}]",
                (200, 200, 200),
            )
            put(
                f"   px={obj.dist_pixel_count} seen={obj.frames_seen} miss={obj.frames_missed}",
                (140, 140, 140),
            )

    def draw_calibration_feedback(self, frame: np.ndarray, message: str) -> np.ndarray:
        out = frame.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 50), (20, 20, 20), -1)
        cv2.putText(out, message, (10, 32), _FONT, 0.7, (80, 255, 200), 2, cv2.LINE_AA)
        return out

def _text_size(text: str, scale: float, thickness: int):
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    return tw, th


def _fmt_ms(value):
    if value is None:
        return "--"
    try:
        return f"{float(value):.0f}ms"
    except (TypeError, ValueError):
        return "--"
