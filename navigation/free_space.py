import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np

import navigation.manehguide as manehguide

logger = logging.getLogger(__name__)


_DECISION_TO_ACTION = {
    "STOP_IMMEDIATE": "STOP",
    "STOP": "STOP",
    "BEAR_LEFT": "BEAR_LEFT",
    "BEAR_RIGHT": "BEAR_RIGHT",
    "GO_LEFT": "GO_LEFT",
    "GO_RIGHT": "GO_RIGHT",
    "CONTINUE": "GO_STRAIGHT",
    "CLEAR": "GO_STRAIGHT",
}

@dataclass
class SectorState:
    name: str
    center_rel: float
    free_distance_m: float = 0.0
    obstacle_label: Optional[str] = None

@dataclass
class FreeSpaceFrame:
    sectors: List[SectorState] = field(default_factory=list)
    best_sector_idx: int = -1
    center_clear_m: float = 0.0
    nearest_global_m: float = float("inf")
    timestamp: float = 0.0

class FreeSpaceAnalyzer:
    def __init__(self, config):
        self.config = config
        names = config.guidance_sector_names
        bounds = np.linspace(0.0, 1.0, len(names) + 1)
        self._sector_bounds: List[tuple] = []
        for i in range(len(names)):
            self._sector_bounds.append((float(bounds[i]), float(bounds[i + 1])))
        self._sector_names = names
        self._ema: List[Optional[float]] = [None] * len(names)
        self._ema_alpha: float = float(getattr(config, "guidance_sector_ema_alpha", 0.4))

    def analyze(self, depth_map, tracked_objects, floor_mask) -> FreeSpaceFrame:
        cfg = self.config
        ts = time.time()
        if depth_map is None:
            sectors = [SectorState(n, (b[0] + b[1]) / 2) for n, b in zip(self._sector_names, self._sector_bounds)]
            return FreeSpaceFrame(sectors=sectors, timestamp=ts)
        h, w = depth_map.shape[:2]
        sectors: List[SectorState] = []
        nearest_global = float("inf")

        alpha = self._ema_alpha
        for i, (name, (lo, hi)) in enumerate(zip(self._sector_names, self._sector_bounds)):
            x1 = int(lo * w)
            x2 = int(hi * w)
            y1 = max(0, int(h * cfg.guidance_sample_top_ratio))
            y2 = h
            roi = depth_map[y1:y2, x1:x2]
            valid_mask = (roi > 0.10) & (roi < cfg.depth_max_range_m) & np.isfinite(roi)
            if floor_mask is not None:
                fm = floor_mask[y1:y2, x1:x2]
                obstacle_mask = valid_mask & (~fm)
            else:
                obstacle_mask = valid_mask
            obstacles = roi[obstacle_mask]
            if obstacles.size < cfg.guidance_min_pixels_for_obstacle:
                raw_d = cfg.guidance_open_horizon_m
            else:
                raw_d = float(np.percentile(obstacles, cfg.guidance_distance_percentile))
            raw_d = min(raw_d, cfg.guidance_open_horizon_m)
            prev = self._ema[i]
            free_d = raw_d if prev is None else (alpha * raw_d + (1.0 - alpha) * prev)
            self._ema[i] = free_d
            sectors.append(SectorState(name=name, center_rel=(lo + hi) / 2, free_distance_m=free_d))
            if free_d < nearest_global:
                nearest_global = free_d

        for obj in tracked_objects:
            if not obj.is_stable or obj.distance_m is None:
                continue
            cx = (float(obj.box[0]) + float(obj.box[2])) / 2.0 / max(1.0, w)
            d = obj.nearest_dist_m if obj.nearest_dist_m is not None else obj.distance_m
            for i, (lo, hi) in enumerate(self._sector_bounds):
                if lo <= cx < hi:
                    if d < sectors[i].free_distance_m:
                        sectors[i].free_distance_m = d
                        sectors[i].obstacle_label = obj.label
                    break

        center_idx = len(sectors) // 2
        center_clear = sectors[center_idx].free_distance_m if sectors else 0.0
        best_idx = int(np.argmax([s.free_distance_m for s in sectors])) if sectors else -1
        return FreeSpaceFrame(
            sectors=sectors,
            best_sector_idx=best_idx,
            center_clear_m=center_clear,
            nearest_global_m=nearest_global,
            timestamp=ts,
        )

@dataclass
class GuidanceCommand:
    action: str
    text: str
    urgency: str
    sector: str = ""

class GuidanceController:
    """Pure decision module: turns a FreeSpaceFrame into a single
    GuidanceCommand. Does NOT speak. SpeechPolicy is the single emit point
    for all navigation speech.
    """

    ACTIONS = ("STOP", "GO_STRAIGHT", "BEAR_LEFT", "BEAR_RIGHT", "GO_LEFT", "GO_RIGHT", "REVERSE")

    def __init__(self, config, speech=None, mode_manager=None):
        # `speech` and `mode_manager` accepted for backward-compat but ignored.
        # All emission goes through SpeechPolicy.evaluate_guidance.
        self.config = config
        # Hysteresis: hold last directional family until a side beats both
        # center and the current family by `switch_margin_m` for
        # `switch_min_frames` consecutive frames. Stops the per-frame
        # BEAR_LEFT ↔ BEAR_RIGHT flip-flop the depth noise was producing.
        self._last_dir_family: str = "STRAIGHT"  # STRAIGHT | LEFT | RIGHT
        self._candidate_family: str = "STRAIGHT"
        self._candidate_streak: int = 0
        self._center_bias_m: float = float(getattr(config, "guidance_center_bias_m", 0.5))
        self._switch_margin_m: float = float(getattr(config, "guidance_switch_margin_m", 0.4))
        self._switch_min_frames: int = int(getattr(config, "guidance_switch_min_frames", 3))

    def _step_via_manehguide(
        self,
        fs: FreeSpaceFrame,
        tracked_detections: Optional[List[dict]],
        best_name: str,
    ) -> Optional[GuidanceCommand]:
        """Adapter: smoothed sectors + perception detections → manehguide
        Decision → free_space.GuidanceCommand."""
        sectors_dists = [(s.name, float(s.free_distance_m)) for s in fs.sectors]
        free = manehguide.free_space_from_sectors(sectors_dists, fs.center_clear_m)
        dets = manehguide.detections_from_dicts(tracked_detections or [])
        decision = manehguide.decide(dets, free)
        text = manehguide.render(decision)
        action = _DECISION_TO_ACTION.get(decision.action, "GO_STRAIGHT")
        return GuidanceCommand(
            action=action,
            text=text,
            urgency=decision.urgency,
            sector=best_name,
        )

    @staticmethod
    def _family(idx: int, center_idx: int) -> str:
        if idx < center_idx:
            return "LEFT"
        if idx > center_idx:
            return "RIGHT"
        return "STRAIGHT"

    def _stable_best_idx(self, fs: FreeSpaceFrame) -> int:
        """Pick best sector with center-bias + temporal hysteresis."""
        cfg = self.config
        sectors = fs.sectors
        n = len(sectors)
        center_idx = n // 2
        center_clear = sectors[center_idx].free_distance_m

        # Center-bias: if no side beats center by margin, stay straight.
        biased_scores = [
            s.free_distance_m + (self._center_bias_m if i == center_idx else 0.0)
            for i, s in enumerate(sectors)
        ]
        raw_best = int(np.argmax(biased_scores))
        raw_family = self._family(raw_best, center_idx)

        # Reject the side proposal if it's not at least `switch_margin_m`
        # better than the center sector — kills tiny-noise direction calls.
        if raw_family != "STRAIGHT":
            side_clear = sectors[raw_best].free_distance_m
            if side_clear - center_clear < self._switch_margin_m:
                raw_best = center_idx
                raw_family = "STRAIGHT"

        # When center is not walkable AND last family was STRAIGHT, take the
        # side proposal immediately — user must learn the path is blocked.
        # When last family was already a side, fall through to hysteresis so
        # LEFT ↔ RIGHT noise wobble is damped (the actual flip-flop bug).
        center_walkable = center_clear >= cfg.guidance_min_go_distance_m
        if not center_walkable and raw_family != "STRAIGHT" and self._last_dir_family == "STRAIGHT":
            self._last_dir_family = raw_family
            self._candidate_family = raw_family
            self._candidate_streak = 0
            return raw_best

        # Hysteresis. Need N consecutive frames of the new family before we
        # switch. STRAIGHT is sticky too — same rule applies in reverse.
        if raw_family == self._last_dir_family:
            self._candidate_family = raw_family
            self._candidate_streak = 0
            return raw_best
        if raw_family == self._candidate_family:
            self._candidate_streak += 1
        else:
            self._candidate_family = raw_family
            self._candidate_streak = 1
        if self._candidate_streak >= self._switch_min_frames:
            self._last_dir_family = raw_family
            self._candidate_streak = 0
            return raw_best
        # Not enough corroboration yet — keep the previous family. Pick the
        # best sector consistent with it.
        if self._last_dir_family == "STRAIGHT":
            return center_idx
        # LEFT or RIGHT: pick best within the chosen half.
        if self._last_dir_family == "LEFT":
            half = range(0, center_idx)
        else:
            half = range(center_idx + 1, n)
        if not half:
            return center_idx
        return max(half, key=lambda j: sectors[j].free_distance_m)

    def step(self, fs: FreeSpaceFrame, target_label: Optional[str], landmark_bearing_rel: Optional[float], tracked_detections: Optional[List[dict]] = None) -> Optional[GuidanceCommand]:
        cfg = self.config
        if not fs.sectors:
            return None

        center_idx = len(fs.sectors) // 2
        center_clear = fs.center_clear_m
        nearest = fs.nearest_global_m
        best_idx = self._stable_best_idx(fs)
        # Republish so consumers (renderer, snapshot) see the stable choice.
        fs.best_sector_idx = best_idx
        best_clear = fs.sectors[best_idx].free_distance_m if 0 <= best_idx < len(fs.sectors) else 0.0
        best_name = fs.sectors[best_idx].name if 0 <= best_idx < len(fs.sectors) else ""

        # Non-landmark cycles delegate the action+phrasing decision to the
        # manehguide rule engine. Landmark navigation keeps the legacy tree
        # since manehguide does not model "go toward X".
        if target_label is None:
            mg_cmd = self._step_via_manehguide(fs, tracked_detections, best_name)
            if mg_cmd is not None:
                return mg_cmd

        # STOP must reflect a blocked walking path, not just a side obstacle.
        # Old behavior used `nearest_global_m`, so a shelter on the left at
        # 0.7m triggered STOP even with the center clear ("stop, shelter
        # left"). Require the center to be blocked, AND no side alternative.
        center_blocked_critical = center_clear < cfg.guidance_stop_distance_m
        no_alternative = best_clear < cfg.guidance_min_go_distance_m

        def _stop_with_suggestion() -> str:
            if best_clear >= cfg.guidance_stop_distance_m * 1.2 and 0 <= best_idx < len(fs.sectors):
                side = "left" if best_idx < center_idx else ("right" if best_idx > center_idx else "")
                if side:
                    return f"Stop. Try {side}."
            return "Stop. Step back."

        if center_blocked_critical and no_alternative:
            cmd = GuidanceCommand("STOP", _stop_with_suggestion(), "critical", best_name)
        elif center_blocked_critical and best_clear >= cfg.guidance_min_go_distance_m:
            # Front blocked but a side is open — guide around instead of stopping.
            if best_idx < center_idx:
                if best_idx == 0:
                    cmd = GuidanceCommand("GO_LEFT", "Hard left.", "warn", best_name)
                else:
                    cmd = GuidanceCommand("BEAR_LEFT", "Bear left.", "warn", best_name)
            else:
                if best_idx == len(fs.sectors) - 1:
                    cmd = GuidanceCommand("GO_RIGHT", "Hard right.", "warn", best_name)
                else:
                    cmd = GuidanceCommand("BEAR_RIGHT", "Bear right.", "warn", best_name)
        elif target_label is not None and landmark_bearing_rel is not None:
            offset = landmark_bearing_rel - 0.5
            if abs(offset) < 0.10 and center_clear >= cfg.guidance_min_go_distance_m:
                cmd = GuidanceCommand("GO_STRAIGHT", f"Go straight toward {target_label}.", "info", "center")
            elif offset < 0:
                if best_idx <= center_idx and best_clear >= cfg.guidance_min_go_distance_m:
                    if best_idx == 0:
                        cmd = GuidanceCommand("GO_LEFT", f"Go left toward {target_label}.", "info", best_name)
                    else:
                        cmd = GuidanceCommand("BEAR_LEFT", f"Bear left toward {target_label}.", "info", best_name)
                else:
                    cmd = GuidanceCommand("STOP", f"{target_label} is to the left, but the way is blocked.", "warn", best_name)
            else:
                if best_idx >= center_idx and best_clear >= cfg.guidance_min_go_distance_m:
                    if best_idx == len(fs.sectors) - 1:
                        cmd = GuidanceCommand("GO_RIGHT", f"Go right toward {target_label}.", "info", best_name)
                    else:
                        cmd = GuidanceCommand("BEAR_RIGHT", f"Bear right toward {target_label}.", "info", best_name)
                else:
                    cmd = GuidanceCommand("STOP", f"{target_label} is to the right, but the way is blocked.", "warn", best_name)
        else:
            if center_clear >= cfg.guidance_min_go_distance_m:
                cmd = GuidanceCommand("GO_STRAIGHT", "Continue straight.", "info", "center")
            elif best_clear >= cfg.guidance_min_go_distance_m:
                if best_idx < center_idx:
                    if best_idx == 0:
                        cmd = GuidanceCommand("GO_LEFT", "Hard left.", "warn", best_name)
                    else:
                        cmd = GuidanceCommand("BEAR_LEFT", "Bear left.", "warn", best_name)
                else:
                    if best_idx == len(fs.sectors) - 1:
                        cmd = GuidanceCommand("GO_RIGHT", "Hard right.", "warn", best_name)
                    else:
                        cmd = GuidanceCommand("BEAR_RIGHT", "Bear right.", "warn", best_name)
            else:
                cmd = GuidanceCommand("STOP", _stop_with_suggestion(), "critical", best_name)

        return cmd
