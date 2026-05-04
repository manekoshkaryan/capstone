import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class FloorResult:
    mask: Optional[np.ndarray]
    coverage: float
    median_depth: float
    horizon_y: int

class FloorSegmenter:
    def __init__(self, config):
        self.config = config

    def segment(
        self,
        depth_map: Optional[np.ndarray],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        camera_height_m: float,
        pitch_deg: float = 0.0,
    ) -> FloorResult:
        if depth_map is None:
            return FloorResult(None, 0.0, 0.0, 0)

        h, w = depth_map.shape[:2]
        v_grid = np.arange(h, dtype=np.float32).reshape(-1, 1)
        v_grid = np.repeat(v_grid, w, axis=1)

        pitch_rad = math.radians(pitch_deg)
        cy_tilted = cy + math.tan(pitch_rad) * fy

        denom = (v_grid - cy_tilted)
        denom = np.where(np.abs(denom) < 1.0, 1.0, denom)
        expected = (camera_height_m * fy) / denom
        expected = np.where(denom > 0, expected, np.inf)

        valid = (depth_map > 0.10) & (depth_map < self.config.depth_max_range_m) & np.isfinite(depth_map)
        ratio_low = self.config.floor_depth_ratio_low
        ratio_high = self.config.floor_depth_ratio_high
        match = valid & (depth_map >= expected * ratio_low) & (depth_map <= expected * ratio_high)
        match = match & (v_grid > (cy_tilted + self.config.floor_min_below_horizon_px))

        try:
            import cv2
            mask_u8 = match.astype(np.uint8) * 255
            mask_u8 = cv2.medianBlur(mask_u8, 5)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
            num, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), 8)
            if num > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest = int(np.argmax(areas)) + 1
                mask_clean = (labels == largest)
            else:
                mask_clean = mask_u8 > 0
        except Exception:
            mask_clean = match

        coverage = float(np.count_nonzero(mask_clean)) / float(h * w)
        if np.any(mask_clean):
            median_depth = float(np.median(depth_map[mask_clean]))
        else:
            median_depth = 0.0
        horizon_y = int(round(cy_tilted))
        return FloorResult(mask=mask_clean, coverage=coverage, median_depth=median_depth, horizon_y=horizon_y)
