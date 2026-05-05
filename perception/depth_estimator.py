import logging
import numpy as np
import cv2
from typing import Optional

logger = logging.getLogger(__name__)

_DEPTH_MIN_M = 0.05
_DEPTH_MAX_M = 20.0


class LiDARDepthSource:
    """Provides depth maps received from the iOS LiDAR app via WebSocket."""

    def __init__(self, server):
        self._server = server

    def get(self, frame_bgr: np.ndarray, label: str = "iphone") -> Optional[np.ndarray]:
        frame = self._server.latest_depth(label)
        if frame is None:
            return None
        h, w = frame_bgr.shape[:2]
        depth = frame.depth_m
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(depth, _DEPTH_MIN_M, _DEPTH_MAX_M)
