import time
import numpy as np
from collections import deque
from typing import Optional, Tuple


class Timer:
    def __init__(self):
        self._start: float = 0.0
        self._elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self._elapsed = (time.perf_counter() - self._start) * 1000.0

    @property
    def ms(self) -> float:
        return self._elapsed


class FPSCounter:
    def __init__(self, window: int = 30):
        self._timestamps: deque = deque(maxlen=window)

    def tick(self):
        self._timestamps.append(time.perf_counter())

    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        dt = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / dt if dt > 1e-9 else 0.0


class EMAFilter:
    def __init__(self, alpha: float = 0.3):
        self._alpha = alpha
        self._value: Optional[float] = None

    def update(self, measurement: float) -> float:
        if self._value is None:
            self._value = measurement
        else:
            self._value = self._alpha * measurement + (1.0 - self._alpha) * self._value
        return self._value

    @property
    def value(self) -> Optional[float]:
        return self._value

    def reset(self):
        self._value = None


class KalmanDepthFilter:
    def __init__(self, process_noise: float = 0.05, measurement_noise: float = 0.5):
        self._x: Optional[float] = None
        self._p: float = 1.0
        self._q = process_noise
        self._r = measurement_noise

    def update(self, measurement: float) -> float:
        if self._x is None:
            self._x = measurement
            return self._x
        self._p = self._p + self._q
        k = self._p / (self._p + self._r)
        self._x = self._x + k * (measurement - self._x)
        self._p = (1.0 - k) * self._p
        return self._x

    @property
    def value(self) -> Optional[float]:
        return self._x

    def reset(self):
        self._x = None
        self._p = 1.0


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0


def build_iou_matrix(tracks: list, detections: list) -> np.ndarray:
    matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            matrix[i, j] = compute_iou(t, d)
    return matrix


def load_vocabulary(path: str) -> list:
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        return lines
    except FileNotFoundError:
        return []


def confidence_class_to_dots(conf_class: str) -> str:
    if conf_class == "high":
        return "●●●"
    elif conf_class == "medium":
        return "●●○"
    return "●○○"


def confidence_class_to_color(conf_class: str) -> Tuple[int, int, int]:
    if conf_class == "high":
        return (0, 220, 80)
    elif conf_class == "medium":
        return (0, 190, 255)
    return (0, 80, 220)
