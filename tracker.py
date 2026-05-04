import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from scipy.optimize import linear_sum_assignment
from utils import compute_iou, build_iou_matrix, EMAFilter, KalmanDepthFilter

logger = logging.getLogger(__name__)


@dataclass
class RawDetection:
    box: np.ndarray
    label: str
    det_conf: float


@dataclass
class TrackedObject:
    track_id: int
    box: np.ndarray
    label: str
    det_conf: float
    frames_seen: int = 1
    frames_missed: int = 0
    distance_m: Optional[float] = None
    nearest_dist_m: Optional[float] = None
    distance_cm: Optional[float] = None
    dist_conf_class: str = "low"
    dist_variance: float = float("inf")
    dist_pixel_count: int = 0
    is_stable: bool = False
    is_obstacle: bool = False
    distance_method: str = "none"

    def update_distance(self, dist_m: Optional[float], nearest_m: Optional[float]):
        self.distance_m = dist_m
        self.nearest_dist_m = nearest_m
        self.distance_cm = round(dist_m * 100.0) if dist_m is not None else None


class Track:
    def __init__(self, track_id: int, detection: RawDetection, min_hits: int, ema_alpha_pos: float = 0.4):
        self.track_id = track_id
        self.box = detection.box.copy().astype(np.float32)
        self.label = detection.label
        self.det_conf = detection.det_conf
        self.frames_seen = 1
        self.frames_missed = 0
        self.min_hits = min_hits
        self._pos_ema_alpha = ema_alpha_pos
        self._depth_filter = KalmanDepthFilter()
        self._nearest_ema = EMAFilter(alpha=0.3)
        self.distance_m: Optional[float] = None
        self.nearest_dist_m: Optional[float] = None
        self.dist_conf_class: str = "low"
        self.dist_variance: float = float("inf")
        self.dist_pixel_count: int = 0
        self.distance_method: str = "none"
        self._label_history: Dict[str, int] = {detection.label: 1}

    @property
    def is_confirmed(self) -> bool:
        return self.frames_seen >= self.min_hits

    def predict_box(self) -> np.ndarray:
        return self.box.copy()

    def update_position(self, new_box: np.ndarray, det_conf: float, label: str):
        alpha = self._pos_ema_alpha
        self.box = alpha * new_box.astype(np.float32) + (1.0 - alpha) * self.box
        self.det_conf = alpha * det_conf + (1.0 - alpha) * self.det_conf
        self._label_history[label] = self._label_history.get(label, 0) + 1
        self.label = max(self._label_history, key=self._label_history.get)
        self.frames_missed = 0
        self.frames_seen += 1

    def update_depth(
        self,
        raw_dist_m: Optional[float],
        nearest_m: Optional[float],
        variance: float,
        pixel_count: int,
        conf_class: str,
        method: str = "monocular_depth",
    ):
        if raw_dist_m is not None:
            smoothed = self._depth_filter.update(raw_dist_m)
            self.distance_m = smoothed
        else:
            self.distance_m = self._depth_filter.value

        if nearest_m is not None:
            self.nearest_dist_m = self._nearest_ema.update(nearest_m)
        else:
            self.nearest_dist_m = self._nearest_ema.value

        self.dist_variance = variance
        self.dist_pixel_count = pixel_count
        self.dist_conf_class = conf_class
        self.distance_method = method

    def mark_missed(self):
        self.frames_missed += 1

    def to_tracked_object(self, obstacle_dist: float) -> TrackedObject:
        dist_m = self.distance_m
        obj = TrackedObject(
            track_id=self.track_id,
            box=self.box.copy(),
            label=self.label,
            det_conf=float(self.det_conf),
            frames_seen=self.frames_seen,
            frames_missed=self.frames_missed,
            distance_m=dist_m,
            nearest_dist_m=self.nearest_dist_m,
            distance_cm=round(dist_m * 100.0) if dist_m is not None else None,
            dist_conf_class=self.dist_conf_class,
            dist_variance=self.dist_variance,
            dist_pixel_count=self.dist_pixel_count,
            is_stable=self.is_confirmed,
            distance_method=self.distance_method,
        )
        nearest = self.nearest_dist_m or dist_m
        obj.is_obstacle = nearest is not None and nearest <= obstacle_dist
        return obj


class MultiObjectTracker:
    def __init__(self, max_age: int, min_hits: int, iou_threshold: float, ema_alpha_pos: float, obstacle_dist: float):
        self._max_age = max_age
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold
        self._ema_alpha_pos = ema_alpha_pos
        self._obstacle_dist = obstacle_dist
        self._tracks: List[Track] = []
        self._next_id: int = 0

    def _match(self, detections: List[RawDetection]):
        if not self._tracks or not detections:
            return [], list(range(len(self._tracks))), list(range(len(detections)))

        predicted_boxes = np.array([t.predict_box() for t in self._tracks])
        det_boxes = np.array([d.box for d in detections])
        iou_mat = build_iou_matrix(predicted_boxes, det_boxes)

        cost = 1.0 - iou_mat
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks = []
        unmatched_tracks = list(range(len(self._tracks)))
        unmatched_dets = list(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= self._iou_threshold:
                matched_tracks.append((r, c))
                if r in unmatched_tracks:
                    unmatched_tracks.remove(r)
                if c in unmatched_dets:
                    unmatched_dets.remove(c)

        return matched_tracks, unmatched_tracks, unmatched_dets

    def update(self, detections: List[RawDetection]) -> List[TrackedObject]:
        matched, unmatched_tracks, unmatched_dets = self._match(detections)

        for t_idx, d_idx in matched:
            self._tracks[t_idx].update_position(
                detections[d_idx].box,
                detections[d_idx].det_conf,
                detections[d_idx].label,
            )

        for t_idx in unmatched_tracks:
            self._tracks[t_idx].mark_missed()

        for d_idx in unmatched_dets:
            new_track = Track(
                track_id=self._next_id,
                detection=detections[d_idx],
                min_hits=self._min_hits,
                ema_alpha_pos=self._ema_alpha_pos,
            )
            self._tracks.append(new_track)
            self._next_id += 1

        self._tracks = [t for t in self._tracks if t.frames_missed <= self._max_age]

        return [t.to_tracked_object(self._obstacle_dist) for t in self._tracks if t.is_confirmed]

    def update_depth_for_track(
        self,
        track_id: int,
        raw_dist_m: Optional[float],
        nearest_m: Optional[float],
        variance: float,
        pixel_count: int,
        conf_class: str,
        method: str = "monocular_depth",
    ):
        for t in self._tracks:
            if t.track_id == track_id:
                t.update_depth(raw_dist_m, nearest_m, variance, pixel_count, conf_class, method)
                return

    def get_confirmed_tracks(self) -> List[Track]:
        return [t for t in self._tracks if t.is_confirmed]

    def reset(self):
        self._tracks.clear()
        self._next_id = 0
