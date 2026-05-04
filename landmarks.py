import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)

@dataclass
class Landmark:
    label: str
    last_distance_m: float
    last_bearing_rel: float
    last_seen_ts: float
    last_frame_id: int
    world_x_m: float = 0.0
    world_z_m: float = 0.0
    sightings: int = 0

class LandmarkMemory:
    def __init__(self, config, imu=None):
        self.config = config
        self.imu = imu
        self._landmarks: Dict[str, Landmark] = {}
        self._cam_x: float = 0.0
        self._cam_z: float = 0.0
        self._yaw_at_session_start: Optional[float] = None

    def update(self, tracked_objects, frame_width: int, frame_id: int) -> None:
        ts = time.time()
        cfg = self.config
        yaw = self._current_yaw_rad()
        for obj in tracked_objects:
            if not obj.is_stable or obj.distance_m is None:
                continue
            if obj.det_conf < cfg.landmark_min_det_conf:
                continue
            cx = (float(obj.box[0]) + float(obj.box[2])) / 2.0
            bearing_rel = cx / max(1.0, frame_width)
            label_key = self._normalize_label(obj.label)
            d = obj.distance_m
            angle = (bearing_rel - 0.5) * cfg.guidance_h_fov_total_deg * math.pi / 180.0
            world_x = self._cam_x + d * math.sin(yaw + angle)
            world_z = self._cam_z + d * math.cos(yaw + angle)
            existing = self._landmarks.get(label_key)
            if existing is None:
                self._landmarks[label_key] = Landmark(
                    label=obj.label,
                    last_distance_m=d,
                    last_bearing_rel=bearing_rel,
                    last_seen_ts=ts,
                    last_frame_id=frame_id,
                    world_x_m=world_x,
                    world_z_m=world_z,
                    sightings=1,
                )
            else:
                a = cfg.landmark_position_alpha
                existing.last_distance_m = d
                existing.last_bearing_rel = bearing_rel
                existing.last_seen_ts = ts
                existing.last_frame_id = frame_id
                existing.world_x_m = a * world_x + (1.0 - a) * existing.world_x_m
                existing.world_z_m = a * world_z + (1.0 - a) * existing.world_z_m
                existing.sightings += 1

    def _current_yaw_rad(self) -> float:
        if self.imu is None:
            return 0.0
        r = self.imu.read()
        if not r.valid:
            return 0.0
        if self._yaw_at_session_start is None:
            self._yaw_at_session_start = r.yaw_deg
        return math.radians(r.yaw_deg - self._yaw_at_session_start)

    def _normalize_label(self, label: str) -> str:
        return label.strip().lower()

    def find(self, query: str) -> Optional[Landmark]:
        q = self._normalize_label(query)
        if not q:
            return None
        if q in self._landmarks:
            return self._landmarks[q]
        for k, lm in self._landmarks.items():
            if q in k or k in q:
                return lm
        return None

    def relative_bearing(self, landmark: Landmark) -> Tuple[float, float]:
        yaw = self._current_yaw_rad()
        dx = landmark.world_x_m - self._cam_x
        dz = landmark.world_z_m - self._cam_z
        dist = math.hypot(dx, dz)
        ang = math.atan2(dx, max(1e-6, dz)) - yaw
        rel = 0.5 + (ang * 180.0 / math.pi) / max(1.0, self.config.guidance_h_fov_total_deg)
        rel = max(0.0, min(1.0, rel))
        return rel, dist

    def list_recent(self, max_age_s: float = 60.0) -> List[Landmark]:
        now = time.time()
        return [lm for lm in self._landmarks.values() if (now - lm.last_seen_ts) <= max_age_s]

    def forget(self, query: str) -> bool:
        q = self._normalize_label(query)
        if q in self._landmarks:
            del self._landmarks[q]
            return True
        return False

    def clear(self) -> None:
        self._landmarks.clear()

    def update_pose_from_motion(self, dt_s: float, forward_speed_m_s: float = 0.0) -> None:
        if forward_speed_m_s == 0.0:
            return
        yaw = self._current_yaw_rad()
        self._cam_x += forward_speed_m_s * dt_s * math.sin(yaw)
        self._cam_z += forward_speed_m_s * dt_s * math.cos(yaw)
