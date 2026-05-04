import time
import numpy as np
from typing import List, Tuple, Optional, Dict
from tracker import TrackedObject
from utils import compute_iou

_NAVIGATION_PRIORITY = {
    "person", "pedestrian", "man", "woman", "child", "cyclist",
    "car", "bus", "truck", "motorcycle", "bicycle", "vehicle",
    "stairs", "step", "curb", "ramp", "escalator", "elevator",
    "door", "doorway", "gate",
    "wall", "partition", "fence", "barrier", "bollard", "pole", "pillar",
    "dog", "cat",
    "traffic light", "stop sign", "crosswalk",
    "pothole", "puddle", "speed bump",
    "wheelchair", "stroller", "shopping cart",
}

_VERBAL_MAP: List[Tuple[float, str]] = [
    (0.40,  "very close"),
    (0.70,  "under one meter"),
    (1.15,  "one meter"),
    (1.65,  "one and a half meters"),
    (2.30,  "two meters"),
    (3.20,  "three meters"),
    (4.30,  "four meters"),
    (5.80,  "five meters"),
    (7.50,  "seven meters"),
    (10.5,  "ten meters"),
    (15.0,  "fifteen meters"),
    (float("inf"), "far away"),
]

_URGENCY_RANK = {"critical": 0, "warn": 1, "info": 2, "beyond": 3}


def get_bearing(box: np.ndarray, frame_width: int) -> str:
    cx = (float(box[0]) + float(box[2])) / 2.0
    rel = cx / max(frame_width, 1)
    if rel < 0.15:
        return "far_left"
    if rel < 0.35:
        return "left"
    if rel < 0.45:
        return "slight_left"
    if rel < 0.55:
        return "center"
    if rel < 0.65:
        return "slight_right"
    if rel < 0.85:
        return "right"
    return "far_right"


def get_direction(box: np.ndarray, frame_width: int) -> str:
    cx = (float(box[0]) + float(box[2])) / 2.0
    rel = cx / max(frame_width, 1)
    if rel < 0.35:
        return "to your left"
    if rel > 0.65:
        return "to your right"
    return "ahead"


def get_urgency(dist_m: float) -> str:
    if dist_m < 1.0:
        return "critical"
    if dist_m < 2.5:
        return "warn"
    if dist_m < 5.0:
        return "info"
    return "beyond"


def meters_to_verbal(dist_m: float) -> str:
    for threshold, label in _VERBAL_MAP:
        if dist_m <= threshold:
            return label
    return "very far away"


def _is_navigation_priority(label: str) -> bool:
    lower = label.lower()
    for p in _NAVIGATION_PRIORITY:
        if p in lower or lower in p:
            return True
    return False


def _suppress_overlapping(
    candidates: List[Tuple[TrackedObject, str, float]],
    iou_threshold: float = 0.45,
) -> List[Tuple[TrackedObject, str, float]]:
    suppressed = set()
    result = []
    for i, (obj_i, msg_i, dist_i) in enumerate(candidates):
        if i in suppressed:
            continue
        result.append((obj_i, msg_i, dist_i))
        for j in range(i + 1, len(candidates)):
            if j in suppressed:
                continue
            iou = compute_iou(candidates[i][0].box, candidates[j][0].box)
            if iou >= iou_threshold:
                suppressed.add(j)
    return result


def tracked_to_detection(obj: TrackedObject, frame_width: int, is_new: bool = True) -> dict:
    dist = obj.nearest_dist_m if obj.nearest_dist_m is not None else obj.distance_m
    urgency = get_urgency(dist) if dist is not None else "info"
    return {
        "class": obj.label,
        "distance_m": dist,
        "bearing": get_bearing(obj.box, frame_width),
        "confidence": float(obj.det_conf),
        "urgency": urgency,
        "is_new": is_new,
    }


def generate_nav_message(obj: TrackedObject, frame_width: int) -> str:
    direction = get_direction(obj.box, frame_width)
    label = obj.label.lower()
    dist_m = obj.nearest_dist_m if obj.nearest_dist_m is not None else obj.distance_m

    if dist_m is None:
        return f"{label} {direction}"

    verbal = meters_to_verbal(dist_m)
    urgency = get_urgency(dist_m)

    if urgency == "critical":
        return f"Stop. {label} {direction}, {verbal}"
    if urgency == "warn":
        return f"Slow. {label} {direction}, {verbal}"
    return f"{label} {direction}, {verbal}"


def prioritize_objects(
    objects: List[TrackedObject],
    max_dist_m: float,
    frame_width: int,
) -> List[Tuple[TrackedObject, str]]:
    candidates = []
    for obj in objects:
        if not obj.is_stable:
            continue
        dist = obj.nearest_dist_m if obj.nearest_dist_m is not None else obj.distance_m
        if dist is None or dist > max_dist_m:
            continue
        priority_boost = 0.0 if _is_navigation_priority(obj.label) else 1.5
        msg = generate_nav_message(obj, frame_width)
        candidates.append((obj, msg, dist + priority_boost))

    candidates.sort(key=lambda x: x[2])
    candidates = _suppress_overlapping(candidates)
    return [(obj, msg) for obj, msg, _ in candidates]


def build_navigation_record(
    objects: List[TrackedObject],
    frame_id: int,
    frame_width: int,
    max_dist_m: float = 20.0,
) -> dict:
    prioritized = prioritize_objects(objects, max_dist_m, frame_width)
    items = []
    for obj, msg in prioritized:
        dist = obj.nearest_dist_m if obj.nearest_dist_m is not None else obj.distance_m
        items.append({
            "track_id": obj.track_id,
            "label": obj.label,
            "bearing": get_bearing(obj.box, frame_width),
            "direction": get_direction(obj.box, frame_width),
            "distance_m": round(float(dist), 2) if dist is not None else None,
            "distance_verbal": meters_to_verbal(dist) if dist is not None else None,
            "urgency": get_urgency(dist) if dist is not None else "unknown",
            "message": msg,
            "is_obstacle": obj.is_obstacle,
            "det_conf": round(float(obj.det_conf), 3),
            "dist_conf": obj.dist_conf_class,
        })
    scene_parts = [msg for _, msg in prioritized[:3]]
    scene_summary = ". ".join(scene_parts) + "." if scene_parts else "Path clear."
    return {
        "ts": round(time.time(), 3),
        "frame_id": frame_id,
        "type": "navigation",
        "scene_summary": scene_summary,
        "object_count": len(items),
        "objects": items,
    }
