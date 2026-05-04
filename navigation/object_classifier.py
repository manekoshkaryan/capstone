"""
Classify a detection as LANDMARK / OBSTACLE / HAZARD.

Used by SpeechPolicy to decide whether an object even deserves an
unsolicited mention in passive mode. The rule from the assistive-nav spec:

  Silent user → speak ONLY for real walking obstacles or immediate hazards.
                Background landmarks (shelter, sign, bench on side, building,
                parked car not blocking path, wall far away) stay silent
                unless the user asks.

Inputs are detection dicts shaped like `navigator.tracked_to_detection`.
"""

from enum import Enum
from typing import Optional


class ObjectKind(str, Enum):
    HAZARD = "hazard"        # in path AND close — STOP / MOVE-around
    OBSTACLE = "obstacle"    # in path but farther — describe + maybe direct
    LANDMARK = "landmark"    # off-path; reference only; do not auto-announce
    BACKGROUND = "background"  # ignore in passive mode


# Walking-path bearings. Off-path = side, treated as landmark/background.
_PATH_BEARINGS = ("slight_left", "center", "slight_right")
_LATERAL_DANGER_BEARINGS = ("left", "right")  # close lateral can still hit you

# Labels that are real walking hazards if in path.
_OBSTACLE_LABELS = {
    "person", "pedestrian", "man", "woman", "child", "cyclist",
    "car", "bus", "truck", "motorcycle", "bicycle", "vehicle",
    "stairs", "step", "curb", "ramp",
    "door", "doorway", "gate",
    "wall", "partition", "fence", "barrier", "bollard", "pole", "pillar",
    "dog", "cat",
    "chair", "table", "bench", "sofa", "bed",
    "trash can", "shopping cart", "stroller", "wheelchair",
    "pothole", "puddle", "speed bump",
    "traffic cone", "fire hydrant",
}

# Labels that are landmarks even when close-ish (orientation, not danger).
_LANDMARK_LABELS = {
    "shelter", "building", "house", "tower", "monument", "tree",
    "sign", "stop sign", "traffic light", "bus stop", "kiosk",
    "fountain", "statue", "lamp", "lamppost", "street light",
    "billboard", "flag", "mailbox", "phone booth",
    "bookcase", "shelf", "cabinet",  # background indoor surfaces
    "window", "ceiling", "floor",
    "bathtub", "toilet", "sink",  # only relevant if user asks where bathroom is
    "picture", "painting", "mirror", "curtain",
}

# Labels never worth auto-announcing.
_BACKGROUND_LABELS = {
    "sky", "cloud", "grass", "ground", "ceiling", "floor",
    "wallpaper", "carpet", "tile",
}


def _label_matches(label: str, table, exact: bool = False) -> bool:
    if not label:
        return False
    s = label.lower().strip()
    if s in table:
        return True
    if exact:
        return False
    # Multi-word substring match for noisy YOLO labels like "young person".
    for k in table:
        if " " in k and k in s:
            return True
        if " " in s and s in k:
            return True
        if k == s:
            return True
    return False


def classify(detection: dict, hazard_distance_m: float = 1.0) -> ObjectKind:
    """Return the kind of object this detection represents."""
    label = str(detection.get("class", "")).lower()
    bearing = str(detection.get("bearing", "center"))
    dist = detection.get("distance_m")

    if _label_matches(label, _BACKGROUND_LABELS, exact=True):
        return ObjectKind.BACKGROUND

    in_path = bearing in _PATH_BEARINGS
    very_close_lateral = (
        dist is not None
        and dist < hazard_distance_m
        and bearing in _LATERAL_DANGER_BEARINGS
    )

    # Pure landmark labels are landmarks regardless of bearing — unless the
    # user is about to walk into one (very close, in path).
    if _label_matches(label, _LANDMARK_LABELS):
        if in_path and dist is not None and dist < hazard_distance_m * 1.5:
            return ObjectKind.HAZARD
        return ObjectKind.LANDMARK

    # Obstacle labels: hazard if close + in path; obstacle if just in path;
    # landmark if off to the side and not close enough to bump.
    if _label_matches(label, _OBSTACLE_LABELS):
        if very_close_lateral:
            return ObjectKind.HAZARD
        if in_path:
            if dist is not None and dist < hazard_distance_m:
                return ObjectKind.HAZARD
            return ObjectKind.OBSTACLE
        return ObjectKind.LANDMARK

    # Unknown label: only call it an obstacle if it's in path. Off-path
    # unknowns are landmarks (don't auto-announce them).
    if in_path:
        if dist is not None and dist < hazard_distance_m:
            return ObjectKind.HAZARD
        return ObjectKind.OBSTACLE
    return ObjectKind.LANDMARK


def is_announceable_in_passive(kind: ObjectKind) -> bool:
    """In passive mode, only obstacles + hazards are auto-announced."""
    return kind in (ObjectKind.OBSTACLE, ObjectKind.HAZARD)
