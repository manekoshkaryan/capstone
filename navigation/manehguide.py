
"""ManehNavigationSystem — deterministic decision + speech renderer.

Pure rule engine. No I/O, no randomness, no LLM. Always produces a complete
spoken sentence on its own. Optional LLM paraphrase wraps `render()`; if the
LLM call fails or fails contract validation the local sentence is used.

This module is the single source of truth for STOP / BEAR / GO / CONTINUE
phrasing. The existing GuidanceController delegates the action decision here
when no landmark target is set; the SpeechPolicy still enforces throttling,
mode gates, and CommandValidator.
"""
from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple


# --------------------------------------------------------------------------
# Public types
# --------------------------------------------------------------------------

Position = Literal[
    "far_left", "left", "slight_left",
    "center",
    "slight_right", "right", "far_right",
]
Action = Literal[
    "STOP_IMMEDIATE", "STOP",
    "BEAR_LEFT", "BEAR_RIGHT",
    "GO_LEFT", "GO_RIGHT",
    "CONTINUE", "CLEAR",
]
Urgency = Literal["critical", "warn", "info"]


@dataclass(frozen=True)
class Detection:
    label: str
    distance_m: float
    position: Position
    confidence: float = 1.0
    velocity_mps: float = 0.0


@dataclass(frozen=True)
class FreeSpace:
    front_m: float
    left_m: float
    right_m: float


@dataclass(frozen=True)
class Decision:
    action: Action
    urgency: Urgency
    object: Optional[Detection]
    spoken_template: str
    distance_m: Optional[float]
    direction: Optional[str]


# --------------------------------------------------------------------------
# Frozen thresholds. Change only via a versioned config, never inline.
# --------------------------------------------------------------------------

T_PANIC: float = 0.5    # < this : STOP_IMMEDIATE
T_SLOW: float = 1.5     # < this front : adjust / bear
T_PREPARE: float = 3.0  # < this : in-path warn
T_CLEAR: float = 3.0    # >= this front and no obstacle in path : CLEAR
MIN_GO: float = 1.6     # any side must beat this to be a viable detour
MIN_CONF: float = 0.35  # below : detection ignored
HARD_TURN_FACTOR: float = 1.5  # side >= MIN_GO * factor → hard-turn

CENTER_BUCKET = {"center", "slight_left", "slight_right"}
LEFT_BUCKET = {"left", "far_left"}
RIGHT_BUCKET = {"right", "far_right"}


# --------------------------------------------------------------------------
# Decision (pure)
# --------------------------------------------------------------------------

def _closest(detections: List[Detection], in_center: bool) -> Optional[Detection]:
    if in_center:
        pool = [
            d for d in detections
            if d.confidence >= MIN_CONF and d.position in CENTER_BUCKET
        ]
    else:
        pool = [
            d for d in detections
            if d.confidence >= MIN_CONF and d.position not in CENTER_BUCKET
        ]
    if not pool:
        return None
    pool.sort(key=lambda d: d.distance_m)
    return pool[0]


def decide(detections: List[Detection], free: FreeSpace) -> Decision:
    """Single source of truth for navigation actions. Pure function."""
    target = _closest(detections, in_center=True)
    lateral = _closest(detections, in_center=False)

    # Rule 1 — panic stop overrides everything.
    if target is not None and target.distance_m < T_PANIC:
        return Decision(
            "STOP_IMMEDIATE", "critical", target,
            "Stop now. {label} ahead.", target.distance_m, "ahead",
        )

    sides_blocked = free.left_m < MIN_GO and free.right_m < MIN_GO

    # Rule 2 — both sides blocked + center blocked → STOP.
    if free.front_m < T_SLOW and sides_blocked:
        return Decision(
            "STOP", "critical", target,
            "Stop. Step back.", None, None,
        )

    # Rule 3 — center blocked, viable detour. When the cause object is
    # known, anchor phrasing on it (object-led BEAR), otherwise emit the
    # geometry-only call. Object-led calls always use BEAR; HARD turn is
    # reserved for purely geometric guidance with no specific obstacle.
    if free.front_m < T_SLOW:
        if free.left_m >= free.right_m:
            side = "left"
            side_clear = free.left_m
            action_bear: Action = "BEAR_LEFT"
            action_go: Action = "GO_LEFT"
        else:
            side = "right"
            side_clear = free.right_m
            action_bear = "BEAR_RIGHT"
            action_go = "GO_RIGHT"
        if target is not None and target.distance_m < T_PREPARE:
            return Decision(
                action_bear, "warn", target,
                "{label} ahead in {dist} meters. Bear " + side + ".",
                target.distance_m, side,
            )
        far = side_clear >= MIN_GO * HARD_TURN_FACTOR
        return Decision(
            action_go if far else action_bear, "warn", target,
            ("Hard " if far else "Bear ") + side + ".", None, side,
        )

    # Rule 4 — object in centre path within prepare zone → directional advisory.
    if target is not None and target.distance_m < T_PREPARE:
        if free.left_m >= free.right_m:
            suggest = "left"
            action: Action = "BEAR_LEFT"
        else:
            suggest = "right"
            action = "BEAR_RIGHT"
        return Decision(
            action, "warn", target,
            "{label} ahead in {dist} meters. Bear " + suggest + ".",
            target.distance_m, suggest,
        )

    # Rule 5 — lateral obstacle inside slow zone → advise away from it.
    if lateral is not None and lateral.distance_m < T_SLOW:
        if lateral.position in LEFT_BUCKET or lateral.position == "slight_left":
            suggest = "right"
            action = "BEAR_RIGHT"
        else:
            suggest = "left"
            action = "BEAR_LEFT"
        return Decision(
            action, "warn", lateral,
            "{label} ahead in {dist} meters. Bear " + suggest + ".",
            lateral.distance_m, suggest,
        )

    # Rule 6 — clear path. Front is wide AND any in-path target is far.
    target_far = target is None or target.distance_m >= T_PREPARE
    if free.front_m >= T_CLEAR and target_far:
        return Decision(
            "CLEAR", "info", None,
            "Path clear.", None, None,
        )

    # Rule 7 — default: keep going.
    return Decision(
        "CONTINUE", "info", target,
        "Continue forward.", None, None,
    )


# --------------------------------------------------------------------------
# Renderer (pure)
# --------------------------------------------------------------------------

def render(d: Decision) -> str:
    """Deterministic template render. Always produces a valid sentence."""
    label = (d.object.label if d.object is not None else "obstacle").strip()
    label = label[:1].upper() + label[1:].lower() if label else "Obstacle"
    dist = f"{d.distance_m:.1f}" if d.distance_m is not None else ""
    text = d.spoken_template.format(label=label, dist=dist)
    return " ".join(text.split())


# --------------------------------------------------------------------------
# LLM output validator
# --------------------------------------------------------------------------

_BANNED = ("maybe", "possibly", "could", "i think", "might", "perhaps",
           "probably", "uncertain")


def validate(text: str, d: Decision) -> Optional[str]:
    """Reject any LLM paraphrase that drops the action verb, exceeds the
    length contract, or smuggles uncertainty. Caller falls back to render()
    when this returns None."""
    if not text:
        return None
    cleaned = text.strip().strip('"').strip("'")
    if len(cleaned) > 60:
        return None
    if cleaned.count(".") > 2:
        return None
    low = cleaned.lower()
    if any(b in low for b in _BANNED):
        return None
    if d.action.startswith("STOP") and not low.startswith("stop"):
        return None
    if d.action in ("GO_LEFT", "BEAR_LEFT") and "left" not in low:
        return None
    if d.action in ("GO_RIGHT", "BEAR_RIGHT") and "right" not in low:
        return None
    if d.action == "CLEAR" and "clear" not in low:
        return None
    if d.action == "CONTINUE" and "continue" not in low:
        return None
    return cleaned


# --------------------------------------------------------------------------
# Helpers for callers building inputs from existing perception structures
# --------------------------------------------------------------------------

def detections_from_dicts(items: List[dict]) -> List[Detection]:
    out: List[Detection] = []
    for it in items:
        try:
            out.append(Detection(
                label=str(it.get("label") or it.get("class") or "obstacle"),
                distance_m=float(it.get("distance_m", it.get("distance", 0.0)) or 0.0),
                position=str(it.get("position") or it.get("bearing") or "center"),
                confidence=float(it.get("confidence", 1.0)),
                velocity_mps=float(it.get("velocity_mps", it.get("velocity", 0.0)) or 0.0),
            ))
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------
# Frozen system prompt (for OpenAI paraphrase layer). Hash-pin via tooling.
# --------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are the speech renderer for ManehNavigationSystem, a navigation "
    "assistant for visually impaired users.\n\n"
    "ROLE\n"
    "- You receive a pre-decided ACTION and a perception SNAPSHOT.\n"
    "- You output ONE short spoken sentence.\n"
    "- You DO NOT decide safety actions. The local rule engine has already "
    "decided STOP / MOVE / CONTINUE. You only render the words.\n\n"
    "HARD CONSTRAINTS\n"
    "1. Output exactly one sentence, <= 9 words, <= 60 characters.\n"
    "2. Use the imperative mood.\n"
    "3. Reuse the ACTION verb verbatim (Stop / Bear / Hard / Continue).\n"
    "4. Mention the distance only if PROVIDED in the snapshot, rounded to "
    "one decimal, in meters, e.g. \"1.5 meters\".\n"
    "5. Mention direction only if PROVIDED in the snapshot.\n"
    "6. Never invent objects, distances, or directions.\n"
    "7. Never use: \"maybe\", \"possibly\", \"could\", \"I think\", \"might\".\n"
    "8. Never list more than one object.\n"
    "9. Never add greetings, sign-offs, or filler.\n"
    "10. Never override the ACTION. If unsure, repeat the ACTION verbatim."
)


def build_user_payload(
    decision: Decision,
    free: FreeSpace,
    mode: str = "passive",
) -> dict:
    """Snapshot the per-cycle context for the LLM. Pass through json.dumps."""
    obj = None
    if decision.object is not None:
        obj = {
            "label": decision.object.label,
            "distance_m": round(decision.object.distance_m, 2),
            "position": decision.object.position,
        }
    return {
        "action": decision.action,
        "urgency": decision.urgency,
        "object": obj,
        "free_space": {
            "front_m": round(free.front_m, 2),
            "left_m": round(free.left_m, 2),
            "right_m": round(free.right_m, 2),
        },
        "mode": mode,
    }


# --------------------------------------------------------------------------
# Wire-in: convert FreeSpaceFrame sectors → ManehGuide FreeSpace
# --------------------------------------------------------------------------

def free_space_from_sectors(
    sectors_distances: List[Tuple[str, float]],
    center_clear_m: float,
) -> FreeSpace:
    """Reduce N-sector frame to a 3-bucket FreeSpace (front / left / right).

    `sectors_distances` is an ordered list of (name, free_distance_m) for
    sectors spanning the image left-to-right. Center is the middle index.
    """
    n = len(sectors_distances)
    if n == 0:
        return FreeSpace(front_m=0.0, left_m=0.0, right_m=0.0)
    center_idx = n // 2
    left_dists = [d for _, d in sectors_distances[:center_idx]]
    right_dists = [d for _, d in sectors_distances[center_idx + 1:]]
    return FreeSpace(
        front_m=float(center_clear_m),
        left_m=max(left_dists) if left_dists else 0.0,
        right_m=max(right_dists) if right_dists else 0.0,
    )
