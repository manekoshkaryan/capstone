"""
ManehNavigationSystem — speech logic module
Mane Koshkaryan, AUA Capstone

Pipeline:
    perception JSON  ->  decide()  ->  Decision  ->  speak()  ->  TTS string

The decision and phrasing layers are fully deterministic. No LLM call sits
in the real-time loop. The OpenAI prompt at the bottom is provided for:
    1. offline authoring / regeneration of phrasing templates,
    2. an optional fallback path for rare scene types.

Run `python maneh_speech.py` for the self-test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Tunable thresholds (meters). Calibrate to your camera + environment.
# ---------------------------------------------------------------------------
STOP_DISTANCE    = 0.5    # immediate danger
SLOW_DISTANCE    = 1.5    # adjust direction now
AVOID_DISTANCE   = 3.0    # prepare to avoid
INFO_DISTANCE    = 5.0    # informational only
CONFIDENCE_FLOOR = 0.40   # detections below this are dropped

Position = Literal["center", "left", "right"]
Action   = Literal["STOP", "SLOW", "AVOID", "INFO", "CLEAR"]
Move     = Literal["stop", "left", "right", "forward"]

# Labels that are useful for the user to hear by name. Anything outside this
# set is collapsed to the generic word "obstacle" to avoid jargon.
FRIENDLY_LABELS = {
    "person", "car", "bicycle", "motorcycle", "bus", "truck",
    "door", "stairs", "step", "chair", "table", "bench", "pole", "wall",
    "tree", "dog", "trash can", "sign", "curb",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Detection:
    label: str
    distance: float
    position: Position
    confidence: float = 1.0

@dataclass(frozen=True)
class Decision:
    action: Action
    distance: float
    direction: str            # center | left | right | both | none
    suggested_move: Move
    label: str

# ---------------------------------------------------------------------------
# Decision layer — pure, deterministic
# ---------------------------------------------------------------------------
def _opposite(p: Position) -> Move:
    return {"center": "left", "left": "right", "right": "left"}[p]

def _threat(d: Detection) -> float:
    """Higher score = more dangerous. Closer + center-position dominates."""
    weight = {"center": 1.0, "left": 0.6, "right": 0.6}[d.position]
    return weight / max(d.distance, 0.1)

def _both_sides_blocked(objs: list[Detection]) -> bool:
    """Stop if forward path is fully boxed in within SLOW range."""
    left_b   = any(o.position == "left"   and o.distance < SLOW_DISTANCE for o in objs)
    right_b  = any(o.position == "right"  and o.distance < SLOW_DISTANCE for o in objs)
    center_b = any(o.position == "center" and o.distance < SLOW_DISTANCE for o in objs)
    return left_b and right_b and center_b

def decide(objects) -> Decision:
    """Perception -> Decision. Accepts dicts or Detection objects."""
    dets: list[Detection] = []
    for o in objects:
        if isinstance(o, Detection):
            d = o
        else:
            d = Detection(
                label=o["label"],
                distance=float(o["distance"]),
                position=o["position"],
                confidence=float(o.get("confidence", 1.0)),
            )
        if d.confidence >= CONFIDENCE_FLOOR:
            dets.append(d)

    if not dets:
        return Decision("CLEAR", 0.0, "none", "forward", "")

    # Boxed-in case takes precedence over single-object logic
    if _both_sides_blocked(dets):
        nearest = min(dets, key=lambda x: x.distance)
        return Decision("STOP", nearest.distance, "both", "stop", nearest.label)

    target = max(dets, key=_threat)
    d, pos, label = target.distance, target.position, target.label

    if d < STOP_DISTANCE:
        return Decision("STOP", d, pos, "stop", label)
    if d < SLOW_DISTANCE:
        return Decision("SLOW", d, pos, _opposite(pos), label)
    if d < AVOID_DISTANCE:
        return Decision("AVOID", d, pos, _opposite(pos), label)
    if d < INFO_DISTANCE and pos == "center":
        return Decision("INFO", d, pos, "forward", label)
    return Decision("CLEAR", 0.0, "none", "forward", "")

# ---------------------------------------------------------------------------
# Phrasing layer — fixed templates, no LLM
# ---------------------------------------------------------------------------
def _friendly(label: str) -> str:
    return label if label in FRIENDLY_LABELS else "obstacle"

def _fmt(d: float) -> str:
    """One decimal, but drop the .0 on whole numbers."""
    return f"{int(round(d))}" if abs(d - round(d)) < 0.05 else f"{d:.1f}"

def speak(decision: Decision) -> str:
    a, d, dirn, mv, label = (
        decision.action, decision.distance, decision.direction,
        decision.suggested_move, decision.label,
    )
    name = _friendly(label).capitalize()

    if a == "STOP" and dirn == "both":
        return "Path blocked on both sides. Stop."
    if a == "STOP" and dirn == "center":
        return f"{name} very close ahead. Stop immediately."
    if a == "STOP":
        return f"{name} very close on your {dirn}. Stop."
    if a == "SLOW" and dirn == "center":
        return f"{name} ahead in {_fmt(d)} meters. Move slightly {mv}."
    if a == "SLOW":
        return f"{name} on your {dirn} in {_fmt(d)} meters. Move slightly {mv}."
    if a == "AVOID" and dirn == "center":
        return f"{name} ahead in {_fmt(d)} meters. Prepare to move {mv}."
    if a == "AVOID":
        return f"{name} on your {dirn} in {_fmt(d)} meters. Prepare to move {mv}."
    if a == "INFO":
        return f"{name} ahead in {_fmt(d)} meters."
    return "Clear path ahead. Continue forward."

# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------
def navigate(objects) -> str:
    """One-shot: perception JSON -> spoken instruction."""
    return speak(decide(objects))

# ---------------------------------------------------------------------------
# OpenAI prompt — for OFFLINE template work or rare-scene fallback ONLY.
# Never call this from the real-time loop.
# ---------------------------------------------------------------------------
OPENAI_SYSTEM_PROMPT = """\
You are the voice of a navigation aid for visually impaired users. You will
receive a JSON object describing one navigation decision and must return
exactly one short spoken instruction.

HARD RULES (any violation is a safety failure):
- Output ONE sentence, max 12 words. No preamble, no quotes, no explanation.
- Imperative tone only: Stop, Move, Continue, Turn, Prepare.
- No hedging words: never use "maybe", "possibly", "I think", "it seems".
- Always include the distance (meters) and direction when both are present.
- Use the friendly label when recognizable (person, car, door, stairs, ...);
  otherwise say "obstacle".
- Match the phrasing patterns below verbatim. Determinism > flair.

PHRASING PATTERNS:
  STOP / both sides:    "Path blocked on both sides. Stop."
  STOP / center:        "<Name> very close ahead. Stop immediately."
  STOP / left | right:  "<Name> very close on your <dir>. Stop."
  SLOW / center:        "<Name> ahead in <d> meters. Move slightly <move>."
  SLOW / left | right:  "<Name> on your <dir> in <d> meters. Move slightly <move>."
  AVOID / center:       "<Name> ahead in <d> meters. Prepare to move <move>."
  AVOID / left | right: "<Name> on your <dir> in <d> meters. Prepare to move <move>."
  INFO:                 "<Name> ahead in <d> meters."
  CLEAR:                "Clear path ahead. Continue forward."

INPUT JSON FIELDS:
  action:         STOP | SLOW | AVOID | INFO | CLEAR
  distance:       float (meters)
  direction:      center | left | right | both | none
  suggested_move: stop | left | right | forward
  label:          string
"""

def openai_user_payload(decision: Decision) -> str:
    return json.dumps({
        "action":         decision.action,
        "distance":       round(decision.distance, 1),
        "direction":      decision.direction,
        "suggested_move": decision.suggested_move,
        "label":          decision.label,
    })

def speak_with_llm(decision: Decision, client) -> str:
    """Optional fallback path. Use temperature=0. Not for the real-time loop."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=40,
        messages=[
            {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
            {"role": "user",   "content": openai_user_payload(decision)},
        ],
    )
    return resp.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# Self-test: 12 canonical input -> output mappings
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cases = [
        # 1. Very close center -> immediate stop
        ([{"label": "wall", "distance": 0.3, "position": "center"}],
         "Wall very close ahead. Stop immediately."),
        # 2. Close center -> slow + suggest left
        ([{"label": "chair", "distance": 1.2, "position": "center"}],
         "Chair ahead in 1.2 meters. Move slightly left."),
        # 3. Close left -> move right
        ([{"label": "table", "distance": 1.0, "position": "left"}],
         "Table on your left in 1 meters. Move slightly right."),
        # 4. Close right -> move left
        ([{"label": "person", "distance": 1.4, "position": "right"}],
         "Person on your right in 1.4 meters. Move slightly left."),
        # 5. Mid-range center -> prepare to avoid
        ([{"label": "pole", "distance": 2.5, "position": "center"}],
         "Pole ahead in 2.5 meters. Prepare to move left."),
        # 6. Far center -> informational
        ([{"label": "door", "distance": 4.0, "position": "center"}],
         "Door ahead in 4 meters."),
        # 7. Empty scene -> clear
        ([], "Clear path ahead. Continue forward."),
        # 8. Multiple objects -> closest center wins
        ([{"label": "table", "distance": 2.5, "position": "right"},
          {"label": "chair", "distance": 1.2, "position": "center"}],
         "Chair ahead in 1.2 meters. Move slightly left."),
        # 9. Both sides + center blocked -> stop
        ([{"label": "wall",  "distance": 1.0, "position": "left"},
          {"label": "wall",  "distance": 1.0, "position": "right"},
          {"label": "chair", "distance": 1.2, "position": "center"}],
         "Path blocked on both sides. Stop."),
        # 10. Unknown label -> "obstacle"
        ([{"label": "fire hydrant", "distance": 1.0, "position": "center"}],
         "Obstacle ahead in 1 meters. Move slightly left."),
        # 11. Low confidence ignored
        ([{"label": "chair", "distance": 1.0, "position": "center", "confidence": 0.2}],
         "Clear path ahead. Continue forward."),
        # 12. Far side object -> ignored (not in center, beyond AVOID)
        ([{"label": "bench", "distance": 4.0, "position": "right"}],
         "Clear path ahead. Continue forward."),
    ]

    width = max(len(e) for _, e in cases)
    passed = 0
    for i, (inp, expected) in enumerate(cases, 1):
        got = navigate(inp)
        ok = got == expected
        passed += ok
        tag = "OK  " if ok else "FAIL"
        print(f"{i:2d} {tag}  {expected:<{width}}")
        if not ok:
            print(f"        got:      {got}")
    print(f"\n{passed}/{len(cases)} cases passed.")
