from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Iterable
import time


class ProtocolMode(str, Enum):
    MINIMAL = "minimal"
    DESCRIPTIVE = "descriptive"
    PROACTIVE = "proactive"
    ADAPTIVE = "adaptive"


ACTION_STOP = "stop"
ACTION_DIRECTIONAL = "directional"
ACTION_CLEAR = "clear"
ACTION_INFO = "info"

URGENT_DISTANCE_M = 0.8
PROACTIVE_DISTANCE_M = 2.0

_LEFT_DIRS = {"left", "far_left", "slight_left"}
_RIGHT_DIRS = {"right", "far_right", "slight_right"}
_AHEAD_DIRS = {"center", "ahead", "front", ""}


@dataclass(frozen=True)
class PerceptionInput:
    label: str = ""
    distance_m: Optional[float] = None
    direction: str = "ahead"
    urgency: str = "info"
    front_clear: bool = True
    left_clear_m: Optional[float] = None
    right_clear_m: Optional[float] = None
    device_label: str = ""


@dataclass(frozen=True)
class ProtocolMessage:
    text: str
    protocol: str
    action_category: str
    word_count: int
    timestamp: float


@dataclass(frozen=True)
class CascadedBaselineConfig:
    asr_ms: float = 600.0
    llm_ms: float = 900.0
    tts_ms: float = 700.0


def normalize_direction(d: str) -> str:
    if not d:
        return "ahead"
    s = str(d).strip().lower()
    if s in _LEFT_DIRS:
        return "left"
    if s in _RIGHT_DIRS:
        return "right"
    if s in _AHEAD_DIRS:
        return "ahead"
    return s


def _friendly_label(label: str) -> str:
    if not label:
        return "obstacle"
    return str(label).strip().lower()


def _format_distance(d: Optional[float]) -> str:
    if d is None:
        return ""
    return f"{float(d):.1f}"


def _word_count(text: str) -> int:
    return len([w for w in str(text).split() if w])


def _has_object(p: PerceptionInput) -> bool:
    return bool(p.label) and (p.distance_m is not None)


def _opposite_direction(d: str) -> str:
    if d == "left":
        return "right"
    if d == "right":
        return "left"
    return "right"


def _suggest_move(p: PerceptionInput) -> str:
    direction = normalize_direction(p.direction)
    if direction in ("left", "right"):
        return _opposite_direction(direction)
    left = p.left_clear_m if p.left_clear_m is not None else 0.0
    right = p.right_clear_m if p.right_clear_m is not None else 0.0
    if left > right:
        return "left"
    if right > left:
        return "right"
    return "right"


def _action_for_distance(distance: Optional[float]) -> str:
    if distance is None:
        return ACTION_INFO
    if distance < URGENT_DISTANCE_M:
        return ACTION_STOP
    if distance < PROACTIVE_DISTANCE_M:
        return ACTION_DIRECTIONAL
    return ACTION_INFO


def _make(text: str, mode: ProtocolMode, action: str) -> ProtocolMessage:
    clean = " ".join(str(text).split())
    return ProtocolMessage(
        text=clean,
        protocol=mode.value,
        action_category=action,
        word_count=_word_count(clean),
        timestamp=time.time(),
    )


def render_minimal(p: PerceptionInput) -> ProtocolMessage:
    direction = normalize_direction(p.direction)
    if not _has_object(p):
        return _make("Clear path.", ProtocolMode.MINIMAL, ACTION_CLEAR)
    label = _friendly_label(p.label).capitalize()
    dist = _format_distance(p.distance_m)
    if direction == "ahead":
        text = f"{label} ahead, {dist} meters."
    else:
        text = f"{label} {direction}, {dist} meters."
    return _make(text, ProtocolMode.MINIMAL, _action_for_distance(p.distance_m))


def render_descriptive(p: PerceptionInput) -> ProtocolMessage:
    direction = normalize_direction(p.direction)
    if not _has_object(p):
        return _make("The path ahead appears clear.", ProtocolMode.DESCRIPTIVE, ACTION_CLEAR)
    label = _friendly_label(p.label)
    article = "an" if label[:1] in "aeiou" else "a"
    dist = _format_distance(p.distance_m)
    if direction == "ahead":
        text = f"There is {article} {label} ahead about {dist} meters away."
    else:
        text = f"There is {article} {label} on your {direction} about {dist} meters away."
    return _make(text, ProtocolMode.DESCRIPTIVE, _action_for_distance(p.distance_m))


def render_proactive(p: PerceptionInput) -> ProtocolMessage:
    direction = normalize_direction(p.direction)
    if not _has_object(p):
        return _make("Path ahead is clear, continue forward.", ProtocolMode.PROACTIVE, ACTION_CLEAR)
    distance = p.distance_m or 0.0
    if direction == "ahead":
        if distance < URGENT_DISTANCE_M:
            return _make("Obstacle ahead, stop.", ProtocolMode.PROACTIVE, ACTION_STOP)
        move = _suggest_move(p)
        return _make(
            f"Obstacle ahead, move slightly {move}.",
            ProtocolMode.PROACTIVE,
            ACTION_DIRECTIONAL,
        )
    move = _opposite_direction(direction)
    if distance < URGENT_DISTANCE_M:
        return _make(
            f"Obstacle on your {direction}, stop.",
            ProtocolMode.PROACTIVE,
            ACTION_STOP,
        )
    return _make(
        f"Obstacle on your {direction}, move slightly {move}.",
        ProtocolMode.PROACTIVE,
        ACTION_DIRECTIONAL,
    )


def render_adaptive(p: PerceptionInput) -> ProtocolMessage:
    direction = normalize_direction(p.direction)
    if not _has_object(p):
        return _make("Path ahead is clear.", ProtocolMode.ADAPTIVE, ACTION_CLEAR)
    distance = p.distance_m or 0.0
    if distance < URGENT_DISTANCE_M:
        return _make("Stop. Obstacle ahead.", ProtocolMode.ADAPTIVE, ACTION_STOP)
    if distance < PROACTIVE_DISTANCE_M:
        if direction == "ahead":
            move = _suggest_move(p)
            return _make(
                f"Obstacle ahead, move {move}.",
                ProtocolMode.ADAPTIVE,
                ACTION_DIRECTIONAL,
            )
        move = _opposite_direction(direction)
        return _make(
            f"Obstacle on your {direction}, move {move}.",
            ProtocolMode.ADAPTIVE,
            ACTION_DIRECTIONAL,
        )
    label = _friendly_label(p.label).capitalize()
    dist = _format_distance(p.distance_m)
    if direction == "ahead":
        text = f"{label} ahead, {dist} meters."
    else:
        text = f"{label} {direction}, {dist} meters."
    return _make(text, ProtocolMode.ADAPTIVE, ACTION_INFO)


_RENDERERS = {
    ProtocolMode.MINIMAL: render_minimal,
    ProtocolMode.DESCRIPTIVE: render_descriptive,
    ProtocolMode.PROACTIVE: render_proactive,
    ProtocolMode.ADAPTIVE: render_adaptive,
}


def coerce_mode(mode: Any) -> ProtocolMode:
    if isinstance(mode, ProtocolMode):
        return mode
    if isinstance(mode, str):
        try:
            return ProtocolMode(mode.strip().lower())
        except ValueError:
            return ProtocolMode.ADAPTIVE
    return ProtocolMode.ADAPTIVE


def render(mode: Any, perception: PerceptionInput) -> ProtocolMessage:
    fn = _RENDERERS.get(coerce_mode(mode), render_adaptive)
    return fn(perception)


def render_all(perception: PerceptionInput) -> Dict[str, ProtocolMessage]:
    return {m.value: _RENDERERS[m](perception) for m in ProtocolMode}


def perception_from_dict(data: Dict[str, Any]) -> PerceptionInput:
    if not data:
        return PerceptionInput()
    distance = data.get("distance_m", data.get("distance"))
    distance = float(distance) if distance is not None else None
    return PerceptionInput(
        label=str(data.get("label") or data.get("class") or ""),
        distance_m=distance,
        direction=str(data.get("direction") or data.get("bearing") or "ahead"),
        urgency=str(data.get("urgency") or "info"),
        front_clear=bool(data.get("front_clear", True)),
        left_clear_m=_safe_float(data.get("left_clear_m")),
        right_clear_m=_safe_float(data.get("right_clear_m")),
        device_label=str(data.get("device_label") or ""),
    )


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def cascaded_simulated_latency_ms(
    actual_response_ms: float,
    cfg: Optional[CascadedBaselineConfig] = None,
) -> float:
    cfg = cfg or CascadedBaselineConfig()
    return float(actual_response_ms) + cfg.asr_ms + cfg.llm_ms + cfg.tts_ms


def latency_improvement_pct(actual_ms: float, baseline_ms: float) -> float:
    base = max(float(baseline_ms), 1e-6)
    return (base - float(actual_ms)) / base * 100.0


DIRECT_SYSTEM_NAME = "Direct Vision-to-Speech Protocol"
CASCADED_SYSTEM_NAME = "Cascaded ASR-LLM-TTS"


def list_modes() -> Iterable[ProtocolMode]:
    return list(ProtocolMode)
