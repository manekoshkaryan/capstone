"""
Shared voice navigation context.

Builds the compact, conversation-friendly snapshot consumed by:

* the local rule-based ConversationHandler (in conversation.py), and
* the OpenAI Realtime voice loop (sent over the data channel as a
  read-only system note).

Both loops MUST see the same shape so behaviour stays consistent. The LLM
side is read-only — it never decides movement. Safety decisions live in
SpeechPolicy + CommandValidator.

The build is throttled (default >=1s between rebuilds) so we don't flood
the data channel with raw YOLO spam.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

from speech.speech_policy import NavigationContext


def _bearing(box, frame_w: int) -> str:
    """left / right / front classification for a single bounding box."""
    if not box or frame_w <= 0:
        return "front"
    x1, _, x2, _ = box[:4]
    cx = (float(x1) + float(x2)) / 2.0
    rel = cx / float(frame_w)
    if rel < 0.40:
        return "left"
    if rel > 0.60:
        return "right"
    return "front"


def _obj_distance(obj) -> Optional[float]:
    d = getattr(obj, "nearest_dist_m", None)
    if d is None:
        d = getattr(obj, "distance_m", None)
    return None if d is None else float(d)


def _obj_summary(obj, frame_w: int) -> Dict[str, Any]:
    return {
        "label": getattr(obj, "label", "object"),
        "distance_m": (round(_obj_distance(obj), 2)
                       if _obj_distance(obj) is not None else None),
        "bearing": _bearing(getattr(obj, "box", None), frame_w),
    }


class VoiceContextProvider:
    """Throttled builder. Reads pipeline state under its own lock."""

    def __init__(self, pipeline, min_interval_s: float = 1.0,
                 max_per_side: int = 4):
        self._pipeline = pipeline
        self._min_interval_s = float(min_interval_s)
        self._max_per_side = int(max_per_side)
        self._lock = threading.Lock()
        self._last_ts: float = 0.0
        self._last_ctx: Dict[str, Any] = self._empty()
        self._last_safety_command: str = ""
        self._last_safety_ts: float = 0.0

    # -------------------------------------------------------------- safety
    def record_safety_command(self, action: str, text: str = "") -> None:
        """Called by pipeline whenever SafetyCommand is emitted."""
        action = (action or "").strip().upper()
        if not action:
            return
        with self._lock:
            self._last_safety_command = action
            self._last_safety_ts = time.time()
            # invalidate cached context so the next read picks the new value
            self._last_ts = 0.0

    @property
    def last_safety_command(self) -> str:
        with self._lock:
            return self._last_safety_command

    # -------------------------------------------------------------- build
    def get(self, force: bool = False) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            if not force and (now - self._last_ts) < self._min_interval_s:
                return dict(self._last_ctx)
        ctx = self._build()
        with self._lock:
            self._last_ts = now
            self._last_ctx = ctx
        return dict(ctx)

    # -------------------------------------------------------------- helpers
    def _empty(self) -> Dict[str, Any]:
        return {
            "front_path_clear": False,
            "left_clear": False,
            "right_clear": False,
            "closest_obstacle": None,
            "objects_left": [],
            "objects_right": [],
            "objects_front": [],
            "last_safety_command": "",
            "ts": time.time(),
        }

    def _build(self) -> Dict[str, Any]:
        pipe = self._pipeline
        cfg = pipe._config
        fw = cfg.frame_width
        min_go = cfg.guidance_min_go_distance_m

        # ---- free-space derived flags --------------------------------
        with pipe._state._lock:
            fs = pipe._state.free_space
        nav_ctx: Optional[NavigationContext] = None
        if fs is not None:
            nav_ctx = NavigationContext.from_free_space(
                fs, [],
                min_go_distance_m=min_go,
                stop_distance_m=cfg.guidance_stop_distance_m,
            )

        if nav_ctx is not None:
            front_clear = bool(nav_ctx.front_path_clear)
            left_clear = bool(nav_ctx.left_clear_m >= min_go)
            right_clear = bool(nav_ctx.right_clear_m >= min_go)
        else:
            front_clear = False
            left_clear = False
            right_clear = False

        # ---- bucket stable objects by side ---------------------------
        try:
            stable = pipe._stable_objects()
        except Exception:
            stable = []
        left, right, front = [], [], []
        closest: Optional[Dict[str, Any]] = None
        closest_d = float("inf")

        for obj in stable:
            d = _obj_distance(obj)
            entry = _obj_summary(obj, fw)
            side = entry["bearing"]
            if side == "left":
                left.append(entry)
            elif side == "right":
                right.append(entry)
            else:
                front.append(entry)
            if d is not None and d < closest_d:
                closest_d = d
                closest = entry

        for bucket in (left, right, front):
            bucket.sort(key=lambda e: (e["distance_m"] is None,
                                       e["distance_m"] or 1e9))
            del bucket[self._max_per_side:]

        with self._lock:
            last_cmd = self._last_safety_command

        return {
            "front_path_clear": front_clear,
            "left_clear": left_clear,
            "right_clear": right_clear,
            "closest_obstacle": closest,
            "objects_left": left,
            "objects_right": right,
            "objects_front": front,
            "last_safety_command": last_cmd,
            "ts": time.time(),
        }
