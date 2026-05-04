"""LLM-based navigation guidance engine.

Watches the perception pipeline for meaningful scene changes and issues
async GPT calls to generate natural-language navigation instructions.
Safety-critical stops bypass GPT entirely.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Callable, List, Optional

from navigation.guidance_schema import (
    FreeSectors, GuidanceResponse, ObstacleInfo, SceneContext,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a real-time navigation assistant for a visually impaired person walking with a phone camera.

You receive structured sensor data: nearby obstacles with distances/bearings and free-path sector clearances.

Your job: decide the safest immediate action and produce a short spoken instruction.

Rules for the `speech` field:
- Imperative sentence, ≤10 words
- No filler ("please", "you should", "it seems")
- Round distances to nearest meter
- DEFAULT: mention only the SINGLE closest obstacle + navigation direction
- EXCEPTION: if user_query contains "around" or "what do you see" → list the 3 closest obstacles by distance, then give direction

Rules for `action`:
- STOP: center is blocked and no safe alternative
- GO_STRAIGHT: center clear, no hazard
- BEAR_LEFT / BEAR_RIGHT: gentle course correction
- GO_LEFT / GO_RIGHT: hard turn required
- WAIT: moving obstacle crossing path

Rules for `urgency`:
- critical: obstacle <1m or no escape route
- warn: obstacle 1–2.5m or path narrowing
- info: open path update or distant obstacle

Examples (default — closest only):
  obstacles=[person 0.9m center, chair 1.5m left], center=0.9m, best=right
    → speech="Stop. Person ahead. Go right.", urgency=critical

  obstacles=[], center=4.5m
    → speech="Path clear. Continue.", urgency=info

  obstacles=[chair 1.8m left, laptop 2.1m right], center=3.2m, best=center
    → speech="Chair left two meters. Continue.", urgency=warn

Example (user asked "what's around me"):
  obstacles=[laptop 0.3m center, chair 1.8m left, person 3.0m right], user_query="what do you see around"
    → speech="Laptop close ahead. Chair two meters left. Person three meters right.", urgency=info

Do not repeat the last_guidance unless the situation changed significantly.
"""

_BEARING_MAP = {
    "far_left": "far_left",
    "left": "left",
    "center": "center",
    "center-left": "left",
    "center-right": "right",
    "right": "right",
    "far_right": "far_right",
}

_SECTOR_NAMES_5 = ["far_left", "left", "center", "right", "far_right"]
_SECTOR_NAMES_3 = ["left", "center", "right"]


def _sectors_from_list(sector_list: list) -> FreeSectors:
    """Convert [(name, dist), ...] from FreeSpaceFrame to FreeSectors."""
    mapping: dict[str, float] = {}
    for name, dist in sector_list:
        key = _BEARING_MAP.get(name, name)
        if key in FreeSectors.model_fields:
            mapping[key] = float(dist)
    return FreeSectors(**mapping)


def _bearing_from_rel(cx_rel: float) -> str:
    if cx_rel < 0.2:
        return "far_left"
    if cx_rel < 0.4:
        return "left"
    if cx_rel < 0.6:
        return "center"
    if cx_rel < 0.8:
        return "right"
    return "far_right"


class SceneDiff:
    """Detects meaningful changes in perception state to trigger GPT calls."""

    def __init__(
        self,
        distance_threshold_m: float = 0.5,
        periodic_interval_s: float = 5.0,
        direction_min_frames: int = 3,
    ):
        self._dist_thresh = distance_threshold_m
        self._periodic = periodic_interval_s
        self._dir_min_frames = direction_min_frames
        self._last_obstacles: dict[str, float] = {}
        self._last_best: str = "center"
        self._candidate_best: str = "center"
        self._candidate_streak: int = 0
        self._last_center_blocked: bool = False
        self._last_call_ts: float = 0.0
        self._last_new_obstacle_ts: float = 0.0
        self._last_dir_change_ts: float = 0.0
        self._min_event_gap_s: float = 2.0  # min seconds between same trigger type

    def check(
        self,
        obstacles: List[ObstacleInfo],
        best_direction: str,
        center_clear_m: float,
        stop_distance_m: float,
    ) -> Optional[str]:
        """Return trigger name if GPT should be called, else None."""
        now = time.time()
        # Use hysteresis band: blocked when < stop_dist, only clears when > stop_dist + 0.2m
        center_blocked = center_clear_m < stop_distance_m
        obs_map = {o.label + o.bearing: o.distance_m for o in obstacles}

        trigger = None

        # New or significantly closer obstacle
        if (now - self._last_new_obstacle_ts) >= self._min_event_gap_s:
            for key, dist in obs_map.items():
                prev = self._last_obstacles.get(key)
                if prev is None:
                    trigger = "new_obstacle"
                    break
                if prev - dist > self._dist_thresh:
                    trigger = "new_obstacle"
                    break

        # Path blocked / unblocked
        if center_blocked and not self._last_center_blocked:
            trigger = "path_blocked"
        elif not center_blocked and self._last_center_blocked:
            trigger = "path_cleared"

        # Direction changed — require N consecutive frames to filter noise
        if best_direction != self._last_best and trigger is None:
            if best_direction == self._candidate_best:
                self._candidate_streak += 1
            else:
                self._candidate_best = best_direction
                self._candidate_streak = 1
            if (self._candidate_streak >= self._dir_min_frames
                    and (now - self._last_dir_change_ts) >= self._min_event_gap_s):
                trigger = "direction_changed"
                self._candidate_streak = 0
        else:
            self._candidate_streak = 0

        # Periodic heartbeat
        if trigger is None and (now - self._last_call_ts) >= self._periodic:
            trigger = "periodic_update"

        if trigger is not None:
            self._last_obstacles = obs_map
            self._last_best = best_direction
            self._last_center_blocked = center_blocked
            self._last_call_ts = now
            if trigger == "new_obstacle":
                self._last_new_obstacle_ts = now
            elif trigger == "direction_changed":
                self._last_dir_change_ts = now

        return trigger


class LLMGuide:
    """Async LLM guidance engine. Thread-safe. Drop-in alongside pipeline."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        stop_distance_m: float = 0.8,
        periodic_interval_s: float = 5.0,
        on_guidance: Optional[Callable[[GuidanceResponse], None]] = None,
    ):
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY not set")

        from openai import OpenAI
        self._client = OpenAI(api_key=key)
        self._model = model
        self._stop_dist = stop_distance_m
        self._on_guidance = on_guidance

        self._diff = SceneDiff(periodic_interval_s=periodic_interval_s)
        self._last_guidance: Optional[GuidanceResponse] = None
        self._last_speech: str = ""
        self._pending = False
        self._backoff_until: float = 0.0
        self._backoff_s: float = 0.0
        self._last_stop_ts: float = 0.0
        self._prev_center_clear: float = float("inf")
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None

        logger.info(f"LLMGuide ready — model={model}")

    @property
    def latest(self) -> Optional[GuidanceResponse]:
        with self._lock:
            return self._last_guidance

    def on_scene_update(
        self,
        free_space_frame,
        tracked_dicts: List[dict],
        user_query: Optional[str] = None,
    ):
        """Called every navigation frame. Triggers GPT only on meaningful change."""
        sectors = free_space_frame.sectors if free_space_frame else []
        center_clear = free_space_frame.center_clear_m if free_space_frame else 0.0
        best_idx = free_space_frame.best_sector_idx if free_space_frame else -1
        best_name = sectors[best_idx].name if 0 <= best_idx < len(sectors) else "center"

        # Safety bypass: STOP only when obstacle freshly enters danger zone
        # or is rapidly approaching. Skips no-data frames (0.0m) and uses
        # a hysteresis band so threshold oscillation doesn't re-trigger.
        if center_clear < self._stop_dist:
            if center_clear == 0.0:
                # No depth data yet — skip entirely
                return
            prev = self._prev_center_clear
            # Require prev to be above (stop_dist + 0.2m) to count as "freshly crossed"
            # — prevents re-triggering when bouncing just above/below threshold
            safe_threshold = self._stop_dist + 0.20
            freshly_crossed = prev >= safe_threshold
            rapidly_closer = (prev - center_clear) > 0.25
            logger.info(
                f"STOP-CHECK center={center_clear:.2f}m prev={prev:.2f}m "
                f"freshly_crossed={freshly_crossed} rapidly_closer={rapidly_closer} "
                f"(delta={prev-center_clear:.2f}m)"
            )
            self._prev_center_clear = center_clear
            if freshly_crossed or rapidly_closer:
                logger.info(f"STOP FIRED — reason={'freshly_crossed' if freshly_crossed else 'rapidly_closer'}")
                stop_resp = GuidanceResponse(
                    action="STOP",
                    speech="Stop.",
                    urgency="critical",
                    reasoning="Obstacle entered danger zone or approaching fast.",
                )
                with self._lock:
                    self._last_guidance = stop_resp
                    self._last_speech = stop_resp.speech
                    self._last_stop_ts = time.time()
                if self._on_guidance:
                    self._on_guidance(stop_resp)
            else:
                logger.debug(f"STOP suppressed — static obstacle at {center_clear:.2f}m")
            return

        self._prev_center_clear = center_clear
        obstacles = self._build_obstacles(tracked_dicts)
        sector_list = [(s.name, s.free_distance_m) for s in sectors]
        free = _sectors_from_list(sector_list)

        trigger = user_query and "user_question" or self._diff.check(
            obstacles, best_name, center_clear, self._stop_dist
        )
        if not trigger:
            return
        logger.info(
            f"GPT TRIGGER={trigger} | center={center_clear:.2f}m best={best_name} "
            f"obstacles={[(o.label, round(o.distance_m,2), o.bearing) for o in obstacles[:3]]}"
        )

        with self._lock:
            if self._pending:
                return
            if time.time() < self._backoff_until:
                return
            self._pending = True

        ctx = SceneContext(
            obstacles=obstacles,
            free_sectors=free,
            best_direction=_BEARING_MAP.get(best_name, "center"),
            center_clear_m=center_clear,
            trigger=trigger,
            last_guidance=self._last_speech,
            user_query=user_query,
        )

        t = threading.Thread(
            target=self._call_gpt, args=(ctx,), daemon=True, name="LLMGuide-GPT"
        )
        t.start()
        self._worker_thread = t

    def _build_obstacles(self, tracked_dicts: List[dict]) -> List[ObstacleInfo]:
        obs = []
        for d in tracked_dicts:
            dist = d.get("distance_m") or d.get("distance")
            if dist is None:
                continue
            raw_bearing = d.get("bearing", "center")
            bearing = _BEARING_MAP.get(raw_bearing, "center")
            obs.append(ObstacleInfo(
                label=d.get("label", "object"),
                distance_m=float(dist),
                bearing=bearing,
                is_moving=bool(d.get("is_moving", False)),
            ))
        # Sort by distance — GPT sees closest first
        obs.sort(key=lambda o: o.distance_m)
        return obs[:6]  # cap context size

    def _call_gpt(self, ctx: SceneContext):
        try:
            t0 = time.perf_counter()
            response = self._client.beta.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": ctx.model_dump_json()},
                ],
                response_format=GuidanceResponse,
                temperature=0.2,
                max_tokens=150,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            result: GuidanceResponse = response.choices[0].message.parsed
            logger.info(
                f"GPT {latency_ms:.0f}ms | trigger={ctx.trigger} | "
                f"action={result.action} urgency={result.urgency} | "
                f"speech={result.speech!r} | reason={result.reasoning!r}"
            )
            with self._lock:
                self._last_guidance = result
                self._last_speech = result.speech
                self._pending = False
                self._backoff_s = 0.0
                self._backoff_until = 0.0
            if self._on_guidance:
                self._on_guidance(result)
        except Exception as e:
            logger.warning(f"LLMGuide GPT call failed: {e}")
            with self._lock:
                self._pending = False
                # Exponential backoff on rate-limit / quota errors
                if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
                    self._backoff_s = min(60.0, max(5.0, self._backoff_s * 2 or 5.0))
                    self._backoff_until = time.time() + self._backoff_s
                    logger.warning(f"LLMGuide backing off {self._backoff_s:.0f}s after 429")
                else:
                    self._backoff_s = 0.0
