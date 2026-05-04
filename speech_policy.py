"""
Wise speech policy + conversation protocol for assistive navigation.

The policy is the single decision point for "should the system speak right
now, and if so, what?". It governs unsolicited safety announcements; user
question handlers (`scene_description`, `find_object`, `where_is`, etc.)
bypass this layer and call `SpeechEngine.say()` directly.

Modules:
- ConversationMode / ConversationModeManager — what protocol the user is in
  (passive, continuous-guidance, minimal-alert, conversation).
- NavigationContext — perception snapshot fed into the policy each cycle.
- ObstacleMemory (the policy's `_entries`) — what we already announced and
  when, keyed by (label, bearing-bucket).
- CommandValidator — guarantees a STOP/MOVE/etc. command is consistent with
  the environment before it reaches the user.
- SpeechPolicy — combines the above and returns one PolicyDecision per cycle.

Design principle: perception != speech. We only speak when the information
changes the user's next action.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import logging
import time

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversation mode
# ---------------------------------------------------------------------------

class ConversationMode(str, Enum):
    PASSIVE = "passive"                    # default — only safety-critical updates
    CONTINUOUS_GUIDANCE = "continuous"     # walking guidance, more frequent
    MINIMAL_ALERT = "minimal"              # only critical danger
    CONVERSATION = "conversation"          # rich answers, never auto-announce

    @property
    def auto_announce_allowed(self) -> bool:
        # Whether the policy may volunteer announcements at all.
        return self in (ConversationMode.PASSIVE, ConversationMode.CONTINUOUS_GUIDANCE)


class ConversationModeManager:
    """Tracks current mode + emits short confirmations on transitions."""

    def __init__(self, initial: ConversationMode = ConversationMode.PASSIVE):
        self._mode = initial
        self._since = time.time()

    @property
    def mode(self) -> ConversationMode:
        return self._mode

    @property
    def seconds_in_mode(self) -> float:
        return time.time() - self._since

    def set(self, mode: ConversationMode) -> bool:
        if mode == self._mode:
            return False
        logger.info(f"conversation-mode: {self._mode.value} -> {mode.value}")
        self._mode = mode
        self._since = time.time()
        return True


# ---------------------------------------------------------------------------
# Risk + bearing helpers
# ---------------------------------------------------------------------------

_RISK_ORDER = {"ignore": 0, "info": 1, "warn": 2, "critical": 3}


def _action_family(action: str) -> str:
    """Group action variants so a freshly-fired BEAR_LEFT does not bypass the
    throttle of a recently-fired GO_LEFT. The caller's full action string
    survives in `PolicyDecision.action` for downstream consumers."""
    a = (action or "").upper()
    if a in ("STOP", "REVERSE"):
        return a
    if a in ("GO_LEFT", "BEAR_LEFT", "MOVE_LEFT"):
        return "LEFT"
    if a in ("GO_RIGHT", "BEAR_RIGHT", "MOVE_RIGHT"):
        return "RIGHT"
    if a in ("GO_STRAIGHT", "KEEP_STRAIGHT", "CLEAR_PATH"):
        return "STRAIGHT"
    return a or "OTHER"

_PATH_BEARINGS = ("slight_left", "center", "slight_right")
_DANGER_BEARINGS = ("left", "slight_left", "center", "slight_right", "right")
_LANDMARK_BEARINGS = ("far_left", "left", "right", "far_right")

# Labels that are "real" obstacles for which interval-based repeats make sense.
# Labels NOT in this set are treated as landmarks: announced once and not
# repeated by elapsed-time alone — only by getting closer or risk escalation.
_OBSTACLE_LABELS = {
    "person", "pedestrian", "man", "woman", "child", "cyclist",
    "car", "bus", "truck", "motorcycle", "bicycle", "vehicle",
    "stairs", "step", "curb", "ramp", "escalator",
    "door", "doorway", "gate",
    "wall", "partition", "fence", "barrier", "bollard", "pole", "pillar",
    "dog", "cat",
    "chair", "table", "bench", "sofa", "bed",
    "trash can", "shopping cart", "stroller", "wheelchair",
    "pothole", "puddle", "speed bump",
    "traffic cone", "fire hydrant",
}

_DYNAMIC_LABELS = {
    "person", "pedestrian", "man", "woman", "child", "cyclist",
    "car", "bus", "truck", "motorcycle", "bicycle", "vehicle",
    "dog", "cat",
}


def _bucket_bearing(bearing: str) -> str:
    if bearing in ("far_left", "left"):
        return "left"
    if bearing in ("far_right", "right"):
        return "right"
    return "center"


def _risk_level(dist_m: Optional[float], crit_m: float, warn_m: float, info_m: float) -> str:
    if dist_m is None:
        return "info"
    if dist_m < crit_m:
        return "critical"
    if dist_m < warn_m:
        return "warn"
    if dist_m < info_m:
        return "info"
    return "ignore"


def _is_obstacle_label(label: str) -> bool:
    if not label:
        return False
    s = label.lower()
    return any(k == s or k in s or s in k for k in _OBSTACLE_LABELS)


def _is_dynamic_label(label: str) -> bool:
    if not label:
        return False
    s = label.lower()
    return any(k == s or k in s or s in k for k in _DYNAMIC_LABELS)


# ---------------------------------------------------------------------------
# NavigationContext — perception snapshot
# ---------------------------------------------------------------------------

@dataclass
class NavigationContext:
    """Per-cycle perception snapshot the policy reasons over."""

    front_clear_m: float = 0.0
    left_clear_m: float = 0.0
    right_clear_m: float = 0.0
    nearest_global_m: float = float("inf")
    front_path_clear: bool = True
    has_open_alternative: bool = False
    detections: List[dict] = field(default_factory=list)

    @classmethod
    def from_free_space(
        cls,
        free_space,
        detections: List[dict],
        min_go_distance_m: float,
        stop_distance_m: float,
    ) -> "NavigationContext":
        """Build context from a `free_space.FreeSpaceFrame` + detection list.
        Falls back gracefully when free_space is unavailable.
        """
        if free_space is None or not getattr(free_space, "sectors", None):
            return cls(detections=list(detections))

        sectors = free_space.sectors
        n = len(sectors)
        center_idx = n // 2
        front = float(sectors[center_idx].free_distance_m)

        left_clear = max(
            (float(s.free_distance_m) for s in sectors[:center_idx]),
            default=0.0,
        )
        right_clear = max(
            (float(s.free_distance_m) for s in sectors[center_idx + 1:]),
            default=0.0,
        )
        nearest = float(getattr(free_space, "nearest_global_m", float("inf")))

        front_clear = front >= min_go_distance_m
        has_alt = (
            max(left_clear, right_clear) >= min_go_distance_m
            and front < stop_distance_m
        )
        return cls(
            front_clear_m=front,
            left_clear_m=left_clear,
            right_clear_m=right_clear,
            nearest_global_m=nearest,
            front_path_clear=front_clear,
            has_open_alternative=has_alt,
            detections=list(detections),
        )


# ---------------------------------------------------------------------------
# CommandValidator — gates STOP / MOVE_* against the environment
# ---------------------------------------------------------------------------

class CommandValidator:
    """
    Reject commands that contradict the current NavigationContext.

    Rules:
    - STOP requires the front to be blocked (or a critical near-field hit).
    - MOVE_LEFT / MOVE_RIGHT require that side to be open.
    - CLEAR_PATH should not be repeated when nothing changed; the policy
      handles that part — the validator only checks logical consistency.
    """

    def __init__(self, config):
        self.cfg = config
        self.stop_distance_m: float = getattr(config, "guidance_stop_distance_m", 0.8)
        self.min_go_distance_m: float = getattr(config, "guidance_min_go_distance_m", 1.6)

    def validate(self, action: str, context: NavigationContext) -> Tuple[bool, str]:
        """Return (ok, reason). Caller drops the message if ok=False."""
        a = (action or "").upper()
        c = context

        if a == "STOP":
            # STOP is justified when the front is not walkable (below the
            # min-go threshold, not just the panic-stop threshold) OR when
            # something is in critical-near range.
            front_not_walkable = c.front_clear_m < self.min_go_distance_m
            critical_near = (
                c.nearest_global_m < self.stop_distance_m and not c.front_path_clear
            )
            if not (front_not_walkable or critical_near):
                return False, "front-clear-no-stop"
            return True, "ok"

        if a in ("MOVE_LEFT", "BEAR_LEFT", "GO_LEFT"):
            if c.left_clear_m < self.min_go_distance_m:
                return False, "left-blocked"
            return True, "ok"

        if a in ("MOVE_RIGHT", "BEAR_RIGHT", "GO_RIGHT"):
            if c.right_clear_m < self.min_go_distance_m:
                return False, "right-blocked"
            return True, "ok"

        if a in ("KEEP_STRAIGHT", "GO_STRAIGHT", "CLEAR_PATH"):
            if not c.front_path_clear:
                return False, "front-not-clear"
            return True, "ok"

        # Pass-through for non-movement utterances (descriptions, alerts).
        return True, "ok"


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@dataclass
class _Entry:
    label: str
    bearing_bucket: str
    last_dist_m: float
    last_risk: str
    last_announce_ts: float
    first_seen_ts: float
    last_seen_ts: float = 0.0
    repeat_count: int = 0
    is_landmark: bool = False


@dataclass
class PolicyDecision:
    speak: bool
    text: str
    urgency: str
    reason: str
    detection: Optional[dict] = None
    action: str = ""


class SpeechPolicy:
    """Decides whether to speak about an obstacle and chooses the wording."""

    def __init__(self, config):
        self.cfg = config

        # Tunable thresholds.
        self.dist_change_threshold_m: float = getattr(
            config, "speech_dist_change_threshold_m", 0.5
        )
        self.repeat_critical_s: float = getattr(config, "speech_repeat_critical_s", 1.5)
        self.repeat_warn_s: float = getattr(config, "speech_repeat_warn_s", 5.0)
        self.repeat_info_s: float = getattr(config, "speech_repeat_info_s", 9.0)
        self.crit_dist_m: float = getattr(config, "speech_critical_dist_m", 1.0)
        self.warn_dist_m: float = getattr(config, "speech_warn_dist_m", 2.5)
        self.info_dist_m: float = getattr(config, "speech_info_dist_m", 5.0)
        self.entry_ttl_s: float = getattr(config, "speech_entry_ttl_s", 12.0)
        self.global_min_gap_s: float = getattr(config, "speech_global_min_gap_s", 0.7)

        self._entries: Dict[Tuple[str, str], _Entry] = {}
        self._last_global_ts: float = 0.0
        self._validator = CommandValidator(config)

    def reset(self) -> None:
        self._entries.clear()
        self._last_global_ts = 0.0

    # -- selection ----------------------------------------------------------

    def _gc(self, now: float) -> None:
        # Keep an entry alive as long as the object is still being seen,
        # even if we stayed silent. Otherwise a stationary shelter would
        # re-announce after TTL just because we haven't talked about it.
        ttl = self.entry_ttl_s
        for k, e in list(self._entries.items()):
            last_touch = max(e.last_seen_ts, e.last_announce_ts)
            if now - last_touch > ttl:
                del self._entries[k]

    def _select_target(self, detections: List[dict]) -> Optional[dict]:
        """Pick the single most relevant obstacle to announce.

        Passive-mode core rule: only OBSTACLE / HAZARD kinds are eligible.
        Pure landmarks (shelter, sign, bench-on-side, building, etc.) are
        excluded so the system stops describing background scenery.
        """
        from object_classifier import classify, ObjectKind

        scored = []
        for d in detections:
            dist = d.get("distance_m")
            bearing = d.get("bearing", "center")
            urgency = d.get("urgency", "info")
            if dist is None or urgency == "beyond":
                continue

            kind = classify(d, hazard_distance_m=self.crit_dist_m)
            if kind in (ObjectKind.LANDMARK, ObjectKind.BACKGROUND):
                # Landmarks are not auto-announced. The user can ask
                # ("what's on my left?") to get them.
                continue

            in_path = bearing in _PATH_BEARINGS
            close_lateral = dist < self.warn_dist_m and bearing in _DANGER_BEARINGS
            if not (in_path or close_lateral):
                continue

            cls = str(d.get("class", "")).lower()
            crossing_bonus = 0.4 if d.get("is_crossing") else 0.0
            person_bonus = 0.2 if _is_dynamic_label(cls) else 0.0
            risk_bonus = _RISK_ORDER.get(urgency, 0) * 0.05
            hazard_bonus = 0.5 if kind == ObjectKind.HAZARD else 0.0
            score = float(dist) - crossing_bonus - person_bonus - risk_bonus - hazard_bonus
            d["_kind"] = kind.value  # forwarded for the caller's use
            scored.append((score, d))
        if not scored:
            return None
        scored.sort(key=lambda x: x[0])
        return scored[0][1]

    def _repeat_interval(self, risk: str) -> float:
        if risk == "critical":
            return self.repeat_critical_s
        if risk == "warn":
            return self.repeat_warn_s
        return self.repeat_info_s

    def _guidance_interval(
        self,
        action: str,
        urgency: str,
        mode: ConversationMode,
    ) -> float:
        """CONTINUOUS_GUIDANCE shortens repeats so the user actually gets
        walking-paced help. Other modes use the same intervals as before.
        """
        if mode != ConversationMode.CONTINUOUS_GUIDANCE:
            return self._repeat_interval(urgency)
        if urgency == "critical":
            return getattr(self.cfg, "continuous_danger_interval_s", 1.5)
        a = (action or "").upper()
        directional = a in (
            "BEAR_LEFT", "BEAR_RIGHT",
            "GO_LEFT", "GO_RIGHT",
            "MOVE_LEFT", "MOVE_RIGHT",
            "REVERSE",
        )
        if directional:
            return getattr(self.cfg, "continuous_directional_interval_s", 3.0)
        # GO_STRAIGHT / KEEP_STRAIGHT / CLEAR_PATH and the like.
        return getattr(self.cfg, "continuous_clear_interval_s", 6.0)

    def _format(self, det: dict) -> str:
        from speech_engine import _format_spp
        return _format_spp(det) or ""

    # -- public --------------------------------------------------------------

    def peek_guidance(
        self,
        guidance,
        context: Optional[NavigationContext] = None,
        mode: ConversationMode = ConversationMode.PASSIVE,
        now: Optional[float] = None,
    ) -> Tuple[PolicyDecision, Optional[callable]]:
        """Pure decision for a GuidanceCommand. Returns (decision, commit).
        Caller invokes commit() if and only if it picks this decision over
        any sibling (e.g. detection) — that way state is updated only for
        the utterance actually spoken.
        """
        if guidance is None:
            return PolicyDecision(False, "", "info", "no-guidance"), None

        now = now if now is not None else time.time()
        action = getattr(guidance, "action", "")
        urgency = getattr(guidance, "urgency", "info")
        text = getattr(guidance, "text", "")

        # Passive mode: only warn/critical guidance auto-announces. Info-
        # level guidance ("Path ahead is clear.", "Bear left.") is silent
        # unless the user is in CONTINUOUS_GUIDANCE mode.
        if mode == ConversationMode.PASSIVE and urgency == "info":
            return PolicyDecision(False, "", urgency, "passive-info-guidance", action=action), None

        # Mode gate (critical may interrupt MINIMAL_ALERT / CONVERSATION).
        if not mode.auto_announce_allowed and urgency != "critical":
            return PolicyDecision(False, "", urgency, f"mode:{mode.value}", action=action), None

        # Validate the action against current NavigationContext.
        if context is not None:
            ok, why = self._validator.validate(action, context)
            if not ok:
                return PolicyDecision(False, "", urgency, f"validator:{why}", action=action), None

        # Throttle by direction family so GO_LEFT/BEAR_LEFT share state
        # (otherwise depth wobble flips the action label and bypasses repeat
        # cooldown — produced the BEAR_LEFT/GO_LEFT/BEAR_LEFT spam in logs).
        family = _action_family(action)
        key = ("__guidance__", family)
        entry = self._entries.get(key)

        if entry is None:
            speak, reason = True, "guidance-new"
        else:
            elapsed = now - entry.last_announce_ts
            interval = self._guidance_interval(action, urgency, mode)
            if action == "STOP" and mode != ConversationMode.CONTINUOUS_GUIDANCE:
                # Standard cooldown for STOP outside CONTINUOUS_GUIDANCE so
                # we do not nag with stop announcements when the user is
                # not actively walking. In continuous mode we want fresh
                # critical alerts at the configured danger interval.
                interval = max(interval, getattr(self.cfg, "guidance_stop_cooldown_s", 12.0))
            # CONTINUOUS_GUIDANCE STOP-spam cap. After N consecutive STOPs
            # without any change in action, back off to a long cooldown so
            # the user (who has already stopped) is not nagged forever.
            # Reset happens automatically when `action` changes — entry key
            # is keyed on action, so a different action gets its own entry
            # with repeat_count=0.
            if action == "STOP" and mode == ConversationMode.CONTINUOUS_GUIDANCE:
                cap = int(getattr(self.cfg, "continuous_stop_repeat_cap", 3))
                if entry.repeat_count >= cap:
                    interval = max(
                        interval,
                        float(getattr(self.cfg, "continuous_stop_backoff_s", 12.0)),
                    )
            risk_up = _RISK_ORDER.get(urgency, 0) > _RISK_ORDER.get(entry.last_risk, 0)
            if risk_up:
                speak, reason = True, "guidance-risk-up"
            elif elapsed >= interval:
                speak, reason = True, "guidance-interval"
            else:
                speak, reason = False, "guidance-throttled"

        if not speak:
            return PolicyDecision(False, "", urgency, reason, action=action), None

        if urgency != "critical" and (now - self._last_global_ts) < self.global_min_gap_s:
            return PolicyDecision(False, "", urgency, "global-min-gap", action=action), None

        if entry is not None and entry.repeat_count > 0 and action == "STOP":
            text = "Still no path."

        decision = PolicyDecision(True, text, urgency, reason, action=action)

        def _commit():
            prev_count = (entry.repeat_count + 1) if entry else 0
            self._entries[key] = _Entry(
                label="__guidance__",
                bearing_bucket=action,
                last_dist_m=0.0,
                last_risk=urgency,
                last_announce_ts=now,
                first_seen_ts=entry.first_seen_ts if entry else now,
                last_seen_ts=now,
                repeat_count=prev_count,
            )
            self._last_global_ts = now

        return decision, _commit

    def evaluate_guidance(self, guidance, context=None, mode=ConversationMode.PASSIVE, now=None) -> PolicyDecision:
        decision, commit = self.peek_guidance(guidance, context, mode, now)
        if commit is not None:
            commit()
        return decision

    def peek_detection(
        self,
        detections: List[dict],
        context: Optional[NavigationContext] = None,
        mode: ConversationMode = ConversationMode.PASSIVE,
        now: Optional[float] = None,
    ) -> Tuple[PolicyDecision, Optional[callable]]:
        """Pure decision for a detection list. Returns (decision, commit).
        Commit is invoked only by the chosen utterance — keeps state clean
        when this loses to a sibling guidance decision in the same cycle.
        """
        now = now if now is not None else time.time()
        self._gc(now)

        target = self._select_target(detections)
        if target is None:
            return PolicyDecision(False, "", "info", "no-target"), None

        cls = str(target.get("class", "obstacle")).lower()
        bearing = str(target.get("bearing", "center"))
        bucket = _bucket_bearing(bearing)
        dist = float(target.get("distance_m", 0.0))
        risk = _risk_level(dist, self.crit_dist_m, self.warn_dist_m, self.info_dist_m)
        if risk == "ignore":
            return PolicyDecision(False, "", "info", "too-far", target), None

        if not mode.auto_announce_allowed and risk != "critical":
            return PolicyDecision(False, "", risk, f"mode:{mode.value}", target), None

        front_clear_now = bool(context is not None and context.front_path_clear)
        is_landmark = (
            risk != "critical"
            and bearing in _LANDMARK_BEARINGS
            and (not _is_obstacle_label(cls) or front_clear_now)
        )

        if risk == "critical":
            key = ("__critical__", bucket)
        else:
            key = (cls, bucket)
        entry = self._entries.get(key)

        if entry is None:
            speak, reason = True, "new"
        else:
            elapsed = now - entry.last_announce_ts
            risk_up = _RISK_ORDER[risk] > _RISK_ORDER.get(entry.last_risk, 0)
            closer = (entry.last_dist_m - dist) >= self.dist_change_threshold_m
            interval = self._repeat_interval(risk)
            # Critical-detection spam cap. Real session log produced 9
            # repeats of "Person very close left." for a stationary person
            # the user already heard about. After the cap, fall back to a
            # long cooldown until either the obstacle gets meaningfully
            # closer or risk steps up.
            if risk == "critical":
                cap = int(getattr(self.cfg, "continuous_stop_repeat_cap", 3))
                if entry.repeat_count >= cap:
                    interval = max(
                        interval,
                        float(getattr(self.cfg, "continuous_stop_backoff_s", 12.0)),
                    )
            if risk_up:
                speak, reason = True, "risk-up"
            elif closer:
                speak, reason = True, "closer"
            elif is_landmark:
                speak, reason = False, "landmark-quiet"
            elif elapsed >= interval:
                speak, reason = True, "interval"
            else:
                speak, reason = False, "throttled"

        if not speak:
            return PolicyDecision(False, "", risk, reason, target), None

        if risk != "critical" and (now - self._last_global_ts) < self.global_min_gap_s:
            return PolicyDecision(False, "", risk, "global-min-gap", target), None

        det_for_phrase = dict(target)
        det_for_phrase["is_new"] = True
        det_for_phrase["urgency"] = "info" if is_landmark else risk
        det_for_phrase["repeat_count"] = entry.repeat_count if entry else 0
        text = self._format(det_for_phrase)
        if not text:
            return PolicyDecision(False, "", risk, "empty-phrase", target), None

        action = ""
        upper = text.upper()
        if upper.startswith("STOP"):
            action = "STOP"
        elif "STEP LEFT" in upper or "MOVE LEFT" in upper or "GO LEFT" in upper:
            action = "MOVE_LEFT"
        elif "STEP RIGHT" in upper or "MOVE RIGHT" in upper or "GO RIGHT" in upper:
            action = "MOVE_RIGHT"
        if action and context is not None:
            ok, why = self._validator.validate(action, context)
            if not ok:
                if action == "STOP":
                    bearing_str = _short_bearing(bearing)
                    text = f"{cls.capitalize()} {bearing_str}."
                    det_for_phrase["urgency"] = "info"
                    action = ""
                else:
                    return PolicyDecision(False, "", risk, f"validator:{why}", target), None

        urgency_out = det_for_phrase.get("urgency", risk)
        decision = PolicyDecision(True, text, urgency_out, reason, target, action=action)

        # Capture entry state at decision time so commit() is consistent
        # even if the cycle's now/state changes by then.
        prev_first = entry.first_seen_ts if entry else now
        prev_count = (entry.repeat_count + 1) if entry else 0

        def _commit():
            self._entries[key] = _Entry(
                label=cls,
                bearing_bucket=bucket,
                last_dist_m=dist,
                last_risk=risk,
                last_announce_ts=now,
                first_seen_ts=prev_first,
                last_seen_ts=now,
                repeat_count=prev_count,
                is_landmark=is_landmark,
            )
            self._last_global_ts = now

        return decision, _commit

    def evaluate(
        self,
        detections: List[dict],
        context: Optional[NavigationContext] = None,
        mode: ConversationMode = ConversationMode.PASSIVE,
        now: Optional[float] = None,
    ) -> PolicyDecision:
        decision, commit = self.peek_detection(detections, context, mode, now)
        if commit is not None:
            commit()
        # Refresh last_seen for the picked target's existing entry even when
        # not committing (so silent-still-seen landmarks don't expire).
        self._refresh_last_seen(detections, now)
        return decision

    def _refresh_last_seen(self, detections, now: Optional[float]) -> None:
        if not detections:
            return
        now = now if now is not None else time.time()
        target = self._select_target(detections)
        if target is None:
            return
        cls = str(target.get("class", "obstacle")).lower()
        bucket = _bucket_bearing(str(target.get("bearing", "center")))
        for k in (("__critical__", bucket), (cls, bucket)):
            e = self._entries.get(k)
            if e is not None:
                e.last_seen_ts = now


def _short_bearing(bearing: str) -> str:
    return {
        "far_left": "on your hard left",
        "left": "on your left",
        "slight_left": "slightly to your left",
        "center": "ahead",
        "slight_right": "slightly to your right",
        "right": "on your right",
        "far_right": "on your hard right",
    }.get(bearing, "ahead")
