"""
Lightweight conversational responder for the navigation system.

The hard command parser in `speech_input.parse_command` only accepts a small
fixed set of phrases. Anything else used to be dropped silently, which made
voice-to-voice feel broken. This module is the fallback: it takes raw user
text and returns a short spoken reply grounded in the current perception
state (NavigationContext + tracked objects), without depending on an LLM.

Design:
- Rule-based intent classifier first (greeting / thanks / help / how-far /
  is-there-a-X / direction / path-clear).
- All numbers come from the live pipeline state — no hallucination.
- Always returns *something* so the user gets a reply.
"""

from typing import List, Optional
import logging

from speech.speech_policy import NavigationContext

logger = logging.getLogger(__name__)


_GREETINGS = ("hi", "hello", "hey", "good morning", "good afternoon", "good evening")
_THANKS = ("thank you", "thanks", "appreciate it")
_HELP_QUERIES = (
    "what can you do", "what commands", "how do you work",
    "what can you say", "help me understand", "how can you help",
)
_YES_NO_PATH = (
    "can i go", "is it safe", "is the path clear", "is the way clear",
    "anything ahead", "is it clear",
)
_HOW_FAR = ("how far", "distance to", "how close")
_EXISTENCE = (
    "is there a ", "is there an ", "is there ",
    "do you see a ", "do you see an ", "do you see ",
    "can you see a ", "can you see an ", "can you see ",
    "any ",
)
_LEFT_WORDS = ("on my left", "to my left", "to the left", "on the left", "left side")
_RIGHT_WORDS = ("on my right", "to my right", "to the right", "on the right", "right side")
_FRONT_WORDS = ("in front", "ahead", "forward")


_FILLER_PREFIXES = ("is ", "are ", "was ", "were ", "the ", "a ", "an ", "my ", "to ")


def _strip_filler(s: str) -> str:
    s = s.strip().rstrip("?.,!")
    changed = True
    while changed:
        changed = False
        for prefix in _FILLER_PREFIXES:
            if s.startswith(prefix):
                s = s[len(prefix):]
                changed = True
                break
    return s


def _first_label_word(s: str) -> str:
    s = _strip_filler(s)
    return s.split()[0] if s else ""


class ConversationHandler:
    """Generates a short reply for free-form user speech."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def respond(
        self,
        text: str,
        context: Optional[NavigationContext] = None,
    ) -> Optional[str]:
        if not text:
            return None
        s = text.strip().lower()
        s = s.rstrip("?.!,")

        # 1. Social / meta.
        if any(g == s or s.startswith(g + " ") or s.endswith(" " + g) for g in _GREETINGS):
            return "Hello."
        if any(t in s for t in _THANKS):
            return "You are welcome."
        if any(q in s for q in _HELP_QUERIES):
            return (
                "Ask me what is in front, on your left, or on your right. "
                "Ask if the path is clear, or how far an object is. "
                "Say guide me to start guidance, or be quiet for minimal alerts."
            )

        # 2. Path-clear question.
        if any(p in s for p in _YES_NO_PATH) or "go forward" in s or "go straight" in s:
            return self._answer_path(context)

        # 3. How-far question.
        for p in _HOW_FAR:
            if p in s:
                tail = s.split(p, 1)[1]
                target = _first_label_word(tail)
                if target:
                    ans = self._answer_distance(target)
                    if ans:
                        return ans

        # 4. Existence question ("is there a X").
        for p in _EXISTENCE:
            if p in s:
                tail = s.split(p, 1)[1].strip()
                target = _first_label_word(tail)
                if target:
                    return self._answer_existence(target)

        # 5. Direction-only mentions.
        if any(w in s for w in _FRONT_WORDS):
            return self._answer_path(context)
        if any(w in s for w in _LEFT_WORDS) or s == "left":
            return self._answer_direction("left")
        if any(w in s for w in _RIGHT_WORDS) or s == "right":
            return self._answer_direction("right")

        # 6. "Where am I" / "what's around" → scene summary.
        if "around" in s or "where am i" in s or "describe" in s or "tell me" in s:
            return self._answer_scene()

        # No intent matched. Stay silent — the session log showed this
        # fallback firing on misrecognized noise from STT and producing
        # "I am not sure I understood. I do not see anything notable
        # right now." on a loop. Silence is better than fake reassurance.
        return None

    # ------------------------------------------------------------------ helpers

    def _stable_objects(self):
        try:
            return self.pipeline._stable_objects()
        except Exception:
            return []

    def _answer_path(self, ctx: Optional[NavigationContext]) -> str:
        if ctx is None:
            # Fall back to building one ad-hoc.
            with self.pipeline._state._lock:
                fs = self.pipeline._state.free_space
            if fs is None or not fs.sectors:
                return "I cannot tell yet."
            ctx = NavigationContext.from_free_space(
                fs, [],
                min_go_distance_m=self.pipeline._config.guidance_min_go_distance_m,
                stop_distance_m=self.pipeline._config.guidance_stop_distance_m,
            )
        if ctx.front_path_clear:
            return "Yes, the path ahead is clear."
        if ctx.has_open_alternative:
            side = "left" if ctx.left_clear_m > ctx.right_clear_m else "right"
            return f"Front is blocked. The {side} side is open."
        return "No clear path ahead. Please wait."

    def _answer_distance(self, target: str) -> Optional[str]:
        from navigation.navigator import get_direction
        fw = self.pipeline._config.frame_width
        target = target.lower()
        matches = []
        for o in self._stable_objects():
            label = o.label.lower()
            if target in label or label in target:
                matches.append(o)
        if not matches:
            return f"I do not see a {target} right now."
        matches.sort(key=lambda o: (o.distance_m is None, o.distance_m or 1e9))
        o = matches[0]
        d = o.nearest_dist_m or o.distance_m
        direction = get_direction(o.box, fw)
        if d is None:
            return f"I see a {o.label} {direction}, but the distance is not ready."
        return f"The {o.label} is about {d:.1f} meters {direction}."

    def _answer_existence(self, target: str) -> str:
        from navigation.navigator import get_direction
        fw = self.pipeline._config.frame_width
        target = target.lower()
        for o in self._stable_objects():
            if target in o.label.lower() or o.label.lower() in target:
                d = o.nearest_dist_m or o.distance_m
                direction = get_direction(o.box, fw)
                if d is not None:
                    return f"Yes, a {o.label} {direction}, about {d:.1f} meters."
                return f"Yes, a {o.label} {direction}."
        return f"No, I do not see a {target} right now."

    def _answer_direction(self, direction: str) -> str:
        from navigation.navigator import get_bearing
        bearings = {
            "left": ("far_left", "left", "slight_left"),
            "right": ("far_right", "right", "slight_right"),
        }[direction]
        fw = self.pipeline._config.frame_width
        items = []
        for o in self._stable_objects():
            if get_bearing(o.box, fw) in bearings:
                d = o.nearest_dist_m or o.distance_m
                items.append((o, d if d is not None else 1e9))
        items.sort(key=lambda t: t[1])
        if not items:
            return f"Nothing notable on your {direction}."
        parts = []
        for o, d in items[:3]:
            if d >= 1e9:
                parts.append(o.label)
            else:
                parts.append(f"{o.label} about {d:.1f} meters")
        return f"On your {direction}: " + ", ".join(parts) + "."

    def _answer_scene(self, prefix: str = "") -> str:
        from navigation.navigator import get_direction, meters_to_verbal
        fw = self.pipeline._config.frame_width
        items = []
        for o in self._stable_objects():
            d = o.nearest_dist_m or o.distance_m
            if d is None:
                continue
            items.append((o, d))
        items.sort(key=lambda t: t[1])
        items = items[:3]
        if not items:
            return prefix + "I do not see anything notable right now."
        parts = []
        for o, d in items:
            parts.append(f"{o.label} {get_direction(o.box, fw)}, {meters_to_verbal(d)}")
        return prefix + "I see " + ", ".join(parts) + "."
