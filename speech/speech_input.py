import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import speech_recognition as sr
    _SR_AVAILABLE = True
except Exception as _sr_e:
    sr = None
    _SR_AVAILABLE = False
    logger.warning(f"speech_recognition import failed: {_sr_e}")

_SCENE_PATTERNS = [
    r"\bwhat (do|did) you see\b",
    r"\bwhat is around me\b",
    r"\bwhat'?s around me\b",
    r"\bdescribe (the )?scene\b",
    r"\bdescribe surroundings\b",
    r"\bwhat can you see\b",
    r"\bdescribe (?:the )?area\b",
]
# Direction questions — target is the direction.
# Match BEFORE _SCENE_PATTERNS so "what do you see on my left" routes here
# (direction-specific) rather than to a scene-wide description.
_DIRECTION_QUERY_PATTERNS = [
    # "what do you see on my left", "what can you see ahead"
    (r"\bwhat (?:do|did|can) you see (?:on )?(?:my )?(left|right|front|ahead)\b", 1),
    # "what is on my left", "whats on my right", "what's ahead"
    (r"\b(?:what(?:'s|s| is)|anything) (?:on )?(?:my )?(left|right|front|ahead)\b", 1),
    # "is there anything on my left"
    (r"\bis there (?:anything|something) (?:on )?(?:my )?(left|right|front|ahead)\b", 1),
    # "what about (my )?left"
    (r"\bwhat about (?:my )?(left|right|front|ahead)\b", 1),
]
# "Can I go forward?" / "Is the path clear?"
_PATH_CHECK_PATTERNS = [
    r"\bcan i (?:go|walk|move) (?:forward|ahead|straight)\b",
    r"\bis (?:the )?(?:path|way) clear\b",
    r"\bis it safe (?:to )?(?:go|walk)\b",
]
_FIND_PATTERNS = [
    r"\bfind (?:the )?(.+)$",
    r"\blocate (?:the )?(.+)$",
    r"\bcan you see (?:the |a |an )?(.+)$",
]
_NAVIGATE_PATTERNS = [
    r"\b(?:guide|navigate|take|lead) me to (?:the |a |an )?(.+)$",
    r"\bgo to (?:the |a |an )?(.+)$",
    r"\bbring me to (?:the |a |an )?(.+)$",
]
_WHERE_PATTERNS = [
    r"\bwhere(?:'s| is) (?:the )?(.+)$",
    r"\bremember(?: where)? (?:is )?(?:the |a |an )?(.+)$",
]
_FORGET_PATTERNS = [
    r"\bforget (?:the |a |an )?(.+)$",
    r"\bclear landmarks?\b",
]
_STOP_NAV_WORDS = ("stop guidance", "cancel guidance", "stop navigation", "cancel navigation", "stop guiding")
_ZERO_HEADING_WORDS = ("zero heading", "set north", "reset heading", "calibrate heading")
_PAUSE_WORDS = ("pause", "stop scanning", "freeze")
_RESUME_WORDS = ("resume", "continue", "start scanning", "unpause")
_CALIB_WORDS = ("calibrate camera", "calibration", "run calibration")
_LOUDER_WORDS = ("louder", "speak up", "volume up")
_QUIETER_WORDS = ("quieter", "softer", "volume down")
_STATUS_WORDS = ("status", "report", "diagnostics")
_QUIT_WORDS = ("shut down", "shutdown", "quit", "exit", "goodbye")

_WHAT_TO_DO_WORDS = (
    "what to do", "what should i do", "what do i do",
    "tell me what to do", "advise me",
)
_WHAT_IS_CLOSEST_WORDS = (
    "what is closest", "what's closest", "what is the closest",
    "closest object", "nearest object", "what is nearest",
    "what's the nearest", "nearest thing",
)
_WHAT_IS_AROUND_WORDS = (
    "what is around me", "whats around me", "what's around me",
    "what is around", "describe around me",
)
_REPEAT_WORDS = ("repeat", "say again", "say that again", "what did you say")
_HELP_WORDS = ("help", "instructions", "what commands")
_MUTE_WORDS = ("mute", "silence speech", "mute voice")
_UNMUTE_WORDS = ("unmute", "speak again", "unmute voice")
_STOP_SPEAKING_WORDS = ("stop speaking", "stop talking now", "be silent")
_PROTOCOL_MINIMAL_WORDS = ("minimal protocol", "set minimal protocol", "use minimal protocol")
_PROTOCOL_DESCRIPTIVE_WORDS = ("descriptive protocol", "set descriptive protocol", "use descriptive protocol")
_PROTOCOL_PROACTIVE_WORDS = ("proactive protocol", "set proactive protocol", "use proactive protocol")
_PROTOCOL_ADAPTIVE_WORDS = ("adaptive protocol", "set adaptive protocol", "use adaptive protocol")
_PROTOCOL_EVAL_WORDS = ("run protocol evaluation", "evaluate protocols", "protocol evaluation")

# Conversation-mode controls.
_MINIMAL_ALERT_WORDS = (
    "be quiet", "stop talking", "only warn me", "only warn me of danger",
    "minimal alert", "silent mode", "shut up",
)
_DESCRIBE_MORE_WORDS = (
    "describe more", "tell me everything", "tell me what's around",
    "tell me what is around", "verbose mode",
)
_GUIDE_ME_WORDS = (
    "guide me", "help me walk", "navigate with me", "keep guiding me",
    "start guiding", "walk with me",
)

@dataclass
class VoiceCommand:
    type: str
    target: Optional[str] = None
    raw: str = ""
    ts: float = 0.0
    source: str = "voice"

def parse_command(text: str) -> Optional[VoiceCommand]:
    if not text:
        return None
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9 \-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None

    for w in _PROTOCOL_EVAL_WORDS:
        if w in s:
            return VoiceCommand(type="run_protocol_evaluation", raw=text, ts=time.time())
    for w in _PROTOCOL_MINIMAL_WORDS:
        if w in s:
            return VoiceCommand(type="set_protocol_minimal", raw=text, ts=time.time())
    for w in _PROTOCOL_DESCRIPTIVE_WORDS:
        if w in s:
            return VoiceCommand(type="set_protocol_descriptive", raw=text, ts=time.time())
    for w in _PROTOCOL_PROACTIVE_WORDS:
        if w in s:
            return VoiceCommand(type="set_protocol_proactive", raw=text, ts=time.time())
    for w in _PROTOCOL_ADAPTIVE_WORDS:
        if w in s:
            return VoiceCommand(type="set_protocol_adaptive", raw=text, ts=time.time())

    for w in _WHAT_TO_DO_WORDS:
        if w in s:
            return VoiceCommand(type="what_to_do", raw=text, ts=time.time())
    for w in _WHAT_IS_CLOSEST_WORDS:
        if w in s:
            return VoiceCommand(type="what_is_closest", raw=text, ts=time.time())
    for w in _WHAT_IS_AROUND_WORDS:
        if w in s:
            return VoiceCommand(type="what_is_around_me", raw=text, ts=time.time())
    for w in _REPEAT_WORDS:
        if w in s:
            return VoiceCommand(type="repeat", raw=text, ts=time.time())
    for w in _UNMUTE_WORDS:
        if w in s:
            return VoiceCommand(type="unmute", raw=text, ts=time.time())
    for w in _MUTE_WORDS:
        if w in s:
            return VoiceCommand(type="mute", raw=text, ts=time.time())
    for w in _STOP_SPEAKING_WORDS:
        if w in s:
            return VoiceCommand(type="stop_speaking", raw=text, ts=time.time())
    for w in _HELP_WORDS:
        if w == s or s.startswith(w + " ") or s.endswith(" " + w):
            return VoiceCommand(type="help", raw=text, ts=time.time())

    # Mode-control commands take priority over generic find/scene patterns.
    for w in _MINIMAL_ALERT_WORDS:
        if w in s:
            return VoiceCommand(type="set_minimal_alert", raw=text, ts=time.time())
    for w in _GUIDE_ME_WORDS:
        if w in s:
            return VoiceCommand(type="set_continuous_guidance", raw=text, ts=time.time())
    for w in _DESCRIBE_MORE_WORDS:
        if w in s:
            return VoiceCommand(type="set_verbose_describe", raw=text, ts=time.time())

    # Direction questions BEFORE scene description — "what do you see on my
    # left" must be routed as a direction query, not a scene-wide describe.
    for pat, grp in _DIRECTION_QUERY_PATTERNS:
        m = re.search(pat, s)
        if m:
            direction = m.group(grp).strip()
            if direction in ("front", "ahead"):
                direction = "front"
            return VoiceCommand(
                type="query_direction", target=direction, raw=text, ts=time.time(),
            )

    for pat in _PATH_CHECK_PATTERNS:
        if re.search(pat, s):
            return VoiceCommand(type="query_path_clear", raw=text, ts=time.time())

    for pat in _SCENE_PATTERNS:
        if re.search(pat, s):
            return VoiceCommand(type="scene_description", raw=text, ts=time.time())

    for w in _STOP_NAV_WORDS:
        if w in s:
            return VoiceCommand(type="stop_navigation", raw=text, ts=time.time())

    for w in _ZERO_HEADING_WORDS:
        if w in s:
            return VoiceCommand(type="zero_heading", raw=text, ts=time.time())

    for pat in _NAVIGATE_PATTERNS:
        m = re.search(pat, s)
        if m:
            target = m.group(1).strip()
            target = re.sub(r"\b(please|now|right now)\b", "", target).strip()
            if target:
                return VoiceCommand(type="navigate_to", target=target, raw=text, ts=time.time())

    for pat in _FIND_PATTERNS:
        m = re.search(pat, s)
        if m:
            target = m.group(1).strip()
            target = re.sub(r"\b(please|now|right now)\b", "", target).strip()
            if target:
                return VoiceCommand(type="find_object", target=target, raw=text, ts=time.time())

    for pat in _WHERE_PATTERNS:
        m = re.search(pat, s)
        if m:
            target = m.group(1).strip()
            if target:
                return VoiceCommand(type="where_is", target=target, raw=text, ts=time.time())

    for pat in _FORGET_PATTERNS:
        m = re.search(pat, s)
        if m:
            target = "all"
            if m.lastindex:
                target = m.group(1).strip() or "all"
            return VoiceCommand(type="forget", target=target, raw=text, ts=time.time())

    for w in _PAUSE_WORDS:
        if w in s:
            return VoiceCommand(type="pause", raw=text, ts=time.time())
    for w in _RESUME_WORDS:
        if w in s:
            return VoiceCommand(type="resume", raw=text, ts=time.time())
    for w in _CALIB_WORDS:
        if w in s:
            return VoiceCommand(type="calibrate", raw=text, ts=time.time())
    for w in _LOUDER_WORDS:
        if w in s:
            return VoiceCommand(type="louder", raw=text, ts=time.time())
    for w in _QUIETER_WORDS:
        if w in s:
            return VoiceCommand(type="quieter", raw=text, ts=time.time())
    for w in _STATUS_WORDS:
        if w in s:
            return VoiceCommand(type="status", raw=text, ts=time.time())
    for w in _QUIT_WORDS:
        if w in s:
            return VoiceCommand(type="shutdown", raw=text, ts=time.time())

    return None

class SpeechListener:
    def __init__(
        self,
        on_command: Callable[[VoiceCommand], None],
        on_text: Optional[Callable[[str], None]] = None,
        is_speaking: Optional[Callable[[], bool]] = None,
        on_barge_in: Optional[Callable[[], None]] = None,
        reverb_tail_s: float = 0.3,
        energy_threshold: int = 300,
        pause_threshold: float = 0.7,
        phrase_time_limit: float = 4.0,
        ambient_calibration_s: float = 0.6,
        dynamic_energy: bool = True,
    ):
        self._on_command = on_command
        # Free-form fallback: if parse_command returns None, raw text is
        # forwarded here so the conversational handler can reply.
        self._on_text = on_text
        # Half-duplex gate: when our TTS is producing audio, suppress mic
        # transcription so the system does not "hear itself" and feed its
        # own voice back through STT.
        self._is_speaking = is_speaking
        # Barge-in: invoked the moment user audio is captured (before STT
        # finishes). Lets the pipeline cancel current TTS chatter so the
        # user can be heard immediately.
        self._on_barge_in = on_barge_in
        # Wait this long after TTS stops before re-opening mic — lets room
        # echo / speaker reverb decay so we don't transcribe ourselves.
        self._reverb_tail_s = float(reverb_tail_s)
        self._tts_done_ts: float = 0.0
        self._energy_threshold = energy_threshold
        self._pause_threshold = pause_threshold
        self._phrase_time_limit = phrase_time_limit
        self._ambient_calibration_s = ambient_calibration_s
        self._dynamic_energy = dynamic_energy
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._recognizer = None
        self._microphone = None
        self._available = _SR_AVAILABLE
        self._stt_listener: Optional[Callable[[str, float, float, str, bool], None]] = None

    def set_stt_listener(self, cb):
        """cb(text, ts_audio_end, ts_transcribe_end, engine, ok)"""
        self._stt_listener = cb

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> bool:
        if not self._available:
            logger.warning("Voice input unavailable: speech_recognition not installed")
            return False

        try:
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = self._energy_threshold
            self._recognizer.pause_threshold = self._pause_threshold
            self._recognizer.dynamic_energy_threshold = self._dynamic_energy
            self._microphone = sr.Microphone()
            with self._microphone as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=self._ambient_calibration_s)
            logger.info(
                f"SpeechListener calibrated: energy={self._recognizer.energy_threshold:.0f} "
                f"dynamic={self._dynamic_energy}"
            )
        except Exception as e:
            logger.error(f"SpeechListener init failed: {e}")
            self._available = False
            return False

        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True, name="SpeechListener")
        self._thread.start()
        logger.info("SpeechListener thread started")
        return True

    def _listen_loop(self):
        consecutive_failures = 0
        base_energy = self._recognizer.energy_threshold
        # Multiplier applied to energy_threshold while TTS plays so the mic
        # only triggers on a genuinely loud user voice (not the speaker
        # leakage). Best-effort full-duplex with a single-stream STT.
        tts_energy_mult = 2.5
        while self._running:
            # Track TTS state for echo-decay timing and threshold raise.
            tts_active = False
            if self._is_speaking is not None:
                try:
                    tts_active = bool(self._is_speaking())
                except Exception:
                    tts_active = False
            if tts_active:
                self._tts_done_ts = time.time()
            elif (time.time() - self._tts_done_ts) < self._reverb_tail_s:
                # Brief tail after TTS stops — room echo decay window. Don't
                # idle the mic completely (we want to catch user voice that
                # started during TTS), but pause briefly to avoid the very
                # last syllable of our own speech.
                time.sleep(0.05)
                continue

            try:
                # Adapt energy threshold on the fly: high during TTS so
                # speaker leakage is rejected, normal otherwise.
                if tts_active:
                    self._recognizer.energy_threshold = base_energy * tts_energy_mult
                else:
                    base_energy = self._recognizer.energy_threshold if not self._dynamic_energy else base_energy
                    if not self._dynamic_energy:
                        self._recognizer.energy_threshold = base_energy
                with self._microphone as source:
                    audio = self._recognizer.listen(
                        source,
                        timeout=2.0,
                        phrase_time_limit=self._phrase_time_limit,
                    )
                ts_audio_end = time.time()
            except sr.WaitTimeoutError:
                continue
            except OSError as e:
                consecutive_failures += 1
                logger.warning(f"Microphone OS error: {e}")
                if consecutive_failures >= 5:
                    logger.error("SpeechListener too many mic errors — stopping")
                    self._running = False
                    break
                time.sleep(0.5)
                continue
            except Exception as e:
                logger.warning(f"SpeechListener listen error: {e}")
                time.sleep(0.3)
                continue

            consecutive_failures = 0

            # Drop sub-300ms clips before paying the STT round-trip — they
            # are almost always speaker leakage / room noise, not commands.
            try:
                sr_rate = int(getattr(audio, "sample_rate", 16000) or 16000)
                sw = int(getattr(audio, "sample_width", 2) or 2)
                clip_ms = (len(audio.frame_data) / max(1, sr_rate * sw)) * 1000.0
            except Exception:
                clip_ms = 9999.0
            if clip_ms < 300.0:
                continue

            # Barge-in: audio was captured. Cancel any chatter still in
            # the TTS queue immediately, even before STT returns. The user
            # is talking — they should not have to wait for "Stop. Wall
            # left." to finish playing.
            if self._on_barge_in is not None:
                try:
                    self._on_barge_in()
                except Exception as e:
                    logger.debug(f"on_barge_in error: {e}")

            text, engine_used = self._transcribe(audio)
            ts_transcribe_end = time.time()
            if self._stt_listener is not None:
                try:
                    self._stt_listener(text, ts_audio_end, ts_transcribe_end, engine_used, bool(text))
                except Exception as e:
                    logger.debug(f"stt_listener error: {e}")
            if not text:
                continue

            logger.info(f"VOICE heard: {text!r}")
            cmd = parse_command(text)
            if cmd is None:
                # Free-form fallback: route to conversational handler if
                # one is registered. Otherwise drop quietly.
                if self._on_text is not None:
                    try:
                        self._on_text(text)
                    except Exception as e:
                        logger.error(f"on_text handler error: {e}")
                else:
                    logger.info(f"VOICE no command match for: {text!r}")
                continue

            try:
                self._on_command(cmd)
            except Exception as e:
                logger.error(f"Command handler error: {e}")

    def _transcribe(self, audio) -> Tuple[str, str]:
        try:
            return self._recognizer.recognize_google(audio), "google"
        except sr.UnknownValueError:
            return "", "google"
        except sr.RequestError as e:
            logger.debug(f"Google STT request error: {e} — trying offline sphinx")
        except Exception as e:
            logger.debug(f"Google STT exception: {e}")

        try:
            return self._recognizer.recognize_sphinx(audio), "sphinx"
        except Exception:
            return "", "sphinx"

    def stop(self):
        if not self._running:
            return
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.5)
        logger.info("SpeechListener stopped")
