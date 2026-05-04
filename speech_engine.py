import logging
import platform
import shutil
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_IS_MACOS = platform.system() == "Darwin"
_IS_WINDOWS = platform.system() == "Windows"

try:
    import pyttsx3 as _pyttsx3
    _PYTTSX3_AVAILABLE = True
except ImportError:
    _pyttsx3 = None
    _PYTTSX3_AVAILABLE = False

_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

_BEARING_SHORT: Dict[str, str] = {
    "far_left":     "hard left",
    "left":         "left",
    "slight_left":  "slightly left",
    "center":       "ahead",
    "slight_right": "slightly right",
    "right":        "right",
    "far_right":    "hard right",
}

_BEARING_DEGREE: Dict[str, str] = {
    "far_left":     "ninety left",
    "left":         "forty-five left",
    "slight_left":  "slightly left",
    "center":       "straight",
    "slight_right": "slightly right",
    "right":        "forty-five right",
    "far_right":    "ninety right",
}

_AVOID: Dict[str, str] = {
    "far_left":     "Continue straight.",
    "left":         "Step right.",
    "slight_left":  "Step right.",
    "center":       "Slight right.",
    "slight_right": "Step left.",
    "right":        "Step left.",
    "far_right":    "Continue straight.",
}

_URGENCY_RANK: Dict[str, int] = {"critical": 0, "warn": 1, "info": 2}


def _to_meters(dist_m: float) -> int:
    return max(1, min(int(dist_m + 0.4999), 5))


def _distance_phrase(dist_m: float) -> str:
    if dist_m < 1.0:
        return "close"
    n = _to_meters(dist_m)
    return f"{_WORDS[n]} meter" if n == 1 else f"{_WORDS[n]} meters"


def _bearing_phrase(bearing: str, degree_form: bool = False) -> str:
    table = _BEARING_DEGREE if degree_form else _BEARING_SHORT
    return table.get(bearing, "ahead")


def _format_spp(detection: dict) -> str:
    is_new = detection.get("is_new", True)
    post_turn = detection.get("post_turn", False)

    if not is_new and not post_turn:
        return ""

    obj_class = detection.get("class")
    distance_m = detection.get("distance_m")
    bearing = detection.get("bearing", "center")
    urgency = detection.get("urgency", "info")
    is_crossing = detection.get("is_crossing", False)
    repeat = int(detection.get("repeat_count", 0) or 0)

    bearing_str = _bearing_phrase(bearing)

    if post_turn:
        if obj_class is None or distance_m is None or distance_m >= 5.0:
            return "Now clear."
        obj_str = obj_class.capitalize()
        dist_str = _distance_phrase(distance_m)
        avoid = _AVOID.get(bearing, "")
        base = f"Now {obj_str} {dist_str} {bearing_str}."
        return f"{base} {avoid}" if avoid else base

    if obj_class is None:
        return ""

    obj_str = obj_class.capitalize()

    if urgency == "critical":
        if repeat > 0:
            if bearing == "center":
                return f"Still {obj_str} ahead. Hold."
            avoid = _AVOID.get(bearing, "")
            base = f"Still {obj_str} {bearing_str}."
            return f"{base} {avoid}" if avoid else base
        if bearing == "center":
            return f"Stop. {obj_str} {bearing_str}."
        avoid = _AVOID.get(bearing, "")
        base = f"{obj_str} very close {bearing_str}."
        return f"{base} {avoid}" if avoid else base

    if distance_m is None:
        return f"{obj_str} {bearing_str}."

    dist_str = _distance_phrase(distance_m)

    if urgency == "warn":
        if is_crossing and bearing == "center":
            return f"{obj_str} crossing ahead. Wait."
        avoid = _AVOID.get(bearing, "")
        base = f"{obj_str} {dist_str} {bearing_str}."
        if avoid:
            candidate = f"{base} {avoid}"
            if len(candidate.replace(".", "").split()) <= 8:
                return candidate
        return base

    if urgency == "info":
        return f"{obj_str} {dist_str} {bearing_str}."

    return ""


def _wpm_to_sapi_rate(wpm: int) -> int:
    base = 180.0
    val = int(round((wpm - base) / 18.0))
    return max(-10, min(10, val))


class _Pyttsx3Backend:
    name = "pyttsx3"

    def __init__(self, default_rate: int, volume: float):
        self._engine = _pyttsx3.init()
        self._engine.setProperty("volume", volume)
        self._default_rate = default_rate
        try:
            self._engine.setProperty("rate", default_rate)
        except Exception:
            pass
        try:
            self._engine.say("")
            self._engine.runAndWait()
        except Exception as e:
            raise RuntimeError(f"pyttsx3 probe failed: {e}")

    def say(self, text: str, rate: int) -> None:
        try:
            self._engine.setProperty("rate", rate)
        except Exception:
            pass
        self._engine.say(text)
        self._engine.runAndWait()

    def set_volume(self, v: float) -> None:
        try:
            self._engine.setProperty("volume", v)
        except Exception:
            pass

    def interrupt(self) -> None:
        try:
            self._engine.stop()
        except Exception:
            pass

    def stop(self) -> None:
        self.interrupt()


class _SAPIBackend:
    name = "sapi-powershell"

    def __init__(self, default_rate: int, volume: float):
        if not _IS_WINDOWS:
            raise RuntimeError("SAPI is Windows-only")
        self._powershell = shutil.which("powershell.exe") or shutil.which("powershell") or "powershell.exe"
        self._default_rate = default_rate
        self._volume_pct = max(0, min(100, int(round(volume * 100.0))))
        self._sapi_rate = _wpm_to_sapi_rate(default_rate)
        self._proc: Optional[subprocess.Popen] = None
        self._probe()

    def _probe(self) -> None:
        try:
            r = subprocess.run(
                [self._powershell, "-NoProfile", "-NonInteractive", "-Command",
                 "Add-Type -AssemblyName System.Speech; "
                 "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                 "$s.Rate = 0; $s.Volume = 100; $s.Speak(' ')"],
                capture_output=True, timeout=8.0,
            )
            if r.returncode != 0:
                err = (r.stderr or b"").decode("utf-8", errors="ignore")[:200]
                raise RuntimeError(f"SAPI probe rc={r.returncode}: {err}")
        except FileNotFoundError as e:
            raise RuntimeError(f"PowerShell missing: {e}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("SAPI probe timed out")

    def say(self, text: str, rate: int) -> None:
        sapi_rate = _wpm_to_sapi_rate(rate)
        safe = text.replace("'", "''")
        cmd = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = {sapi_rate}; $s.Volume = {self._volume_pct}; "
            f"$s.Speak('{safe}')"
        )
        try:
            self._proc = subprocess.Popen(
                [self._powershell, "-NoProfile", "-NonInteractive", "-Command", cmd],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            try:
                self._proc.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
        except Exception as e:
            logger.debug(f"SAPI Speak error: {e}")
        finally:
            self._proc = None

    def set_volume(self, v: float) -> None:
        self._volume_pct = max(0, min(100, int(round(v * 100.0))))

    def interrupt(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def stop(self) -> None:
        pass


class _MacSayBackend:
    name = "macos-say"

    def __init__(self, default_rate: int, volume: float):
        if not _IS_MACOS:
            raise RuntimeError("'say' is macOS-only")
        if shutil.which("say") is None:
            raise RuntimeError("'say' binary not found")
        self._volume = volume
        self._proc: Optional[subprocess.Popen] = None
        self._apply_volume()

    def _apply_volume(self) -> None:
        try:
            pct = max(0, min(100, int(round(self._volume * 100.0))))
            subprocess.Popen(
                ["osascript", "-e", f"set volume output volume {pct}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    def say(self, text: str, rate: int) -> None:
        # Popen + wait so interrupt() can kill the running process for
        # real-time barge-in (subprocess.run blocks uninterruptibly).
        try:
            self._proc = subprocess.Popen(
                ["say", "-r", str(rate), text],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            try:
                self._proc.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
        except Exception as e:
            logger.debug(f"say error: {e}")
        finally:
            self._proc = None

    def set_volume(self, v: float) -> None:
        self._volume = v
        self._apply_volume()

    def interrupt(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def stop(self) -> None:
        self.interrupt()


def _build_backend(default_rate: int, volume: float):
    if _IS_MACOS:
        try:
            return _MacSayBackend(default_rate, volume)
        except Exception as e:
            logger.warning(f"macos-say backend unavailable: {e}")

    if _PYTTSX3_AVAILABLE:
        try:
            be = _Pyttsx3Backend(default_rate, volume)
            logger.info("TTS backend: pyttsx3")
            return be
        except Exception as e:
            logger.warning(f"pyttsx3 backend failed ({e}) — trying SAPI fallback")

    if _IS_WINDOWS:
        try:
            be = _SAPIBackend(default_rate, volume)
            logger.info("TTS backend: SAPI (PowerShell)")
            return be
        except Exception as e:
            logger.error(f"SAPI fallback failed: {e}")

    logger.error("TTS DISABLED — no working backend (install pyttsx3, or run on Windows/macOS)")
    return None


class SpeechEngine:
    def __init__(self, rate: int = 185):
        self._rate_normal = rate
        self._rate_critical = 235
        self._lock = threading.Lock()
        self._queue: List[Tuple[int, str, str, int, float, dict]] = []
        self._recent: Dict[Tuple[str, str], Tuple[float, str]] = {}
        self._running = True
        self._volume: float = 0.95
        self._volume_step: float = 0.10
        self._volume_min: float = 0.10
        self._volume_max: float = 1.0
        self._text_listener = None
        self._announce_listener = None
        self._overlap_listener = None
        self._mute_local: bool = False
        self._busy: bool = False
        self._current_urgency: Optional[str] = None
        self._backend = _build_backend(rate, self._volume)

        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="SpeechEngine"
        )
        self._thread.start()
        backend_name = self._backend.name if self._backend is not None else "none"
        logger.info(f"SpeechEngine ready — backend={backend_name} rate={rate} vol={self._volume:.2f}")

    @property
    def backend_name(self) -> str:
        return self._backend.name if self._backend is not None else "none"

    @property
    def is_available(self) -> bool:
        return self._backend is not None

    def is_speaking(self) -> bool:
        """True while TTS is producing audio. Used by SpeechListener as a
        half-duplex gate so the mic does not capture our own voice and
        round-trip it through STT."""
        with self._lock:
            return self._busy or bool(self._queue)

    def set_text_listener(self, cb):
        self._text_listener = cb

    def set_announce_listener(self, cb):
        """cb(text, urgency, ts_enqueue, ts_speak_start, ts_speak_end, meta)"""
        self._announce_listener = cb

    def set_overlap_listener(self, cb):
        self._overlap_listener = cb

    def set_mute_local(self, muted: bool):
        self._mute_local = bool(muted)
        logger.info(f"Local TTS mute -> {self._mute_local}")

    def adjust_volume(self, delta: float) -> float:
        with self._lock:
            self._volume = max(self._volume_min, min(self._volume_max, self._volume + delta))
            v = self._volume
        if self._backend is not None:
            try:
                self._backend.set_volume(v)
            except Exception:
                pass
        logger.info(f"TTS volume -> {v:.2f}")
        return v

    def louder(self) -> float:
        return self.adjust_volume(self._volume_step)

    def quieter(self) -> float:
        return self.adjust_volume(-self._volume_step)

    def cancel_pending(self, keep_critical: bool = True, stop_current: bool = False) -> int:
        """Drop queued non-critical utterances. Used for barge-in: when the
        user starts speaking, abandon stale chatter so we listen instead.

        If `stop_current` is True, also kill the currently-playing utterance
        (best effort — depends on backend supporting interrupt). Critical
        playback is left alone unless `keep_critical=False`.
        """
        with self._lock:
            before = len(self._queue)
            if keep_critical:
                self._queue = [item for item in self._queue if item[0] == 0]
            else:
                self._queue.clear()
            dropped = before - len(self._queue)
            backend = self._backend
            busy = self._busy
        if dropped:
            logger.info(f"speech: cancelled {dropped} pending utterances (barge-in)")
        if stop_current and busy and backend is not None:
            interrupt = getattr(backend, "interrupt", None)
            if callable(interrupt):
                try:
                    interrupt()
                    logger.info("speech: interrupted current TTS (barge-in)")
                except Exception as e:
                    logger.debug(f"interrupt error: {e}")
        return dropped

    @property
    def volume(self) -> float:
        return self._volume

    def say(self, text: str, urgency: str = "info", meta: Optional[dict] = None) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        rate = self._rate_critical if urgency == "critical" else self._rate_normal
        priority = _URGENCY_RANK.get(urgency, 2)
        now = time.time()
        meta = dict(meta) if meta else {}
        meta.setdefault("ts_tts_requested", now)
        with self._lock:
            # 1. Dedup: drop if same text spoken or queued in the last 2s
            #    (anti-stutter for repeated guidance frames).
            last = self._recent.get((text, urgency))
            if last and (now - last[0]) < 2.0:
                self._notify_overlap(text, urgency, "dedup-recent", meta)
                return False
            for item in self._queue:
                if item[2] == text:
                    self._notify_overlap(text, urgency, "dedup-queued", meta)
                    return False
            # 2. Bound queue. Critical messages preempt; non-critical that
            #    would back up TTS get dropped to keep latency low.
            preempt_now = False
            if urgency == "critical":
                self._queue = [it for it in self._queue if it[0] == 0]
                self._queue.append((priority, urgency, text, rate, now, meta))
                if self._busy and self._current_urgency not in ("critical", None):
                    preempt_now = True
            else:
                if len(self._queue) >= 3:
                    # Drop oldest non-critical.
                    for i, it in enumerate(self._queue):
                        if it[0] > 0:
                            self._queue.pop(i)
                            break
                self._queue.append((priority, urgency, text, rate, now, meta))
            self._recent[(text, urgency)] = (now, urgency)
            # GC recent map.
            if len(self._recent) > 32:
                cutoff = now - 10.0
                self._recent = {k: v for k, v in self._recent.items() if v[0] > cutoff}
        if preempt_now and self._backend is not None:
            interrupt = getattr(self._backend, "interrupt", None)
            if callable(interrupt):
                try:
                    interrupt()
                    logger.info("speech: preempted non-critical TTS for critical alert")
                except Exception as e:
                    logger.debug(f"critical preempt error: {e}")
        return True

    def _notify_overlap(self, text: str, urgency: str, reason: str, meta: dict) -> None:
        if self._overlap_listener is None:
            return
        try:
            self._overlap_listener(text, urgency, reason, meta)
        except Exception as e:
            logger.debug(f"overlap_listener error: {e}")

    def _say_sync(self, text: str, rate: int):
        if self._backend is None:
            return
        try:
            self._backend.say(text, rate)
        except Exception as e:
            logger.warning(f"TTS backend '{self._backend.name}' failed mid-speak: {e}")
            if _IS_WINDOWS and self._backend.name != "sapi-powershell":
                try:
                    self._backend = _SAPIBackend(self._rate_normal, self._volume)
                    logger.info("TTS switched to SAPI fallback after runtime failure")
                    self._backend.say(text, rate)
                except Exception as e2:
                    logger.error(f"SAPI fallback also failed: {e2}")
                    self._backend = None

    def _worker(self):
        while self._running:
            item = None
            with self._lock:
                if self._queue:
                    # Critical-first ordering: a fresh critical may have been
                    # appended after non-critical; pop the highest-priority
                    # (lowest priority number) item first.
                    if any(it[0] == 0 for it in self._queue) and self._queue[0][0] != 0:
                        for i, it in enumerate(self._queue):
                            if it[0] == 0:
                                item = self._queue.pop(i)
                                break
                    else:
                        item = self._queue.pop(0)
                    self._busy = True
                    self._current_urgency = item[1]
            if item is None:
                with self._lock:
                    self._busy = False
                    self._current_urgency = None
                time.sleep(0.03)
                continue
            _, urgency, text, rate, ts_enqueue, meta = item
            logger.info(f"SPEAK [{urgency.upper()}]: {text}")
            if self._text_listener is not None:
                try:
                    self._text_listener(text, urgency)
                except Exception as e:
                    logger.debug(f"text_listener error: {e}")
            ts_speak_start = time.time()
            if not self._mute_local and self._backend is not None:
                self._say_sync(text, rate)
            ts_speak_end = time.time()
            if self._announce_listener is not None:
                try:
                    self._announce_listener(
                        text, urgency, ts_enqueue, ts_speak_start, ts_speak_end, meta,
                    )
                except Exception as e:
                    logger.debug(f"announce_listener error: {e}")
            with self._lock:
                self._busy = False
                self._current_urgency = None

    def _format(self, detection: dict) -> str:
        return _format_spp(detection)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._backend is not None:
            try:
                self._backend.stop()
            except Exception:
                pass
