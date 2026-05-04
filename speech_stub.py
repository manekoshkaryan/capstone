import logging
import platform
import subprocess
import threading
import time
from typing import Optional, List, Dict, Tuple
from tracker import TrackedObject

logger = logging.getLogger(__name__)

_IS_MACOS = platform.system() == "Darwin"

try:
    import pyttsx3
    _PYTTSX3_AVAILABLE = True
except ImportError:
    _PYTTSX3_AVAILABLE = False


class SpeechStub:
    def __init__(
        self,
        enabled: bool = False,
        rate: int = 160,
        volume: float = 0.95,
        track_cooldown_s: float = 4.0,
        global_cooldown_s: float = 1.0,
        no_depth_cooldown_s: float = 5.0,
    ):
        self._enabled = enabled
        self._rate = rate
        self._volume = volume
        self._queue: List[str] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._track_cooldowns: Dict[int, float] = {}
        self._last_announced: float = 0.0
        self._last_no_depth_ts: float = 0.0
        self._track_cooldown_s = track_cooldown_s
        self._global_cooldown_s = global_cooldown_s
        self._no_depth_cooldown_s = no_depth_cooldown_s
        self._engine = None

        if enabled:
            self._init_engine()

    def _init_engine(self):
        if _IS_MACOS:
            self._running = True
            self._thread = threading.Thread(target=self._worker_macos, daemon=True, name="SpeechThread")
            self._thread.start()
            logger.info("Speech engine initialized (macOS say command)")
            return

        if not _PYTTSX3_AVAILABLE:
            logger.warning("pyttsx3 not installed — speech disabled")
            self._enabled = False
            return

        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._rate)
            self._engine.setProperty("volume", self._volume)
            self._running = True
            self._thread = threading.Thread(target=self._worker_pyttsx3, daemon=True, name="SpeechThread")
            self._thread.start()
            logger.info("Speech engine initialized (pyttsx3)")
        except Exception as e:
            logger.warning(f"pyttsx3 init failed: {e} — speech disabled")
            self._enabled = False

    def _worker_macos(self):
        while self._running:
            text = None
            with self._lock:
                if self._queue:
                    text = self._queue.pop(0)
            if text:
                try:
                    subprocess.run(
                        ["say", "-r", str(self._rate), text],
                        timeout=15,
                        capture_output=True,
                    )
                except Exception as e:
                    logger.debug(f"say error: {e}")
            else:
                time.sleep(0.05)

    def _worker_pyttsx3(self):
        while self._running:
            text = None
            with self._lock:
                if self._queue:
                    text = self._queue.pop(0)
            if text and self._engine:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception as e:
                    logger.debug(f"pyttsx3 error: {e}")
            else:
                time.sleep(0.05)

    def _enqueue(self, text: str):
        with self._lock:
            if len(self._queue) < 2:
                self._queue.append(text)
            else:
                self._queue[-1] = text

    def announce_navigation(self, prioritized: List[Tuple[TrackedObject, str]]):
        if not self._enabled:
            return
        now = time.time()
        if now - self._last_announced < self._global_cooldown_s:
            return

        for obj, msg in prioritized:
            last = self._track_cooldowns.get(obj.track_id, 0.0)
            if now - last >= self._track_cooldown_s:
                self._track_cooldowns[obj.track_id] = now
                self._last_announced = now
                logger.info(f"NAV TTS: {msg}")
                self._enqueue(msg)
                return

    def announce_detections_no_depth(self, objects: List[TrackedObject], frame_width: int):
        if not self._enabled:
            return
        now = time.time()
        if now - self._last_no_depth_ts < self._no_depth_cooldown_s:
            return
        stable = [o for o in objects if o.is_stable]
        if not stable:
            return

        from navigator import get_direction
        parts = []
        for obj in stable[:3]:
            direction = get_direction(obj.box, frame_width)
            parts.append(f"{obj.label.lower()} {direction}")

        if parts:
            msg = ". ".join(parts) + "."
            self._last_no_depth_ts = now
            self._last_announced = now
            logger.info(f"NAV TTS (no depth): {msg}")
            self._enqueue(msg)

    def speak(self, text: str, force: bool = False):
        if not self._enabled:
            return
        now = time.time()
        if not force and (now - self._last_announced) < self._global_cooldown_s:
            return
        self._last_announced = now
        self._enqueue(text)

    def announce_obstacle(self, label: str, distance_m: float):
        if not self._enabled:
            return
        msg = f"Warning: {label} at {distance_m:.1f} meters"
        self.speak(msg, force=True)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
