"""Piper TTS backend — drop-in alongside SpeechEngine for neural local TTS."""
from __future__ import annotations

import io
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = Path(__file__).parent.parent / "piper_test" / "models" / "en_US-lessac-low.onnx"
_DEFAULT_CONFIG = _DEFAULT_MODEL.with_suffix(".onnx.json")


class PiperTTS:
    """Thread-safe Piper TTS engine. Queues utterances, plays sequentially."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        mp = Path(model_path) if model_path else _DEFAULT_MODEL
        cp = Path(config_path) if config_path else _DEFAULT_CONFIG

        if not mp.exists():
            raise FileNotFoundError(f"Piper model not found: {mp}")

        try:
            from piper import PiperVoice
            self._voice = PiperVoice.load(str(mp), config_path=str(cp), use_cuda=False)
        except ImportError:
            raise ImportError("piper-tts not installed. Run: pip install piper-tts")

        self._queue: queue.Queue = queue.Queue()
        self._busy = False
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name="PiperTTS")
        self._thread.start()
        logger.info(f"PiperTTS ready — model={mp.name}")

    def is_speaking(self) -> bool:
        with self._lock:
            return self._busy or not self._queue.empty()

    def say(self, text: str, urgency: str = "info") -> None:
        """Enqueue text for synthesis and playback."""
        if not text or not text.strip():
            return
        self._queue.put((text.strip(), urgency))

    def synthesize_bytes(self, text: str) -> bytes:
        """Synthesize text and return raw WAV bytes without playing."""
        return self._synthesize(text.strip()) if text and text.strip() else b""

    def say_now(self, text: str, urgency: str = "critical") -> None:
        """Clear queue and speak immediately (for critical stops)."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put((text.strip(), urgency))

    def stop(self):
        self._running = False

    def _synthesize(self, text: str) -> bytes:
        chunks = list(self._voice.synthesize(text))
        if not chunks:
            return b""
        pcm = b"".join(c.audio_int16_bytes for c in chunks)
        sr = chunks[0].sample_rate
        sw = chunks[0].sample_width
        nc = chunks[0].sample_channels
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(nc)
            wf.setsampwidth(sw)
            wf.setframerate(sr)
            wf.writeframes(pcm)
        return buf.getvalue()

    def _play(self, wav_bytes: bytes):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp = f.name
        try:
            subprocess.run(["afplay", tmp], check=True, capture_output=True)
        except FileNotFoundError:
            try:
                subprocess.run(["aplay", tmp], check=True, capture_output=True)
            except Exception as e:
                logger.warning(f"PiperTTS playback failed: {e}")
        except Exception as e:
            logger.warning(f"PiperTTS afplay error: {e}")
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _worker(self):
        while self._running:
            try:
                text, urgency = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._lock:
                self._busy = True
            try:
                t0 = time.perf_counter()
                wav = self._synthesize(text)
                synth_ms = (time.perf_counter() - t0) * 1000
                logger.debug(f"PiperTTS synth {synth_ms:.0f}ms | {text!r}")
                if wav:
                    self._play(wav)
            except Exception as e:
                logger.warning(f"PiperTTS worker error: {e}")
            finally:
                with self._lock:
                    self._busy = False
