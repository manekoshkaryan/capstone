import json
import logging
import os
import threading
import time
from typing import List, Optional
from perception.tracker import TrackedObject

logger = logging.getLogger(__name__)


class EventLogger:
    def __init__(
        self,
        enabled: bool,
        output_path: str,
        nav_output_path: str = "data/logs/navigation.jsonl",
        latency_output_path: str = "data/logs/latency.jsonl",
    ):
        self._enabled = enabled
        self._path = output_path
        self._nav_path = nav_output_path
        self._latency_path = latency_output_path
        self._lock = threading.Lock()
        self._file = None
        self._nav_file = None
        self._latency_file = None
        self._frame_count = 0
        self._last_nav_ts: float = 0.0
        self._nav_throttle_s: float = 2.0

        if enabled:
            try:
                self._file = open(output_path, "a", encoding="utf-8", buffering=1)
                logger.info(f"Event log opened: {output_path}")
            except Exception as e:
                logger.warning(f"Event log open failed: {e}")
                self._enabled = False
            try:
                self._nav_file = open(nav_output_path, "a", encoding="utf-8", buffering=1)
                logger.info(f"Navigation log opened: {nav_output_path}")
            except Exception as e:
                logger.warning(f"Navigation log open failed: {e}")
            try:
                self._latency_file = open(latency_output_path, "a", encoding="utf-8", buffering=1)
                logger.info(f"Latency log opened: {latency_output_path}")
            except Exception as e:
                logger.warning(f"Latency log open failed: {e}")

    def log_detections(self, tracked_objects: List[TrackedObject], frame_id: int):
        if not self._enabled or self._file is None:
            return
        self._frame_count += 1
        if self._frame_count % 3 != 0:
            return
        ts = time.time()
        records = []
        for obj in tracked_objects:
            if not obj.is_stable:
                continue
            records.append({
                "track_id": obj.track_id,
                "label": obj.label,
                "det_conf": round(obj.det_conf, 3),
                "box": [round(float(v), 1) for v in obj.box],
                "distance_m": round(obj.distance_m, 3) if obj.distance_m is not None else None,
                "nearest_m": round(obj.nearest_dist_m, 3) if obj.nearest_dist_m is not None else None,
                "dist_conf": obj.dist_conf_class,
                "is_obstacle": obj.is_obstacle,
                "distance_method": getattr(obj, "distance_method", "monocular_depth"),
            })
        event = {
            "ts": round(ts, 3),
            "frame_id": frame_id,
            "objects": records,
        }
        try:
            with self._lock:
                self._file.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.debug(f"Event log write error: {e}")

    def log_navigation(self, nav_record: dict):
        if not self._enabled or self._nav_file is None:
            return
        now = time.time()
        if now - self._last_nav_ts < self._nav_throttle_s:
            return
        self._last_nav_ts = now
        try:
            with self._lock:
                self._nav_file.write(json.dumps(nav_record) + "\n")
        except Exception as e:
            logger.debug(f"Nav log write error: {e}")

    def _write_latency(self, record: dict) -> None:
        if not self._enabled or self._latency_file is None:
            return
        try:
            with self._lock:
                self._latency_file.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.debug(f"Latency log write error: {e}")

    def log_announcement(
        self,
        text: str,
        urgency: str,
        ts_enqueue: float,
        ts_speak_start: float,
        ts_speak_end: float,
        hazard_ts: Optional[float] = None,
        source: Optional[str] = None,
        action: Optional[str] = None,
        is_repeat: bool = False,
        reason: Optional[str] = None,
    ) -> None:
        rec = {
            "kind": "announcement",
            "ts": round(ts_speak_start, 3),
            "text": text,
            "urgency": urgency,
            "source": source,
            "action": action,
            "queue_wait_ms": round((ts_speak_start - ts_enqueue) * 1000.0, 2),
            "speak_dur_ms": round((ts_speak_end - ts_speak_start) * 1000.0, 2),
        }
        if reason:
            rec["reason"] = reason
        if is_repeat:
            rec["is_repeat"] = True
        if hazard_ts is not None:
            rec["hazard_ts"] = round(hazard_ts, 3)
            rec["hazard_to_audio_ms"] = round((ts_speak_start - hazard_ts) * 1000.0, 2)
            rec["hazard_to_enqueue_ms"] = round((ts_enqueue - hazard_ts) * 1000.0, 2)
        self._write_latency(rec)

    def log_stt(
        self,
        text: str,
        ts_audio_end: float,
        ts_transcribe_end: float,
        engine: str,
        ok: bool = True,
    ) -> None:
        rec = {
            "kind": "stt",
            "ts": round(ts_transcribe_end, 3),
            "text": text,
            "engine": engine,
            "ok": ok,
            "transcribe_ms": round((ts_transcribe_end - ts_audio_end) * 1000.0, 2),
        }
        self._write_latency(rec)

    def log_latency(self, name: str, value_ms: float, **extra) -> None:
        rec = {"kind": "latency", "ts": round(time.time(), 3), "name": name, "value_ms": round(value_ms, 2)}
        rec.update(extra)
        self._write_latency(rec)

    def close(self):
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
        if self._nav_file:
            try:
                self._nav_file.close()
            except Exception:
                pass
            self._nav_file = None
        if self._latency_file:
            try:
                self._latency_file.close()
            except Exception:
                pass
            self._latency_file = None
