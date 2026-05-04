import csv
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

from communication_protocols import (
    CascadedBaselineConfig,
    DIRECT_SYSTEM_NAME,
    CASCADED_SYSTEM_NAME,
    cascaded_simulated_latency_ms,
    latency_improvement_pct,
)

logger = logging.getLogger(__name__)


METRICS_FIELDS = [
    "ts",
    "protocol",
    "selected_protocol",
    "source",
    "device",
    "message",
    "word_count",
    "closest_label",
    "distance_m",
    "direction",
    "action_category",
    "t_frame_received",
    "t_detection_done",
    "t_decision_started",
    "t_message_generated",
    "t_tts_requested",
    "t_audio_started",
    "t_audio_completed",
    "detection_latency_ms",
    "decision_latency_ms",
    "speech_latency_ms",
    "response_latency_ms",
    "cascaded_latency_ms",
    "improvement_pct",
    "is_duplicate",
    "overlap_skipped",
]

COMPARISON_FIELDS = [
    "ts",
    "protocol",
    "source",
    "device",
    "message",
    "word_count",
    "action_category",
    "actual_latency_ms",
    "cascaded_latency_ms",
    "improvement_pct",
    "asr_ms",
    "llm_ms",
    "tts_ms",
    "direct_system",
    "cascaded_system",
]


@dataclass
class ProtocolEvent:
    protocol: str = ""
    selected_protocol: str = ""
    source: str = ""
    device: str = ""
    message: str = ""
    word_count: int = 0
    closest_label: str = ""
    distance_m: Optional[float] = None
    direction: str = ""
    action_category: str = ""
    t_frame_received: Optional[float] = None
    t_detection_done: Optional[float] = None
    t_decision_started: Optional[float] = None
    t_message_generated: Optional[float] = None
    t_tts_requested: Optional[float] = None
    t_audio_started: Optional[float] = None
    t_audio_completed: Optional[float] = None
    is_duplicate: bool = False
    overlap_skipped: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


def _ms_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return round((float(a) - float(b)) * 1000.0, 2)


def compute_latencies(event: ProtocolEvent) -> Dict[str, Optional[float]]:
    detection = _ms_diff(event.t_detection_done, event.t_frame_received)
    decision = _ms_diff(event.t_message_generated, event.t_decision_started)
    speech = _ms_diff(event.t_audio_started, event.t_tts_requested)
    response = _ms_diff(event.t_audio_started, event.t_frame_received)
    return {
        "detection_latency_ms": detection,
        "decision_latency_ms": decision,
        "speech_latency_ms": speech,
        "response_latency_ms": response,
    }


class ProtocolLogger:
    def __init__(
        self,
        events_path: str = "protocol_events.jsonl",
        metrics_path: str = "protocol_metrics.csv",
        comparison_path: str = "latency_comparison.csv",
        baseline: Optional[CascadedBaselineConfig] = None,
        enabled: bool = True,
    ):
        self._events_path = events_path
        self._metrics_path = metrics_path
        self._comparison_path = comparison_path
        self._baseline = baseline or CascadedBaselineConfig()
        self._enabled = enabled
        self._lock = threading.Lock()
        self._events_file = None
        self._metrics_file = None
        self._metrics_writer = None
        self._comparison_file = None
        self._comparison_writer = None
        self._duplicate_count: int = 0
        self._overlap_count: int = 0
        self._counts_by_protocol: Dict[str, int] = {}
        self._last_event: Optional[ProtocolEvent] = None
        self._last_response_ms: Optional[float] = None
        self._last_speech_ms: Optional[float] = None
        if enabled:
            self._open()

    def _open(self) -> None:
        try:
            self._events_file = open(self._events_path, "a", encoding="utf-8", buffering=1)
        except Exception as e:
            logger.warning(f"protocol events open failed: {e}")
        try:
            metrics_new = (
                not os.path.exists(self._metrics_path)
                or os.path.getsize(self._metrics_path) == 0
            )
            self._metrics_file = open(
                self._metrics_path, "a", encoding="utf-8", buffering=1, newline=""
            )
            self._metrics_writer = csv.DictWriter(
                self._metrics_file, fieldnames=METRICS_FIELDS, extrasaction="ignore"
            )
            if metrics_new:
                self._metrics_writer.writeheader()
        except Exception as e:
            logger.warning(f"protocol metrics open failed: {e}")
        try:
            comp_new = (
                not os.path.exists(self._comparison_path)
                or os.path.getsize(self._comparison_path) == 0
            )
            self._comparison_file = open(
                self._comparison_path, "a", encoding="utf-8", buffering=1, newline=""
            )
            self._comparison_writer = csv.DictWriter(
                self._comparison_file, fieldnames=COMPARISON_FIELDS, extrasaction="ignore"
            )
            if comp_new:
                self._comparison_writer.writeheader()
        except Exception as e:
            logger.warning(f"latency comparison open failed: {e}")

    @property
    def baseline(self) -> CascadedBaselineConfig:
        return self._baseline

    @property
    def duplicate_count(self) -> int:
        return self._duplicate_count

    @property
    def overlap_count(self) -> int:
        return self._overlap_count

    @property
    def last_event(self) -> Optional[ProtocolEvent]:
        return self._last_event

    @property
    def last_response_ms(self) -> Optional[float]:
        return self._last_response_ms

    @property
    def last_speech_ms(self) -> Optional[float]:
        return self._last_speech_ms

    @property
    def counts_by_protocol(self) -> Dict[str, int]:
        return dict(self._counts_by_protocol)

    def mark_duplicate(self, protocol: str = "") -> int:
        with self._lock:
            self._duplicate_count += 1
            return self._duplicate_count

    def mark_overlap(self, protocol: str = "") -> int:
        with self._lock:
            self._overlap_count += 1
            return self._overlap_count

    def log(self, event: ProtocolEvent) -> Dict[str, Any]:
        latencies = compute_latencies(event)
        ts_ref = event.t_audio_started or event.t_message_generated or time.time()
        cascaded_ms = None
        improvement = None
        actual_response_ms = latencies.get("response_latency_ms")
        if actual_response_ms is not None:
            cascaded_ms = round(
                cascaded_simulated_latency_ms(actual_response_ms, self._baseline), 2
            )
            improvement = round(
                latency_improvement_pct(actual_response_ms, cascaded_ms), 2
            )
        record = {
            "ts": round(ts_ref, 3),
            "protocol": event.protocol,
            "selected_protocol": event.selected_protocol or event.protocol,
            "source": event.source,
            "device": event.device,
            "message": event.message,
            "word_count": int(event.word_count),
            "closest_label": event.closest_label,
            "distance_m": (
                round(float(event.distance_m), 3)
                if event.distance_m is not None
                else None
            ),
            "direction": event.direction,
            "action_category": event.action_category,
            "t_frame_received": _round_ts(event.t_frame_received),
            "t_detection_done": _round_ts(event.t_detection_done),
            "t_decision_started": _round_ts(event.t_decision_started),
            "t_message_generated": _round_ts(event.t_message_generated),
            "t_tts_requested": _round_ts(event.t_tts_requested),
            "t_audio_started": _round_ts(event.t_audio_started),
            "t_audio_completed": _round_ts(event.t_audio_completed),
            "detection_latency_ms": latencies["detection_latency_ms"],
            "decision_latency_ms": latencies["decision_latency_ms"],
            "speech_latency_ms": latencies["speech_latency_ms"],
            "response_latency_ms": latencies["response_latency_ms"],
            "cascaded_latency_ms": cascaded_ms,
            "improvement_pct": improvement,
            "is_duplicate": bool(event.is_duplicate),
            "overlap_skipped": bool(event.overlap_skipped),
        }
        if event.extra:
            record_extra = dict(record)
            record_extra["extra"] = event.extra
        else:
            record_extra = record
        with self._lock:
            self._last_event = event
            self._last_response_ms = latencies.get("response_latency_ms")
            self._last_speech_ms = latencies.get("speech_latency_ms")
            key = event.protocol or "unknown"
            self._counts_by_protocol[key] = self._counts_by_protocol.get(key, 0) + 1
            if event.is_duplicate:
                self._duplicate_count += 1
            if event.overlap_skipped:
                self._overlap_count += 1
            if self._events_file is not None:
                try:
                    self._events_file.write(json.dumps(record_extra) + "\n")
                except Exception as e:
                    logger.debug(f"protocol events write error: {e}")
            if self._metrics_writer is not None:
                try:
                    self._metrics_writer.writerow(record)
                except Exception as e:
                    logger.debug(f"protocol metrics write error: {e}")
            if self._comparison_writer is not None and actual_response_ms is not None:
                comp_row = {
                    "ts": record["ts"],
                    "protocol": event.protocol,
                    "source": event.source,
                    "device": event.device,
                    "message": event.message,
                    "word_count": event.word_count,
                    "action_category": event.action_category,
                    "actual_latency_ms": actual_response_ms,
                    "cascaded_latency_ms": cascaded_ms,
                    "improvement_pct": improvement,
                    "asr_ms": self._baseline.asr_ms,
                    "llm_ms": self._baseline.llm_ms,
                    "tts_ms": self._baseline.tts_ms,
                    "direct_system": DIRECT_SYSTEM_NAME,
                    "cascaded_system": CASCADED_SYSTEM_NAME,
                }
                try:
                    self._comparison_writer.writerow(comp_row)
                except Exception as e:
                    logger.debug(f"latency comparison write error: {e}")
        return record

    def close(self) -> None:
        with self._lock:
            for f in (self._events_file, self._metrics_file, self._comparison_file):
                if f is not None:
                    try:
                        f.close()
                    except Exception:
                        pass
            self._events_file = None
            self._metrics_file = None
            self._metrics_writer = None
            self._comparison_file = None
            self._comparison_writer = None


def _round_ts(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 3)
