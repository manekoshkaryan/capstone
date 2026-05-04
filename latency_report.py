"""Summarize latency.jsonl produced by EventLogger.

Reports p50 / p95 / max for:
  - hazard_to_audio_ms (e2e safety alert)
  - hazard_to_enqueue_ms (policy decision)
  - queue_wait_ms (TTS queue overhead)
  - speak_dur_ms (utterance duration)
  - transcribe_ms (STT latency)
"""
import argparse
import json
import statistics
from collections import defaultdict
from typing import Dict, List


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def _summarize(name: str, xs: List[float]) -> str:
    if not xs:
        return f"{name:30s} n=0"
    return (
        f"{name:30s} n={len(xs):4d}  "
        f"p50={_percentile(xs, 50):7.1f}ms  "
        f"p95={_percentile(xs, 95):7.1f}ms  "
        f"max={max(xs):7.1f}ms  "
        f"mean={statistics.fmean(xs):7.1f}ms"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", default="latency.jsonl", nargs="?")
    p.add_argument("--by-source", action="store_true", help="group announcements by source/action")
    args = p.parse_args()

    fields = [
        "hazard_to_audio_ms",
        "hazard_to_enqueue_ms",
        "queue_wait_ms",
        "speak_dur_ms",
        "transcribe_ms",
    ]
    bucket: Dict[str, List[float]] = defaultdict(list)
    by_action: Dict[str, List[float]] = defaultdict(list)
    by_engine: Dict[str, List[float]] = defaultdict(list)
    n_total = 0

    with open(args.path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            n_total += 1
            is_repeat = bool(rec.get("is_repeat"))
            for k in fields:
                if k in rec and isinstance(rec[k], (int, float)):
                    if is_repeat and k in ("hazard_to_audio_ms", "hazard_to_enqueue_ms"):
                        continue
                    bucket[k].append(float(rec[k]))
            if rec.get("kind") == "announcement" and "hazard_to_audio_ms" in rec and not is_repeat:
                act = rec.get("action") or rec.get("source") or "?"
                by_action[act].append(float(rec["hazard_to_audio_ms"]))
            if rec.get("kind") == "stt" and "transcribe_ms" in rec:
                eng = rec.get("engine") or "?"
                by_engine[eng].append(float(rec["transcribe_ms"]))

    print(f"records: {n_total}")
    print()
    for k in fields:
        print(_summarize(k, bucket[k]))

    if args.by_source and by_action:
        print()
        print("hazard_to_audio_ms by action:")
        for k, xs in sorted(by_action.items()):
            print("  " + _summarize(k, xs))

    if by_engine:
        print()
        print("transcribe_ms by engine:")
        for k, xs in sorted(by_engine.items()):
            print("  " + _summarize(k, xs))


if __name__ == "__main__":
    main()
