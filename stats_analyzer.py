import argparse
import json
import logging
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class SessionStats:
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_s: float = 0.0
    total_events: int = 0
    total_object_records: int = 0
    unique_track_ids: int = 0
    avg_fps_estimated: float = 0.0
    per_class_counts: Dict[str, int] = field(default_factory=dict)
    per_class_avg_conf: Dict[str, float] = field(default_factory=dict)
    per_class_avg_dist_m: Dict[str, float] = field(default_factory=dict)
    distance_method_counts: Dict[str, int] = field(default_factory=dict)
    obstacle_warnings: int = 0
    nav_critical_count: int = 0
    nav_warn_count: int = 0
    nav_info_count: int = 0
    most_common_objects: List[Dict[str, Any]] = field(default_factory=list)
    source_events_path: str = ""
    source_nav_path: str = ""


def analyze_events(
    events_path: str = "events.jsonl",
    nav_path: str = "navigation.jsonl",
) -> SessionStats:
    stats = SessionStats(source_events_path=events_path, source_nav_path=nav_path)

    if not os.path.isfile(events_path):
        logger.warning(f"Events file not found: {events_path}")
        return stats

    class_counts: Counter = Counter()
    class_conf_sum: Dict[str, float] = defaultdict(float)
    class_conf_n: Dict[str, int] = defaultdict(int)
    class_dist_sum: Dict[str, float] = defaultdict(float)
    class_dist_n: Dict[str, int] = defaultdict(int)
    method_counts: Counter = Counter()
    track_ids = set()
    obstacles = 0
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    n_events = 0
    n_records = 0

    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            n_events += 1
            ts = ev.get("ts")
            if isinstance(ts, (int, float)):
                if first_ts is None or ts < first_ts:
                    first_ts = ts
                if last_ts is None or ts > last_ts:
                    last_ts = ts
            for obj in ev.get("objects", []):
                n_records += 1
                label = (obj.get("label") or "unknown").strip()
                class_counts[label] += 1
                conf = obj.get("det_conf")
                if isinstance(conf, (int, float)):
                    class_conf_sum[label] += float(conf)
                    class_conf_n[label] += 1
                d = obj.get("distance_m")
                if isinstance(d, (int, float)):
                    class_dist_sum[label] += float(d)
                    class_dist_n[label] += 1
                method = (obj.get("distance_method") or "unknown").strip()
                method_counts[method] += 1
                tid = obj.get("track_id")
                if tid is not None:
                    track_ids.add(tid)
                if obj.get("is_obstacle"):
                    obstacles += 1

    nav_critical = nav_warn = nav_info = 0
    if os.path.isfile(nav_path):
        with open(nav_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                for it in rec.get("objects", []):
                    u = (it.get("urgency") or "").lower()
                    if u == "critical":
                        nav_critical += 1
                    elif u == "warn":
                        nav_warn += 1
                    elif u == "info":
                        nav_info += 1

    stats.started_at = first_ts or 0.0
    stats.ended_at = last_ts or 0.0
    stats.duration_s = (last_ts - first_ts) if first_ts and last_ts else 0.0
    stats.total_events = n_events
    stats.total_object_records = n_records
    stats.unique_track_ids = len(track_ids)
    stats.avg_fps_estimated = (n_events * 3.0) / stats.duration_s if stats.duration_s > 0 else 0.0
    stats.per_class_counts = dict(class_counts)
    stats.per_class_avg_conf = {
        k: round(class_conf_sum[k] / class_conf_n[k], 3)
        for k in class_conf_n if class_conf_n[k] > 0
    }
    stats.per_class_avg_dist_m = {
        k: round(class_dist_sum[k] / class_dist_n[k], 3)
        for k in class_dist_n if class_dist_n[k] > 0
    }
    stats.distance_method_counts = dict(method_counts)
    stats.obstacle_warnings = obstacles
    stats.nav_critical_count = nav_critical
    stats.nav_warn_count = nav_warn
    stats.nav_info_count = nav_info
    stats.most_common_objects = [
        {"label": k, "count": v} for k, v in class_counts.most_common(10)
    ]
    return stats


def print_summary_table(stats: SessionStats):
    print("=" * 60)
    print(" SESSION SUMMARY")
    print("=" * 60)
    print(f"  Duration:               {stats.duration_s:.1f} s")
    print(f"  Total event lines:      {stats.total_events}")
    print(f"  Total object records:   {stats.total_object_records}")
    print(f"  Unique track IDs:       {stats.unique_track_ids}")
    print(f"  Estimated FPS:          {stats.avg_fps_estimated:.1f}")
    print(f"  Obstacle warnings:      {stats.obstacle_warnings}")
    print(f"  Nav critical/warn/info: {stats.nav_critical_count} / "
          f"{stats.nav_warn_count} / {stats.nav_info_count}")
    print(f"  Distance methods:       {stats.distance_method_counts}")
    print()
    print("  Top 10 detected classes:")
    for entry in stats.most_common_objects:
        label = entry["label"]
        cnt = entry["count"]
        avg_conf = stats.per_class_avg_conf.get(label, 0.0)
        avg_dist = stats.per_class_avg_dist_m.get(label, 0.0)
        print(f"    {label:<24s} count={cnt:<6d} avg_conf={avg_conf:.2f}  avg_dist={avg_dist:.2f}m")
    print("=" * 60)


def save_summary_json(stats: SessionStats, out_dir: str = ".") -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time())
    path = os.path.join(out_dir, f"session_summary_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)
    logger.info(f"Session summary saved: {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description="Analyze NavVision session events")
    ap.add_argument("--events", default="events.jsonl")
    ap.add_argument("--nav", default="navigation.jsonl")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    stats = analyze_events(args.events, args.nav)
    print_summary_table(stats)
    if not args.no_save:
        save_summary_json(stats, args.out_dir)


if __name__ == "__main__":
    main()
