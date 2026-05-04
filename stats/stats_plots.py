import argparse
import json
import logging
import os
import time
from collections import Counter, defaultdict
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _load_events(events_path: str):
    if not os.path.isfile(events_path):
        return []
    out = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _gather(events) -> Tuple[Counter, List[float], List[float], List[Tuple[float, int]]]:
    label_counts: Counter = Counter()
    confs: List[float] = []
    dists: List[float] = []
    timeline: List[Tuple[float, int]] = []

    first_ts = None
    for ev in events:
        ts = ev.get("ts")
        if not isinstance(ts, (int, float)):
            continue
        if first_ts is None:
            first_ts = ts
        rel = ts - first_ts
        n_objs = 0
        for obj in ev.get("objects", []):
            label = (obj.get("label") or "unknown").strip()
            label_counts[label] += 1
            n_objs += 1
            c = obj.get("det_conf")
            if isinstance(c, (int, float)):
                confs.append(float(c))
            d = obj.get("distance_m")
            if isinstance(d, (int, float)):
                dists.append(float(d))
        timeline.append((rel, n_objs))
    return label_counts, confs, dists, timeline


def _approx_fps_timeline(timeline: List[Tuple[float, int]], window_s: float = 5.0) -> Tuple[List[float], List[float]]:
    if not timeline:
        return [], []
    timeline = sorted(timeline, key=lambda x: x[0])
    times = [t for t, _ in timeline]
    fps_x: List[float] = []
    fps_y: List[float] = []
    i = 0
    n = len(times)
    for j in range(n):
        t_end = times[j]
        t_start = t_end - window_s
        while i < j and times[i] < t_start:
            i += 1
        events_in_window = j - i + 1
        elapsed = max(t_end - times[i], 1e-6)
        fps = (events_in_window * 3.0) / elapsed
        fps_x.append(t_end)
        fps_y.append(fps)
    return fps_x, fps_y


def generate_plots(events_path: str = "events.jsonl", out_dir: str = ".") -> str:
    events = _load_events(events_path)
    if not events:
        logger.warning(f"No events to plot from {events_path}")
        return ""

    label_counts, confs, dists, timeline = _gather(events)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("NavVision Session Analytics", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    top = label_counts.most_common(15)
    labels = [k for k, _ in top]
    counts = [v for _, v in top]
    ax.barh(labels[::-1], counts[::-1], color="#3aa66e")
    ax.set_title("Object Frequency (Top 15)")
    ax.set_xlabel("Count")

    ax = axes[0, 1]
    if confs:
        ax.hist(confs, bins=20, color="#3a8ac6", edgecolor="black")
        ax.axvline(sum(confs) / len(confs), color="red", linestyle="--", label=f"Mean={sum(confs)/len(confs):.2f}")
        ax.legend()
    ax.set_title("Detection Confidence Distribution")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")

    ax = axes[1, 0]
    if dists:
        ax.hist(dists, bins=20, color="#c66a3a", edgecolor="black")
        ax.axvline(sum(dists) / len(dists), color="black", linestyle="--", label=f"Mean={sum(dists)/len(dists):.2f}m")
        ax.legend()
    ax.set_title("Distance Distribution (m)")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Count")

    ax = axes[1, 1]
    fx, fy = _approx_fps_timeline(timeline)
    if fx:
        ax.plot(fx, fy, color="#7a3ac6")
    ax.set_title("Estimated FPS Over Time")
    ax.set_xlabel("Session time (s)")
    ax.set_ylabel("Estimated FPS")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time())
    path = os.path.join(out_dir, f"session_plots_{ts}.png")
    plt.savefig(path, dpi=140)
    plt.close(fig)
    logger.info(f"Plots saved: {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description="Generate plots from NavVision session events")
    ap.add_argument("--events", default="events.jsonl")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    out = generate_plots(args.events, args.out_dir)
    if out:
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
