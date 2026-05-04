import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv
import glob
import json
import logging
import os
import time
from collections import Counter
from typing import Dict, List, Optional

import cv2
import numpy as np

from config import AppConfig
from perception.calibration import CalibrationData
from perception.detector import ObjectDetector
from perception.depth_estimator import DepthEstimator

logger = logging.getLogger(__name__)


def _list_session_dirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    out = []
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if not os.path.isdir(full):
            continue
        if os.path.isdir(os.path.join(full, "left")) or os.path.isfile(os.path.join(full, "metadata.json")):
            out.append(full)
    return out


def _list_frames(session_dir: str, side: str = "left") -> List[str]:
    folder = os.path.join(session_dir, side)
    if not os.path.isdir(folder):
        return []
    return sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))


def _load_metadata(session_dir: str) -> Dict:
    meta_path = os.path.join(session_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _roi_median_distance(depth_map: np.ndarray, box) -> Optional[float]:
    if depth_map is None:
        return None
    h, w = depth_map.shape[:2]
    x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
    x2 = min(w - 1, int(box[2])); y2 = min(h - 1, int(box[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = depth_map[y1:y2, x1:x2]
    valid = roi[(roi > 0.05) & (roi < 25.0) & np.isfinite(roi)]
    if valid.size < 20:
        return None
    return float(np.median(valid))


def _process_session(
    session_dir: str,
    detector: ObjectDetector,
    depth: DepthEstimator,
    config: AppConfig,
    max_frames: int,
    side: str,
) -> Dict:
    frames = _list_frames(session_dir, side)
    if max_frames > 0:
        frames = frames[:max_frames]
    metadata = _load_metadata(session_dir)
    env = metadata.get("environment") or os.path.basename(session_dir).split("_")[0]

    n_frames = 0
    n_objects_total = 0
    conf_sum = 0.0
    conf_n = 0
    dist_sum = 0.0
    dist_n = 0
    label_counts: Counter = Counter()
    timing_total_ms = 0.0
    failed_frames = 0

    started = time.time()
    for fp in frames:
        frame = cv2.imread(fp)
        if frame is None:
            failed_frames += 1
            continue
        t0 = time.perf_counter()
        try:
            dets = detector.detect(frame)
        except Exception as e:
            logger.debug(f"detect failed: {e}")
            dets = []
        try:
            dmap = depth.estimate(frame)
        except Exception:
            dmap = None
        timing_total_ms += (time.perf_counter() - t0) * 1000.0

        n_frames += 1
        for d in dets:
            n_objects_total += 1
            label_counts[d.label] += 1
            conf_sum += float(d.det_conf); conf_n += 1
            est = _roi_median_distance(dmap, d.box)
            if est is not None:
                dist_sum += est; dist_n += 1
    duration = max(1e-6, time.time() - started)

    avg_objects = n_objects_total / n_frames if n_frames else 0.0
    avg_conf = conf_sum / conf_n if conf_n else 0.0
    avg_dist = dist_sum / dist_n if dist_n else 0.0
    avg_fps = n_frames / duration

    top5 = label_counts.most_common(5)
    top5_str = ";".join(f"{k}:{v}" for k, v in top5)

    return {
        "session": os.path.basename(session_dir),
        "environment": env,
        "n_frames": n_frames,
        "failed_frames": failed_frames,
        "avg_objects_detected": round(avg_objects, 3),
        "avg_confidence": round(avg_conf, 3),
        "avg_distance_m": round(avg_dist, 3),
        "top_5_classes": top5_str,
        "avg_fps": round(avg_fps, 2),
        "duration_s": round(duration, 2),
        "lighting_estimate": metadata.get("lighting_estimate", ""),
        "avg_brightness": metadata.get("avg_frame_brightness", ""),
        "avg_sync_delta_ms": metadata.get("avg_sync_delta_ms", ""),
    }


def main():
    ap = argparse.ArgumentParser(description="Batch-analyze recorded stereo sessions across environments")
    ap.add_argument("--recordings-dir", default="recordings")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--max-frames-per-session", type=int, default=120)
    ap.add_argument("--out-csv", default="batch_analysis.csv")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    sessions = _list_session_dirs(args.recordings_dir)
    if not sessions:
        logger.error(f"No session directories found under {args.recordings_dir}")
        return

    config = AppConfig()
    if args.device != "auto":
        config.device_preference = args.device

    logger.info(f"Loading detector + depth on device={config.device}")
    detector = ObjectDetector(config); detector.load()
    depth = DepthEstimator(config); depth.load()

    rows = []
    for sdir in sessions:
        logger.info(f"Analyzing session: {sdir}")
        row = _process_session(sdir, detector, depth, config, args.max_frames_per_session, args.side)
        rows.append(row)
        print(f"  {row['session']:<40s} env={row['environment']:<12s} "
              f"frames={row['n_frames']:<5d} avg_obj={row['avg_objects_detected']:.2f} "
              f"avg_conf={row['avg_confidence']:.2f} avg_dist={row['avg_distance_m']:.2f}m "
              f"avg_fps={row['avg_fps']:.1f}")

    if not rows:
        logger.warning("No sessions analyzed")
        return

    keys = list(rows[0].keys())
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.info(f"Comparison CSV written: {args.out_csv}")


if __name__ == "__main__":
    main()
