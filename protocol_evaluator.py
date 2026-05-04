import csv
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Iterable

from communication_protocols import (
    PerceptionInput,
    ProtocolMode,
    ProtocolMessage,
    render,
    render_all,
    list_modes,
    DIRECT_SYSTEM_NAME,
    CASCADED_SYSTEM_NAME,
    cascaded_simulated_latency_ms,
    latency_improvement_pct,
)
from protocol_logger import METRICS_FIELDS

logger = logging.getLogger(__name__)


SCENARIO_OUTPUT_FILE = "scenario_protocol_outputs.csv"
SUMMARY_CSV = "protocol_summary.csv"
SUMMARY_JSON = "protocol_summary.json"
ACCURACY_CSV = "protocol_accuracy.csv"
GROUND_TRUTH_FILE = "ground_truth_protocol.csv"
DEFAULT_PLOT_DIR = os.path.join("outputs", "protocol_plots")


@dataclass(frozen=True)
class Scenario:
    scenario_id: int
    description: str
    perception: PerceptionInput


def builtin_scenarios() -> List[Scenario]:
    return [
        Scenario(
            1, "chair center 0.6m",
            PerceptionInput(label="chair", distance_m=0.6, direction="center", urgency="critical"),
        ),
        Scenario(
            2, "person left 1.2m",
            PerceptionInput(label="person", distance_m=1.2, direction="left", urgency="warn"),
        ),
        Scenario(
            3, "table right 1.8m",
            PerceptionInput(label="table", distance_m=1.8, direction="right", urgency="warn"),
        ),
        Scenario(
            4, "no object",
            PerceptionInput(label="", distance_m=None, direction="ahead", urgency="info", front_clear=True),
        ),
        Scenario(
            5, "door center 3.0m",
            PerceptionInput(label="door", distance_m=3.0, direction="center", urgency="info"),
        ),
        Scenario(
            6, "obstacle center 0.7m left side open",
            PerceptionInput(
                label="obstacle", distance_m=0.7, direction="center", urgency="critical",
                front_clear=False, left_clear_m=2.5, right_clear_m=0.4,
            ),
        ),
        Scenario(
            7, "obstacle center 0.7m right side open",
            PerceptionInput(
                label="obstacle", distance_m=0.7, direction="center", urgency="critical",
                front_clear=False, left_clear_m=0.4, right_clear_m=2.5,
            ),
        ),
        Scenario(
            8, "two objects, closest is person at 0.9m",
            PerceptionInput(
                label="person", distance_m=0.9, direction="center", urgency="warn",
                front_clear=False, left_clear_m=1.8, right_clear_m=2.0,
            ),
        ),
    ]


def run_scenarios(
    scenarios: Optional[Iterable[Scenario]] = None,
    output_path: str = SCENARIO_OUTPUT_FILE,
) -> List[Dict[str, Any]]:
    scenarios = list(scenarios) if scenarios is not None else builtin_scenarios()
    rows: List[Dict[str, Any]] = []
    fieldnames = [
        "scenario_id", "scenario_description", "protocol",
        "input_label", "input_distance", "input_direction",
        "message", "word_count", "action_category",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sc in scenarios:
            for mode in list_modes():
                msg = render(mode, sc.perception)
                row = {
                    "scenario_id": sc.scenario_id,
                    "scenario_description": sc.description,
                    "protocol": msg.protocol,
                    "input_label": sc.perception.label,
                    "input_distance": (
                        sc.perception.distance_m
                        if sc.perception.distance_m is not None
                        else ""
                    ),
                    "input_direction": sc.perception.direction,
                    "message": msg.text,
                    "word_count": msg.word_count,
                    "action_category": msg.action_category,
                }
                writer.writerow(row)
                rows.append(row)
    logger.info(f"Scenario outputs saved: {output_path}")
    return rows


def _safe_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _safe_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None or v == "":
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "t")


def load_metrics_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def summarize_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = row.get("protocol") or "unknown"
        grouped.setdefault(key, []).append(row)

    summary: Dict[str, Dict[str, Any]] = {}
    for protocol, items in grouped.items():
        response = [r for r in (_safe_float(x.get("response_latency_ms")) for x in items) if r is not None]
        speech = [r for r in (_safe_float(x.get("speech_latency_ms")) for x in items) if r is not None]
        decision = [r for r in (_safe_float(x.get("decision_latency_ms")) for x in items) if r is not None]
        detection = [r for r in (_safe_float(x.get("detection_latency_ms")) for x in items) if r is not None]
        cascaded = [r for r in (_safe_float(x.get("cascaded_latency_ms")) for x in items) if r is not None]
        improvement = [r for r in (_safe_float(x.get("improvement_pct")) for x in items) if r is not None]
        word_counts = [_safe_int(x.get("word_count")) for x in items]
        actions = [str(x.get("action_category") or "").lower() for x in items]
        stop_count = sum(1 for a in actions if a == "stop")
        directional_count = sum(1 for a in actions if a == "directional")
        clear_count = sum(1 for a in actions if a == "clear")
        info_count = sum(1 for a in actions if a == "info")
        duplicate_count = sum(1 for x in items if _safe_bool(x.get("is_duplicate")))
        overlap_count = sum(1 for x in items if _safe_bool(x.get("overlap_skipped")))
        summary[protocol] = {
            "protocol": protocol,
            "messages": len(items),
            "avg_response_latency_ms": _mean(response),
            "avg_speech_latency_ms": _mean(speech),
            "avg_decision_latency_ms": _mean(decision),
            "avg_detection_latency_ms": _mean(detection),
            "avg_cascaded_latency_ms": _mean(cascaded),
            "avg_improvement_pct": _mean(improvement),
            "avg_word_count": _mean(word_counts),
            "stop_warnings": stop_count,
            "directional_commands": directional_count,
            "clear_path_messages": clear_count,
            "info_messages": info_count,
            "duplicate_speech_events": duplicate_count,
            "overlapping_speech_errors": overlap_count,
        }
    return summary


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(statistics.mean(values), 3)


def write_summary(
    summary: Dict[str, Dict[str, Any]],
    csv_path: str = SUMMARY_CSV,
    json_path: str = SUMMARY_JSON,
) -> None:
    fieldnames = [
        "protocol", "messages",
        "avg_response_latency_ms", "avg_speech_latency_ms",
        "avg_decision_latency_ms", "avg_detection_latency_ms",
        "avg_cascaded_latency_ms", "avg_improvement_pct",
        "avg_word_count",
        "stop_warnings", "directional_commands", "clear_path_messages",
        "info_messages",
        "duplicate_speech_events", "overlapping_speech_errors",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for protocol in sorted(summary.keys()):
            writer.writerow(summary[protocol])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Protocol summary saved: {csv_path}, {json_path}")


def generate_plots(
    summary: Dict[str, Dict[str, Any]],
    rows: List[Dict[str, Any]],
    plot_dir: str = DEFAULT_PLOT_DIR,
) -> List[str]:
    os.makedirs(plot_dir, exist_ok=True)
    saved: List[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"matplotlib unavailable, skipping plots: {e}")
        return saved

    if not summary:
        logger.info("Empty protocol summary; skipping plot generation.")
        return saved

    protocols = sorted(summary.keys())

    latency_path = os.path.join(plot_dir, "average_latency_by_protocol.png")
    latency_values = [summary[p].get("avg_response_latency_ms") or 0.0 for p in protocols]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(protocols, latency_values, color="#3a7bd5")
    ax.set_title("Average Response Latency by Protocol")
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Protocol")
    fig.tight_layout()
    fig.savefig(latency_path, dpi=120)
    plt.close(fig)
    saved.append(latency_path)

    word_path = os.path.join(plot_dir, "average_word_count_by_protocol.png")
    word_values = [summary[p].get("avg_word_count") or 0.0 for p in protocols]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(protocols, word_values, color="#33a467")
    ax.set_title("Average Word Count by Protocol")
    ax.set_ylabel("Words")
    ax.set_xlabel("Protocol")
    fig.tight_layout()
    fig.savefig(word_path, dpi=120)
    plt.close(fig)
    saved.append(word_path)

    dist_path = os.path.join(plot_dir, "message_type_distribution_by_protocol.png")
    categories = ["stop_warnings", "directional_commands", "clear_path_messages", "info_messages"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.2
    xs = list(range(len(protocols)))
    for i, cat in enumerate(categories):
        values = [summary[p].get(cat, 0) for p in protocols]
        ax.bar(
            [x + (i - 1.5) * width for x in xs], values, width,
            label=cat.replace("_", " "),
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(protocols)
    ax.set_title("Message Type Distribution by Protocol")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(dist_path, dpi=120)
    plt.close(fig)
    saved.append(dist_path)

    comp_path = os.path.join(plot_dir, "actual_vs_simulated_cascaded_latency.png")
    actual_values = [summary[p].get("avg_response_latency_ms") or 0.0 for p in protocols]
    cascaded_values = [summary[p].get("avg_cascaded_latency_ms") or 0.0 for p in protocols]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    xs = list(range(len(protocols)))
    ax.bar([x - width / 2 for x in xs], actual_values, width,
           label=DIRECT_SYSTEM_NAME, color="#3a7bd5")
    ax.bar([x + width / 2 for x in xs], cascaded_values, width,
           label=CASCADED_SYSTEM_NAME, color="#d54a3a")
    ax.set_xticks(xs)
    ax.set_xticklabels(protocols)
    ax.set_title("Actual vs Simulated Cascaded Latency")
    ax.set_ylabel("Latency (ms)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(comp_path, dpi=120)
    plt.close(fig)
    saved.append(comp_path)

    logger.info(f"Saved {len(saved)} plot(s) to {plot_dir}")
    return saved


def evaluate_with_ground_truth(
    metrics_rows: List[Dict[str, Any]],
    ground_truth_path: str = GROUND_TRUTH_FILE,
    accuracy_path: str = ACCURACY_CSV,
) -> Optional[Dict[str, Any]]:
    if not os.path.exists(ground_truth_path):
        msg = "Ground truth file not found; annotation-based accuracy skipped."
        logger.info(msg)
        return None
    truth: Dict[str, Dict[str, str]] = {}
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = str(row.get("frame_id") or "").strip()
            if not fid:
                continue
            truth[fid] = {
                "expected_label": str(row.get("expected_label") or "").strip().lower(),
                "expected_action": str(row.get("expected_action") or "").strip().lower(),
                "expected_direction": str(row.get("expected_direction") or "").strip().lower(),
            }
    if not truth:
        logger.info("Ground truth file empty; skipping accuracy evaluation.")
        return None

    rows: List[Dict[str, Any]] = []
    label_correct = 0
    action_correct = 0
    direction_correct = 0
    overall_correct = 0
    matched = 0
    for row in metrics_rows:
        fid = str(row.get("frame_id") or row.get("ts") or "").strip()
        gt = truth.get(fid)
        if gt is None:
            continue
        matched += 1
        actual_label = str(row.get("closest_label") or "").strip().lower()
        actual_action = str(row.get("action_category") or "").strip().lower()
        actual_direction = str(row.get("direction") or "").strip().lower()
        l_ok = bool(gt["expected_label"]) and (gt["expected_label"] in actual_label or actual_label in gt["expected_label"])
        a_ok = bool(gt["expected_action"]) and (gt["expected_action"] == actual_action)
        d_ok = bool(gt["expected_direction"]) and (gt["expected_direction"] in actual_direction or actual_direction in gt["expected_direction"])
        if l_ok:
            label_correct += 1
        if a_ok:
            action_correct += 1
        if d_ok:
            direction_correct += 1
        if l_ok and a_ok and d_ok:
            overall_correct += 1
        rows.append({
            "frame_id": fid,
            "expected_label": gt["expected_label"],
            "actual_label": actual_label,
            "label_match": l_ok,
            "expected_action": gt["expected_action"],
            "actual_action": actual_action,
            "action_match": a_ok,
            "expected_direction": gt["expected_direction"],
            "actual_direction": actual_direction,
            "direction_match": d_ok,
            "overall_correct": l_ok and a_ok and d_ok,
        })

    if matched == 0:
        logger.info("Ground truth provided but no frames matched metrics.")
        return None

    fieldnames = [
        "frame_id",
        "expected_label", "actual_label", "label_match",
        "expected_action", "actual_action", "action_match",
        "expected_direction", "actual_direction", "direction_match",
        "overall_correct",
    ]
    with open(accuracy_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    summary = {
        "matched_frames": matched,
        "label_accuracy": round(label_correct / matched, 4),
        "action_accuracy": round(action_correct / matched, 4),
        "direction_accuracy": round(direction_correct / matched, 4),
        "overall_correctness": round(overall_correct / matched, 4),
    }
    logger.info(f"Protocol accuracy saved: {accuracy_path} ({summary})")
    return summary


@dataclass
class EvaluationResult:
    summary: Dict[str, Dict[str, Any]]
    accuracy: Optional[Dict[str, Any]]
    plot_paths: List[str]
    scenario_rows: List[Dict[str, Any]]
    metrics_rows: int


def run_evaluation(
    metrics_path: str = "protocol_metrics.csv",
    plot_dir: str = DEFAULT_PLOT_DIR,
    summary_csv: str = SUMMARY_CSV,
    summary_json: str = SUMMARY_JSON,
    scenario_output: str = SCENARIO_OUTPUT_FILE,
    ground_truth_path: str = GROUND_TRUTH_FILE,
    accuracy_path: str = ACCURACY_CSV,
    include_scenarios: bool = True,
) -> EvaluationResult:
    metrics_rows = load_metrics_csv(metrics_path)
    scenario_rows: List[Dict[str, Any]] = []
    if include_scenarios:
        scenario_rows = run_scenarios(output_path=scenario_output)
    summary = summarize_metrics(metrics_rows)
    if not summary and include_scenarios:
        summary = summarize_metrics(_metrics_rows_from_scenarios(scenario_rows))
    write_summary(summary, summary_csv, summary_json)
    plot_paths = generate_plots(summary, metrics_rows, plot_dir)
    accuracy = evaluate_with_ground_truth(metrics_rows, ground_truth_path, accuracy_path)
    return EvaluationResult(
        summary=summary,
        accuracy=accuracy,
        plot_paths=plot_paths,
        scenario_rows=scenario_rows,
        metrics_rows=len(metrics_rows),
    )


def _metrics_rows_from_scenarios(scenario_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    base_ts = time.time()
    for r in scenario_rows:
        out.append({
            "ts": base_ts,
            "protocol": r.get("protocol", ""),
            "selected_protocol": r.get("protocol", ""),
            "source": "scenario",
            "device": "scenario",
            "message": r.get("message", ""),
            "word_count": r.get("word_count", 0),
            "closest_label": r.get("input_label", ""),
            "distance_m": r.get("input_distance", ""),
            "direction": r.get("input_direction", ""),
            "action_category": r.get("action_category", ""),
            "detection_latency_ms": "",
            "decision_latency_ms": "",
            "speech_latency_ms": "",
            "response_latency_ms": "",
            "cascaded_latency_ms": "",
            "improvement_pct": "",
            "is_duplicate": False,
            "overlap_skipped": False,
        })
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = run_evaluation()
    print(json.dumps(result.summary, indent=2))
    if result.accuracy:
        print("Accuracy:", result.accuracy)
