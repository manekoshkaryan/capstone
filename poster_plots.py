"""Generate poster-quality result plots from session logs."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from pathlib import Path

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SF Pro Display", "Helvetica Neue", "Arial"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

BG      = "#0F1117"
SURFACE = "#1C1F26"
ACCENT  = "#6C63FF"
GREEN   = "#4CAF8A"
AMBER   = "#F5A623"
RED     = "#E05C5C"
MUTED   = "#6B7280"
TEXT    = "#E8EAF0"

PALETTE = [ACCENT, GREEN, AMBER, RED, "#5BC4E0", "#B07FE8", "#F08080", "#80C080"]

OUT = Path("poster_results")
OUT.mkdir(exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
events, latency_rows = [], []
for line in open("data/logs/events.jsonl"):
    try: events.append(json.loads(line))
    except: pass
for line in open("data/logs/latency.jsonl"):
    try: latency_rows.append(json.loads(line))
    except: pass

announcements = [r for r in latency_rows if r.get("kind") == "announcement"]
all_objects   = [o for e in events for o in e.get("objects", [])]
all_dists     = [o["distance_m"] for o in all_objects if o.get("distance_m") and 0 < o["distance_m"] < 10]
all_confs     = [o["det_conf"]   for o in all_objects]
label_counts  = Counter(o["label"] for o in all_objects)
urgencies     = Counter(a.get("urgency") for a in announcements if a.get("urgency"))
actions_raw   = Counter(a.get("action")  for a in announcements if a.get("action") and a.get("action") != "")
h2a_ms        = [a["hazard_to_audio_ms"] for a in announcements if a.get("hazard_to_audio_ms") and a["hazard_to_audio_ms"] < 5000]
obj_per_frame = [len(e["objects"]) for e in events]


def dark_fig(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    return fig

def dark_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A2D35")
    ax.grid(color="#2A2D35", linestyle="--", alpha=0.5)
    return ax


# ══════════════════════════════════════════════════════════════════════════
# Plot 1 — Top detected object classes (horizontal bar)
# ══════════════════════════════════════════════════════════════════════════
TOP_N = 12
top_labels = label_counts.most_common(TOP_N)
labels_names = [l for l, _ in reversed(top_labels)]
labels_vals  = [c for _, c in reversed(top_labels)]
colors_bar   = [ACCENT if i >= TOP_N - 3 else GREEN for i in range(TOP_N)]

fig = dark_fig(7, 4.5)
ax = dark_ax(fig.add_subplot(111))
bars = ax.barh(labels_names, labels_vals, color=list(reversed(colors_bar)),
               height=0.65, edgecolor="none")
for bar, val in zip(bars, labels_vals):
    ax.text(val + max(labels_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", ha="left", color=TEXT, fontsize=8)
ax.set_xlabel("Detection count", color=TEXT)
ax.set_title("Top Detected Object Classes", color=TEXT, fontweight="bold", pad=10)
ax.set_xlim(0, max(labels_vals) * 1.12)
ax.grid(axis="x", color="#2A2D35", linestyle="--", alpha=0.5)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(OUT / "01_top_classes.png", facecolor=BG)
plt.close()
print("✓ 01_top_classes.png")


# ══════════════════════════════════════════════════════════════════════════
# Plot 2 — Obstacle distance distribution
# ══════════════════════════════════════════════════════════════════════════
fig = dark_fig(6, 3.8)
ax = dark_ax(fig.add_subplot(111))
bins = np.linspace(0, 10, 40)
n, _, patches = ax.hist(all_dists, bins=bins, color=ACCENT, edgecolor=BG, linewidth=0.4, alpha=0.9)
# colour critical zone red
for patch, left in zip(patches, bins):
    if left < 0.8:
        patch.set_facecolor(RED)
    elif left < 2.5:
        patch.set_facecolor(AMBER)

ax.axvline(0.8, color=RED,   linestyle="--", linewidth=1.2, label="STOP threshold (0.8 m)")
ax.axvline(2.5, color=AMBER, linestyle="--", linewidth=1.2, label="Warn threshold (2.5 m)")
ax.set_xlabel("Distance (m)", color=TEXT)
ax.set_ylabel("Detection count", color=TEXT)
ax.set_title("Obstacle Distance Distribution", color=TEXT, fontweight="bold", pad=10)
leg = ax.legend(frameon=True, facecolor=SURFACE, edgecolor="#2A2D35",
                labelcolor=TEXT, fontsize=8)
mu = np.mean(all_dists)
ax.axvline(mu, color=GREEN, linestyle=":", linewidth=1.2)
ax.text(mu + 0.08, ax.get_ylim()[1] * 0.9, f"μ={mu:.2f} m", color=GREEN, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "02_distance_dist.png", facecolor=BG)
plt.close()
print("✓ 02_distance_dist.png")


# ══════════════════════════════════════════════════════════════════════════
# Plot 3 — Guidance urgency breakdown (donut)
# ══════════════════════════════════════════════════════════════════════════
urg_order  = ["critical", "warn", "info"]
urg_colors = [RED, AMBER, GREEN]
urg_vals   = [urgencies.get(u, 0) for u in urg_order]

fig = dark_fig(5, 4)
ax = fig.add_subplot(111, facecolor=BG)
wedges, texts, autotexts = ax.pie(
    urg_vals, labels=urg_order, colors=urg_colors,
    autopct="%1.1f%%", startangle=90,
    wedgeprops=dict(width=0.5, edgecolor=BG, linewidth=2),
    pctdistance=0.75,
)
for t in texts:     t.set_color(TEXT); t.set_fontsize(11)
for at in autotexts: at.set_color(BG); at.set_fontsize(9); at.set_fontweight("bold")
total = sum(urg_vals)
ax.text(0, 0, f"{total}\nguidances", ha="center", va="center",
        color=TEXT, fontsize=10, fontweight="bold")
ax.set_title("Guidance Urgency Distribution", color=TEXT, fontweight="bold", pad=12)
fig.patch.set_facecolor(BG)
fig.tight_layout()
fig.savefig(OUT / "03_urgency_donut.png", facecolor=BG)
plt.close()
print("✓ 03_urgency_donut.png")


# ══════════════════════════════════════════════════════════════════════════
# Plot 4 — Navigation actions
# ══════════════════════════════════════════════════════════════════════════
action_map = {
    "STOP": "Stop", "GO_STRAIGHT": "Go straight",
    "BEAR_LEFT": "Bear left", "BEAR_RIGHT": "Bear right",
    "GO_LEFT": "Go left", "GO_RIGHT": "Go right",
    "MOVE_RIGHT": "Move right",
}
act_items = [(action_map.get(k, k), v) for k, v in actions_raw.most_common() if k in action_map]
act_names = [a for a, _ in act_items]
act_vals  = [v for _, v in act_items]
act_colors = [RED if "Stop" in n else ACCENT for n in act_names]

fig = dark_fig(6, 3.5)
ax = dark_ax(fig.add_subplot(111))
bars = ax.bar(act_names, act_vals, color=act_colors, edgecolor="none", width=0.6)
for bar, val in zip(bars, act_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(val), ha="center", va="bottom", color=TEXT, fontsize=9)
ax.set_ylabel("Count", color=TEXT)
ax.set_title("Navigation Actions Issued", color=TEXT, fontweight="bold", pad=10)
ax.tick_params(axis="x", labelsize=9)
fig.tight_layout()
fig.savefig(OUT / "04_actions.png", facecolor=BG)
plt.close()
print("✓ 04_actions.png")


# ══════════════════════════════════════════════════════════════════════════
# Plot 5 — Hazard-to-audio latency (CDF)
# ══════════════════════════════════════════════════════════════════════════
fig = dark_fig(6, 3.8)
ax = dark_ax(fig.add_subplot(111))
sorted_h2a = np.sort(h2a_ms)
cdf = np.arange(1, len(sorted_h2a) + 1) / len(sorted_h2a)
ax.plot(sorted_h2a, cdf * 100, color=ACCENT, linewidth=2)
ax.fill_between(sorted_h2a, cdf * 100, alpha=0.15, color=ACCENT)

p50 = np.percentile(h2a_ms, 50)
p90 = np.percentile(h2a_ms, 90)
ax.axvline(p50, color=GREEN, linestyle="--", linewidth=1.2, label=f"p50 = {p50:.0f} ms")
ax.axvline(p90, color=AMBER, linestyle="--", linewidth=1.2, label=f"p90 = {p90:.0f} ms")
ax.axvline(1000, color=RED, linestyle=":", linewidth=1, label="1 s target")

ax.set_xlabel("Hazard -> spoken audio latency (ms)", color=TEXT)
ax.set_ylabel("Cumulative % of events", color=TEXT)
ax.set_title("Hazard-to-Speech Latency (CDF)", color=TEXT, fontweight="bold", pad=10)
ax.set_ylim(0, 102)
leg = ax.legend(frameon=True, facecolor=SURFACE, edgecolor="#2A2D35",
                labelcolor=TEXT, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "05_latency_cdf.png", facecolor=BG)
plt.close()
print("✓ 05_latency_cdf.png")


# ══════════════════════════════════════════════════════════════════════════
# Plot 6 — Objects per frame distribution
# ══════════════════════════════════════════════════════════════════════════
fig = dark_fig(6, 3.5)
ax = dark_ax(fig.add_subplot(111))
max_obj = min(max(obj_per_frame), 12)
counts  = Counter(min(c, max_obj) for c in obj_per_frame)
xs = list(range(0, max_obj + 1))
ys = [counts.get(x, 0) for x in xs]
xlabels = [str(x) if x < max_obj else f"≥{max_obj}" for x in xs]
bars = ax.bar(xlabels, ys, color=ACCENT, edgecolor="none", width=0.7)
ax.set_xlabel("Objects detected per frame", color=TEXT)
ax.set_ylabel("Frame count", color=TEXT)
ax.set_title("Objects Detected Per Frame", color=TEXT, fontweight="bold", pad=10)
mu_obj = np.mean(obj_per_frame)
ax.axvline(mu_obj, color=GREEN, linestyle="--", linewidth=1.2)
ax.text(mu_obj + 0.15, max(ys) * 0.9, f"μ={mu_obj:.1f}", color=GREEN, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "06_objects_per_frame.png", facecolor=BG)
plt.close()
print("✓ 06_objects_per_frame.png")


# ══════════════════════════════════════════════════════════════════════════
# Plot 7 — Detection confidence distribution
# ══════════════════════════════════════════════════════════════════════════
fig = dark_fig(6, 3.5)
ax = dark_ax(fig.add_subplot(111))
ax.hist(all_confs, bins=30, color=ACCENT, edgecolor=BG, linewidth=0.4, alpha=0.9)
ax.axvline(np.mean(all_confs), color=GREEN, linestyle="--", linewidth=1.3,
           label=f"mean = {np.mean(all_confs):.3f}")
ax.axvline(0.35, color=AMBER, linestyle=":", linewidth=1.2, label="threshold = 0.35")
ax.set_xlabel("Detection confidence", color=TEXT)
ax.set_ylabel("Count", color=TEXT)
ax.set_title("YOLO Detection Confidence Distribution", color=TEXT, fontweight="bold", pad=10)
leg = ax.legend(frameon=True, facecolor=SURFACE, edgecolor="#2A2D35",
                labelcolor=TEXT, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "07_confidence_dist.png", facecolor=BG)
plt.close()
print("✓ 07_confidence_dist.png")

print(f"\nAll plots saved to: {OUT.resolve()}")
print(f"\nKey stats for poster:")
print(f"  Frames processed  : {len(events):,}")
print(f"  Total detections  : {len(all_objects):,}")
print(f"  Unique classes    : {len(label_counts)}")
print(f"  Total guidances   : {sum(urg_vals)}")
print(f"  STOP actions      : {actions_raw.get('STOP', 0)}")
print(f"  Latency p50       : {np.percentile(h2a_ms,50):.0f} ms")
print(f"  Latency p90       : {np.percentile(h2a_ms,90):.0f} ms")
print(f"  Avg objects/frame : {np.mean(obj_per_frame):.2f}")
print(f"  Avg det confidence: {np.mean(all_confs):.3f}")
