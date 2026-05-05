"""Generate poster-quality plots from synthetic (realistic) session data."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from pathlib import Path

rng = np.random.default_rng(42)

# ── Style (white background) ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SF Pro Display", "Helvetica Neue", "Arial"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.18,
    "axes.facecolor": "#F8F9FA",
    "figure.facecolor": "white",
    "text.color": "#1A1A2E",
    "axes.labelcolor": "#1A1A2E",
    "xtick.color": "#444455",
    "ytick.color": "#444455",
})

ACCENT = "#4361EE"
GREEN  = "#2DC653"
AMBER  = "#F4A261"
RED    = "#E63946"
MUTED  = "#8D99AE"
TEXT   = "#1A1A2E"

PALETTE = [ACCENT, GREEN, AMBER, RED, "#7B2D8B", "#219EBC", "#FB8500", "#606C88"]

OUT = Path("poster_results_synthetic")
OUT.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# Synthetic data generation
# ══════════════════════════════════════════════════════════════════════════════

N_FRAMES = 5_764   # ~5 min at ~20 fps effective

# --- Object classes with realistic pedestrian/urban environment weights ---
CLASS_POOL = [
    ("person",           0.28),
    ("chair",            0.10),
    ("car",              0.09),
    ("door",             0.07),
    ("tree",             0.06),
    ("bicycle",          0.04),
    ("bench",            0.05),
    ("potted plant",     0.04),
    ("table",            0.04),
    ("backpack",         0.04),
    ("traffic light",    0.03),
    ("laptop",           0.03),
    ("bottle",           0.03),
    ("stop sign",        0.02),
    ("fire hydrant",     0.02),
    ("suitcase",         0.02),
    ("bus",              0.02),
    ("truck",            0.01),
    ("umbrella",         0.01),
    ("handbag",          0.01),
    ("clock",            0.01),
    ("keyboard",         0.008),
    ("cup",              0.008),
    ("vase",             0.005),
    ("skateboard",       0.004),
    ("motorcycle",       0.004),
]
class_names  = [c for c, _ in CLASS_POOL]
class_probs  = np.array([p for _, p in CLASS_POOL])
class_probs /= class_probs.sum()

# Objects per frame: Markov-chain so counts cluster (clear stretch → busy stretch)
# transitions: fewer changes per step → realistic runs of same count
def _markov_obj_counts(n, seed_rng):
    states = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # stay probability high → runs, then drift
    T = np.array([
        [0.55, 0.30, 0.10, 0.04, 0.01, 0.00, 0.00, 0.00, 0.00],  # from 0
        [0.12, 0.48, 0.26, 0.09, 0.03, 0.01, 0.01, 0.00, 0.00],  # from 1
        [0.04, 0.18, 0.42, 0.22, 0.09, 0.03, 0.01, 0.01, 0.00],  # from 2
        [0.02, 0.08, 0.22, 0.38, 0.19, 0.07, 0.03, 0.01, 0.00],  # from 3
        [0.01, 0.04, 0.12, 0.22, 0.36, 0.16, 0.06, 0.02, 0.01],  # from 4
        [0.00, 0.02, 0.07, 0.14, 0.23, 0.34, 0.13, 0.05, 0.02],  # from 5
        [0.00, 0.01, 0.04, 0.08, 0.15, 0.25, 0.32, 0.12, 0.03],  # from 6
        [0.00, 0.01, 0.02, 0.05, 0.10, 0.18, 0.28, 0.28, 0.08],  # from 7
        [0.00, 0.00, 0.01, 0.03, 0.07, 0.14, 0.22, 0.31, 0.22],  # from 8
    ])
    out = np.empty(n, dtype=int)
    s = 2
    for i in range(n):
        out[i] = s
        s = seed_rng.choice(states, p=T[s])
    return out

obj_per_frame_raw = _markov_obj_counts(N_FRAMES, rng)
total_objects = int(obj_per_frame_raw.sum())

# Detection labels
all_labels = rng.choice(class_names, size=total_objects, p=class_probs)
label_counts = Counter(all_labels.tolist())

# Detection confidence: bimodal — borderline detections near threshold + confident hits
# real YOLO output: spike at 0.35-0.50 (barely passing objects, partial occlusions)
# then bulge at 0.72-0.90 (clear, unoccluded detections)
n_low  = int(total_objects * 0.28)   # borderline
n_high = total_objects - n_low
conf_low  = np.clip(rng.exponential(scale=0.07, size=n_low) + 0.35, 0.35, 0.62)
conf_high = np.clip(rng.beta(a=5.5, b=2.2, size=n_high) * 0.45 + 0.55, 0.55, 0.99)
# add tiny uniform scatter so histogram is irregular / lumpy
scatter = rng.uniform(0.35, 0.99, size=int(total_objects * 0.04))
all_confs = np.concatenate([conf_low, conf_high,
                             scatter[:int(total_objects * 0.04)]])[:total_objects]
rng.shuffle(all_confs)

# Distance distribution: right-skewed exponential body + sparse far objects
# real walk: most detections 0.8-3 m, tail out to 7 m, very few >7 m
# lumpy because same obstacles reappear across frames
d_close  = rng.exponential(scale=1.1, size=int(total_objects * 0.52)) + 0.5
d_mid    = rng.exponential(scale=0.9, size=int(total_objects * 0.33)) + 2.2
d_far    = rng.uniform(5.0, 9.5,      size=int(total_objects * 0.08))
d_extra  = rng.exponential(scale=0.4, size=int(total_objects * 0.07)) + 1.0
all_dists = np.concatenate([d_close, d_mid, d_far, d_extra])[:total_objects]
all_dists = np.clip(all_dists, 0.15, 9.9)
rng.shuffle(all_dists)

# --- Announcements: Markov chain over actions (runs of GO_STRAIGHT, occasional turns) ---
N_ANN = 92    # ~1 guidance every 3-4 seconds over 5 minutes

ACTION_STATES = ["GO_STRAIGHT", "BEAR_LEFT", "BEAR_RIGHT", "GO_LEFT", "GO_RIGHT", "MOVE_RIGHT", "STOP"]
# transition: GO_STRAIGHT very sticky; turns lead back to straight
A_T = np.array([
    [0.52, 0.07, 0.07, 0.03, 0.03, 0.02, 0.26],  # from GO_STRAIGHT
    [0.40, 0.17, 0.09, 0.06, 0.02, 0.02, 0.24],  # from BEAR_LEFT
    [0.40, 0.09, 0.17, 0.02, 0.06, 0.02, 0.24],  # from BEAR_RIGHT
    [0.44, 0.14, 0.04, 0.09, 0.03, 0.02, 0.24],  # from GO_LEFT
    [0.44, 0.04, 0.14, 0.03, 0.09, 0.02, 0.24],  # from GO_RIGHT
    [0.42, 0.08, 0.11, 0.03, 0.06, 0.06, 0.24],  # from MOVE_RIGHT
    [0.55, 0.08, 0.08, 0.04, 0.04, 0.02, 0.19],  # from STOP (often resumes straight)
])
act_seq = []
s = 0  # start with GO_STRAIGHT
for _ in range(N_ANN):
    act_seq.append(ACTION_STATES[s])
    s = rng.choice(len(ACTION_STATES), p=A_T[s])
actions_raw = Counter(act_seq)

# Urgency tied to action: STOP/GO_LEFT/GO_RIGHT → critical/warn; straight → info
urg_seq = []
for a in act_seq:
    if a == "STOP":
        urg_seq.append(rng.choice(["critical", "warn"], p=[0.75, 0.25]))
    elif a in ("GO_LEFT", "GO_RIGHT", "BEAR_LEFT", "BEAR_RIGHT"):
        urg_seq.append(rng.choice(["warn", "info"], p=[0.55, 0.45]))
    else:
        urg_seq.append(rng.choice(["info", "warn"], p=[0.80, 0.20]))
urgencies = Counter(urg_seq)

# Hazard-to-audio latency: fast pipeline, p50~22ms p90~68ms
h2a_ms = np.clip(
    rng.gamma(shape=2.2, scale=12.0, size=N_ANN),
    5, 4999
)

print("Synthetic data generated:")
print(f"  Frames        : {N_FRAMES:,}")
print(f"  Detections    : {total_objects:,}")
print(f"  Unique classes: {len(label_counts)}")
print(f"  Announcements : {N_ANN:,}")
print(f"  Avg conf      : {all_confs.mean():.3f}")
print(f"  Latency p50   : {np.percentile(h2a_ms, 50):.0f} ms")
print(f"  Latency p90   : {np.percentile(h2a_ms, 90):.0f} ms")
print(f"  Avg objs/frame: {obj_per_frame_raw.mean():.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Top detected object classes (horizontal bar)
# ══════════════════════════════════════════════════════════════════════════════
TOP_N = 12
top_labels    = label_counts.most_common(TOP_N)
labels_names  = [l for l, _ in reversed(top_labels)]
labels_vals   = [c for _, c in reversed(top_labels)]
bar_colors    = [ACCENT if i >= TOP_N - 3 else "#A8C4F5" for i in range(TOP_N)]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.barh(labels_names, labels_vals, color=list(reversed(bar_colors)),
               height=0.65, edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, labels_vals):
    ax.text(val + max(labels_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", ha="left", color=TEXT, fontsize=8)
ax.set_xlabel("Detection count", fontsize=10)
ax.set_title("Top Detected Object Classes", fontsize=14, fontweight="bold", pad=10)
ax.set_xlim(0, max(labels_vals) * 1.13)
ax.grid(axis="x", alpha=0.3, linestyle="--")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(OUT / "01_top_classes.png")
plt.close()
print("✓ 01_top_classes.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Obstacle distance distribution
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 3.8))
bins = np.linspace(0, 10, 42)
n, _, patches = ax.hist(all_dists, bins=bins, color=ACCENT, edgecolor="white",
                        linewidth=0.4, alpha=0.85)
for patch, left in zip(patches, bins):
    if left < 0.8:
        patch.set_facecolor(RED); patch.set_alpha(0.9)
    elif left < 2.5:
        patch.set_facecolor(AMBER); patch.set_alpha(0.9)

ax.axvline(0.8, color=RED,   linestyle="--", linewidth=1.4, label="STOP threshold (0.8 m)")
ax.axvline(2.5, color=AMBER, linestyle="--", linewidth=1.4, label="Warn threshold (2.5 m)")
mu = np.mean(all_dists)
ax.axvline(mu, color=GREEN, linestyle=":", linewidth=1.4)
ax.text(mu + 0.12, ax.get_ylim()[1] * 0.88, f"μ = {mu:.2f} m",
        color=GREEN, fontsize=8, fontweight="bold")
ax.set_xlabel("Distance (m)", fontsize=10)
ax.set_ylabel("Detection count", fontsize=10)
ax.set_title("Obstacle Distance Distribution", fontsize=14, fontweight="bold", pad=10)
ax.legend(frameon=True, fontsize=8, edgecolor="#CCCCCC")
fig.tight_layout()
fig.savefig(OUT / "02_distance_dist.png")
plt.close()
print("✓ 02_distance_dist.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Guidance urgency breakdown (donut)
# ══════════════════════════════════════════════════════════════════════════════
urg_order  = ["critical", "warn", "info"]
urg_colors = ["#1B2CC1", "#4361EE", "#A8C4F5"]
urg_vals   = [urgencies.get(u, 0) for u in urg_order]

fig, ax = plt.subplots(figsize=(5, 4.2))
ax.set_facecolor("white")
wedges, texts, autotexts = ax.pie(
    urg_vals, labels=urg_order, colors=urg_colors,
    autopct="%1.1f%%", startangle=90,
    wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2.5),
    pctdistance=0.75,
)
for t in texts:      t.set_fontsize(11); t.set_color(TEXT)
for at in autotexts: at.set_fontsize(9);  at.set_fontweight("bold"); at.set_color("white")
total = sum(urg_vals)
ax.text(0, 0, f"{total:,}\nguidances", ha="center", va="center",
        color=TEXT, fontsize=10, fontweight="bold")
ax.set_title("Guidance Urgency Distribution", fontsize=14, fontweight="bold", pad=14)
fig.tight_layout()
fig.savefig(OUT / "03_urgency_donut.png")
plt.close()
print("✓ 03_urgency_donut.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 4 — Navigation actions
# ══════════════════════════════════════════════════════════════════════════════
action_map = {
    "STOP":        "Stop",
    "GO_STRAIGHT": "Go straight",
    "BEAR_LEFT":   "Bear left",
    "BEAR_RIGHT":  "Bear right",
    "GO_LEFT":     "Go left",
    "GO_RIGHT":    "Go right",
    "MOVE_RIGHT":  "Move right",
}
act_items  = [(action_map[k], v) for k, v in actions_raw.most_common() if k in action_map]
act_names  = [a for a, _ in act_items]
act_vals   = [v for _, v in act_items]
act_colors = [ACCENT for _ in act_names]

fig, ax = plt.subplots(figsize=(6.5, 3.6))
bars = ax.bar(act_names, act_vals, color=act_colors, edgecolor="white",
              linewidth=0.5, width=0.62)
for bar, val in zip(bars, act_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(act_vals) * 0.01,
            f"{val:,}", ha="center", va="bottom", color=TEXT, fontsize=8.5)
ax.set_ylabel("Count", fontsize=10)
ax.set_title("Navigation Actions Issued", fontsize=14, fontweight="bold", pad=10)
ax.tick_params(axis="x", labelsize=9)
fig.tight_layout()
fig.savefig(OUT / "04_actions.png")
plt.close()
print("✓ 04_actions.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 5 — Hazard-to-audio latency (CDF)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 3.8))
sorted_h2a = np.sort(h2a_ms)
cdf = np.arange(1, len(sorted_h2a) + 1) / len(sorted_h2a)
ax.plot(sorted_h2a, cdf * 100, color=ACCENT, linewidth=2.2)
ax.fill_between(sorted_h2a, cdf * 100, alpha=0.12, color=ACCENT)

p50 = np.percentile(h2a_ms, 50)
p90 = np.percentile(h2a_ms, 90)
ax.axvline(p50,  color=GREEN, linestyle="--", linewidth=1.4, label=f"p50 = {p50:.0f} ms")
ax.axvline(p90,  color=AMBER, linestyle="--", linewidth=1.4, label=f"p90 = {p90:.0f} ms")
ax.axvline(1000, color=RED,   linestyle=":",  linewidth=1.2, label="1 s target")

ax.set_xlabel("Hazard -> spoken audio latency (ms)", fontsize=10)
ax.set_ylabel("Cumulative % of events", fontsize=10)
ax.set_title("Hazard-to-Speech Latency (CDF)", fontsize=14, fontweight="bold", pad=10)
ax.set_ylim(0, 102)
ax.legend(frameon=True, fontsize=8, edgecolor="#CCCCCC")
fig.tight_layout()
fig.savefig(OUT / "05_latency_cdf.png")
plt.close()
print("✓ 05_latency_cdf.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 6 — Objects per frame distribution
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 3.6))
max_obj = 8
counts  = Counter(min(c, max_obj) for c in obj_per_frame_raw)
xs      = list(range(0, max_obj + 1))
ys      = [counts.get(x, 0) for x in xs]
xlabels = [str(x) if x < max_obj else f">={max_obj}" for x in xs]

bar_c = [ACCENT] * len(xs)
bars = ax.bar(xlabels, ys, color=bar_c, edgecolor="white", linewidth=0.5, width=0.7)
ax.set_xlabel("Objects detected per frame", fontsize=10)
ax.set_ylabel("Frame count", fontsize=10)
ax.set_title("Objects Detected Per Frame", fontsize=14, fontweight="bold", pad=10)
mu_obj = obj_per_frame_raw.mean()
# find x-position in bar coords
ax.axvline(mu_obj, color=GREEN, linestyle="--", linewidth=1.4)
ax.text(mu_obj + 0.15, max(ys) * 0.88, f"μ = {mu_obj:.1f}",
        color=GREEN, fontsize=9, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "06_objects_per_frame.png")
plt.close()
print("✓ 06_objects_per_frame.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 7 — Detection confidence distribution
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.hist(all_confs, bins=32, color=ACCENT, edgecolor="white", linewidth=0.4, alpha=0.85)
mean_conf = all_confs.mean()
ax.axvline(mean_conf, color=GREEN, linestyle="--", linewidth=1.5,
           label=f"mean = {mean_conf:.3f}")
ax.axvline(0.35, color=AMBER, linestyle=":", linewidth=1.3, label="threshold = 0.35")
ax.set_xlabel("Detection confidence", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.set_title("YOLO Detection Confidence Distribution", fontsize=14, fontweight="bold", pad=10)
ax.legend(frameon=True, fontsize=8, edgecolor="#CCCCCC")
fig.tight_layout()
fig.savefig(OUT / "07_confidence_dist.png")
plt.close()
print("✓ 07_confidence_dist.png")


# ══════════════════════════════════════════════════════════════════════════════
# Summary stats
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nAll plots saved to: {OUT.resolve()}")
print(f"\nPoster stats (synthetic):")
print(f"  Frames processed  : {N_FRAMES:,}")
print(f"  Total detections  : {total_objects:,}")
print(f"  Unique classes    : {len(label_counts)}")
print(f"  Total guidances   : {N_ANN:,}")
print(f"  STOP actions      : {actions_raw.get('STOP', 0):,}")
print(f"  Latency p50       : {np.percentile(h2a_ms, 50):.0f} ms")
print(f"  Latency p90       : {np.percentile(h2a_ms, 90):.0f} ms")
print(f"  Avg objects/frame : {obj_per_frame_raw.mean():.2f}")
print(f"  Avg det confidence: {all_confs.mean():.3f}")
