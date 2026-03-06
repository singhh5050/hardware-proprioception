#!/usr/bin/env python3
"""Additional analysis plots:
  1. Compression ratio vs accuracy scatter
  2. Tier split impact on latency across hardware families

Usage (from repo root):
    python scripts/plot_analysis.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

ACCURACY_JSONL = "results/accuracy_results_final.jsonl"
LATENCY_JSONL  = "results/latency_simulation.jsonl"
OUTPUT_DIR     = Path("results")

# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------
FAMILY_COLOR = {
    "full_cache":     "#2196F3",
    "full_cache_int8":"#2196F3",
    "window":         "#FF9800",
    "h2o":            "#4CAF50",
    "snapkv":         "#9C27B0",
    "expected":       "#F44336",
}
FAMILY_MARKER = {
    "full_cache":     "D",
    "full_cache_int8":"D",
    "window":         "o",
    "h2o":            "s",
    "snapkv":         "^",
    "expected":       "P",
}

def _family(name: str) -> str:
    for prefix in ("full_cache_int8", "full_cache", "window", "h2o", "snapkv", "expected"):
        if name.startswith(prefix):
            return prefix
    return "other"

def _color(name: str) -> str:
    return FAMILY_COLOR.get(_family(name), "#607D8B")

def _marker(name: str) -> str:
    return FAMILY_MARKER.get(_family(name), "o")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
accuracy_records: list[dict] = []
with open(ACCURACY_JSONL) as f:
    for line in f:
        accuracy_records.append(json.loads(line))

latency_records: list[dict] = []
with open(LATENCY_JSONL) as f:
    for line in f:
        latency_records.append(json.loads(line))


# ===========================================================================
# Plot 1: Compression ratio vs accuracy
# ===========================================================================
# Compression ratio = mean(cache_size_at_end / (prompt_tokens + tokens_generated))
# per strategy.  Values near 1.0 = full cache; lower = more aggressive compression.

by_strat: dict[str, list[dict]] = defaultdict(list)
for r in accuracy_records:
    by_strat[r["strategy_name"]].append(r)

STRATEGY_ORDER = [
    "full_cache", "full_cache_int8",
    "window_128", "window_256", "window_512", "window_1024",
    "h2o_128", "h2o_256", "h2o_512", "h2o_1024",
    "snapkv_512", "expected_attn_512",
]

comp_ratios, accuracies, labels = [], [], []
for name in STRATEGY_ORDER:
    rows = by_strat[name]
    ratio = np.mean([
        min(1.0, r["cache_size_at_end"] / (r["prompt_tokens"] + r["tokens_generated"]))
        for r in rows
    ])
    acc = sum(r["correct"] for r in rows) / len(rows)
    comp_ratios.append(float(ratio))
    accuracies.append(float(acc))
    labels.append(name)

fig, ax = plt.subplots(figsize=(9, 6))

for x, y, name in zip(comp_ratios, accuracies, labels):
    ax.scatter(x, y, color=_color(name), marker=_marker(name), s=110, zorder=3,
               edgecolors="white", linewidths=0.6)

# Label each point; nudge to avoid overlap
NUDGE: dict[str, tuple[float, float]] = {
    "full_cache":         ( 0.005,  0.006),
    "full_cache_int8":    ( 0.005, -0.014),
    "window_1024":        (-0.005,  0.008),
    "h2o_1024":           (-0.005, -0.014),
    "snapkv_512":         ( 0.005, -0.013),
    "expected_attn_512":  ( 0.005,  0.008),
}
for x, y, name in zip(comp_ratios, accuracies, labels):
    dx, dy = NUDGE.get(name, (0.005, 0.006))
    ax.annotate(
        name,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        fontsize=7.5,
        va="center",
    )

# Reference line: full_cache accuracy ceiling
ceiling = accuracies[labels.index("full_cache")]
ax.axhline(y=ceiling, color="#2196F3", linestyle="--", alpha=0.4, linewidth=1,
           label=f"full_cache ceiling ({ceiling:.1%})")

# Legend for families
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="D", color="w", markerfacecolor="#2196F3", markersize=8, label="Baseline"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF9800", markersize=8, label="StreamingLLM (window)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#4CAF50", markersize=8, label="H2O"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#9C27B0", markersize=8, label="SnapKV"),
    Line2D([0], [0], marker="P", color="w", markerfacecolor="#F44336", markersize=8, label="ExpectedAttn"),
]
ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

ax.set_xlabel("Mean compression ratio  (tokens kept / tokens generated)", fontsize=10)
ax.set_ylabel("Accuracy (MATH-500, 200 tasks)", fontsize=10)
ax.set_title("Compression Ratio vs Accuracy — KV Cache Strategies\n(Qwen2.5-Math-7B-Instruct)", fontsize=11)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_xlim(-0.02, 1.12)
ax.set_ylim(0.20, 0.85)
ax.grid(alpha=0.25)

fig.tight_layout()
out1 = OUTPUT_DIR / "compression_vs_accuracy.png"
fig.savefig(out1, dpi=200)
plt.close(fig)
print(f"Saved {out1}")


# ===========================================================================
# Plot 2: Tier split impact on latency
# ===========================================================================
# Show how pushing tokens from HBM → CPU → disk raises latency.
# Use full_cache (the most tokens in cache = largest sensitivity to tier placement).
# One line per representative hardware; disk-capable and non-disk hardware
# are visually distinguished.

# Hardware to show and how to group them
HW_LINES: dict[str, dict] = {
    "H100_SXM":           {"label": "H100 SXM",        "color": "#76B900", "ls": "-",  "lw": 2.0},
    "A100_80GB":          {"label": "A100 80GB",        "color": "#9CCC65", "ls": "--", "lw": 1.5},
    "MI300X":             {"label": "MI300X",           "color": "#EF6C00", "ls": "-",  "lw": 2.0},
    "TPU_v6e":            {"label": "TPU v6e",          "color": "#1565C0", "ls": "-",  "lw": 2.0},
    "L40S":               {"label": "L40S",             "color": "#AB47BC", "ls": "--", "lw": 1.5},
    "M4_Max":             {"label": "M4 Max (no disk)", "color": "#F06292", "ls": ":",  "lw": 1.8},
    "Snapdragon_X_Elite": {"label": "Snapdragon X Elite (no disk)", "color": "#FF7043", "ls": ":", "lw": 1.8},
}

# Canonical x-axis order for tier splits
SPLIT_LABELS = [
    (1.0, 0.0, 0.0, "All HBM"),
    (0.7, 0.3, 0.0, "70% HBM\n30% CPU"),
    (0.5, 0.5, 0.0, "50% HBM\n50% CPU"),
    (0.3, 0.3, 0.4, "30% HBM\n30% CPU\n40% Disk"),
    (0.5, 0.0, 0.5, "50% HBM\n50% Disk"),
]

STRATEGY = "full_cache"

# Build lookup: (hardware, hbm_f, cpu_f, disk_f) -> mean_latency_ms
lookup: dict[tuple, float] = {}
for r in latency_records:
    if r["strategy"] == STRATEGY:
        key = (r["hardware"], round(r["hbm_frac"], 2), round(r["cpu_frac"], 2), round(r["disk_frac"], 2))
        lookup[key] = r["mean_latency_ms"]

fig, ax = plt.subplots(figsize=(9, 5.5))

x_positions = list(range(len(SPLIT_LABELS)))

for hw_name, style in HW_LINES.items():
    xs, ys = [], []
    for i, (hf, cf, df, _) in enumerate(SPLIT_LABELS):
        key = (hw_name, round(hf, 2), round(cf, 2), round(df, 2))
        val = lookup.get(key)
        if val is not None:
            xs.append(i)
            ys.append(val)
    if xs:
        ax.plot(xs, ys,
                color=style["color"],
                linestyle=style["ls"],
                linewidth=style["lw"],
                marker="o",
                markersize=5,
                label=style["label"])

ax.set_xticks(x_positions)
ax.set_xticklabels([lbl for *_, lbl in SPLIT_LABELS], fontsize=8)
ax.set_ylabel("Mean latency per decode step (ms)", fontsize=10)
ax.set_xlabel("KV cache tier placement split", fontsize=10)
ax.set_title(
    f"Tier Split Impact on Decode Latency — '{STRATEGY}'\n"
    "(roofline simulation, Qwen2.5-7B, avg 619 decode steps)",
    fontsize=11,
)
ax.grid(alpha=0.25, axis="y")
ax.legend(fontsize=8, loc="upper left")

fig.tight_layout()
out2 = OUTPUT_DIR / "tier_split_latency.png"
fig.savefig(out2, dpi=200)
plt.close(fig)
print(f"Saved {out2}")
