#!/usr/bin/env python3
"""Additional analysis plots:
  1. Compression ratio vs accuracy scatter
  2. Tier split impact on latency across hardware families
  3. Latency vs context length (when does KV cache compression start to matter?)

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


# ===========================================================================
# Plot 3: Latency vs context length — when does KV cache compression matter?
# ===========================================================================
# Shows all strategies on H100_SXM (all-HBM) across a log-scale context sweep.
# Top panel: absolute latency. Bottom panel: % savings vs full_cache.

SWEEP_JSONL = OUTPUT_DIR / "latency_context_sweep.jsonl"

if not SWEEP_JSONL.exists():
    print(f"Skipping plot 3: {SWEEP_JSONL} not found — run run_latency_simulation.py first")
else:
    sweep_records: list[dict] = []
    with SWEEP_JSONL.open() as f:
        for line in f:
            sweep_records.append(json.loads(line))

    PLOT_HW = "H100_SXM"

    # Strategies to highlight (one per family + the two extremes)
    HIGHLIGHT = [
        "full_cache",
        "window_1024", "window_512", "window_256", "window_128",
        "h2o_1024",    "h2o_512",    "h2o_256",    "h2o_128",
        "snapkv_512",
        "expected_attn_512",
        "full_cache_int8",
    ]
    # Thinner/lighter lines for the "interior" strategies to reduce clutter
    BOLD = {"full_cache", "window_128", "h2o_128", "snapkv_512", "expected_attn_512"}

    # Build lookup: (strategy, context_length) -> mean_latency_ms
    hw_rows = [r for r in sweep_records if r["hardware"] == PLOT_HW]
    sweep_lookup: dict[tuple, float] = {
        (r["strategy"], r["context_length"]): r["mean_latency_ms"]
        for r in hw_rows
    }
    ctx_lengths = sorted({r["context_length"] for r in hw_rows})

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for strat in HIGHLIGHT:
        xs = ctx_lengths
        ys = [sweep_lookup.get((strat, c)) for c in xs]
        if any(v is None for v in ys):
            continue

        lw = 2.2 if strat in BOLD else 1.2
        alpha = 0.95 if strat in BOLD else 0.55
        color = _color(strat)
        ls = "--" if strat == "full_cache_int8" else "-"

        ax_top.plot(xs, ys, color=color, linewidth=lw, alpha=alpha,
                    linestyle=ls, label=strat)

    ax_top.set_xscale("log")
    ax_top.set_ylabel("Mean latency per decode step (ms)", fontsize=10)
    ax_top.set_title(
        f"Decode Latency vs Context Length — {PLOT_HW}, all-HBM\n"
        "(roofline simulation, Qwen2.5-7B)",
        fontsize=11,
    )
    ax_top.grid(alpha=0.25)
    ax_top.legend(fontsize=7, loc="upper left", ncol=2)

    # Bottom panel: % savings vs full_cache
    full_cache_lat = {c: sweep_lookup.get(("full_cache", c)) for c in ctx_lengths}

    for strat in HIGHLIGHT:
        if strat == "full_cache":
            continue
        xs, ys = [], []
        for c in ctx_lengths:
            strat_lat = sweep_lookup.get((strat, c))
            base = full_cache_lat.get(c)
            if strat_lat is not None and base and base > 0:
                xs.append(c)
                ys.append(100.0 * (base - strat_lat) / base)

        lw = 2.2 if strat in BOLD else 1.2
        alpha = 0.95 if strat in BOLD else 0.55
        ax_bot.plot(xs, ys, color=_color(strat), linewidth=lw, alpha=alpha,
                    linestyle="-", label=strat)

    ax_bot.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax_bot.set_xscale("log")
    ax_bot.set_xlabel("Context length (tokens, log scale)", fontsize=10)
    ax_bot.set_ylabel("Latency savings vs full_cache (%)", fontsize=10)
    ax_bot.set_title("How much faster is each strategy vs no compression?", fontsize=10)
    ax_bot.grid(alpha=0.25)
    ax_bot.legend(fontsize=7, loc="upper left", ncol=2)

    # x-axis tick labels as round numbers
    ax_bot.set_xticks(ctx_lengths)
    ax_bot.set_xticklabels(
        [f"{c//1024}K" if c >= 1024 else str(c) for c in ctx_lengths],
        fontsize=8,
    )

    fig.tight_layout()
    out3 = OUTPUT_DIR / "latency_vs_context_length.png"
    fig.savefig(out3, dpi=200)
    plt.close(fig)
    print(f"Saved {out3}")


# ===========================================================================
# Plot 4: Pareto grid — all 16 hardware configs at 128K context
# ===========================================================================
# Each panel is one hardware config. Y-axis: real accuracy (fixed).
# X-axis: simulated latency per decode step at 128K tokens (varies by hardware).
# Panels sorted by ascending full_cache latency (fastest hardware top-left).
# Each of the 12 strategies gets its own unique color so dots are identifiable.

if not SWEEP_JSONL.exists():
    print("Skipping plot 4: latency_context_sweep.jsonl not found")
else:
    LONG_CTX = 131072

    # Unique color per strategy (tab20 gives 20 distinct colors)
    _cmap = matplotlib.colormaps["tab20"].resampled(len(STRATEGY_ORDER))
    STRAT_COLOR = {s: _cmap(i) for i, s in enumerate(STRATEGY_ORDER)}
    STRAT_MARKER = {
        "full_cache": "D", "full_cache_int8": "D",
        "window_128": "o", "window_256": "o", "window_512": "o", "window_1024": "o",
        "h2o_128": "s", "h2o_256": "s", "h2o_512": "s", "h2o_1024": "s",
        "snapkv_512": "^", "expected_attn_512": "P",
    }

    # Per-strategy accuracy from real eval (same for every hardware panel)
    strat_acc = {
        name: sum(r["correct"] for r in rows_) / len(rows_)
        for name, rows_ in by_strat.items()
    }

    # Build lookup: (hardware, strategy, context_length) -> mean_latency_ms
    sweep_lat: dict[tuple, float] = {
        (r["hardware"], r["strategy"], r["context_length"]): r["mean_latency_ms"]
        for r in sweep_records
    }

    all_hw = sorted({r["hardware"] for r in sweep_records})

    # Sort hardware panels by full_cache latency at 128K (fastest first)
    hw_order = sorted(
        all_hw,
        key=lambda hw: sweep_lat.get((hw, "full_cache", LONG_CTX), float("inf")),
    )

    NCOLS = 4
    NROWS = int(np.ceil(len(hw_order) / NCOLS))
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(NCOLS * 4, NROWS * 3.5))
    axes_flat = axes.flatten()

    for i, hw_name in enumerate(hw_order):
        ax = axes_flat[i]

        for strat in STRATEGY_ORDER:
            lat = sweep_lat.get((hw_name, strat, LONG_CTX))
            acc = strat_acc.get(strat)
            if lat is None or acc is None:
                continue
            ax.scatter(lat, acc,
                       color=STRAT_COLOR[strat],
                       marker=STRAT_MARKER.get(strat, "o"),
                       s=70, zorder=3, edgecolors="white", linewidths=0.5)

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylim(0.20, 0.85)
        ax.set_title(hw_name, fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

        if i >= len(hw_order) - NCOLS:
            ax.set_xlabel("ms/token", fontsize=7)
        if i % NCOLS == 0:
            ax.set_ylabel("Accuracy", fontsize=7)

    # Hide unused panels
    for j in range(len(hw_order), len(axes_flat)):
        axes_flat[j].set_visible(False)

    from matplotlib.lines import Line2D as _L2D
    legend_elements = [
        _L2D([0], [0],
             marker=STRAT_MARKER.get(s, "o"),
             color="w",
             markerfacecolor=STRAT_COLOR[s],
             markersize=9,
             label=s)
        for s in STRATEGY_ORDER
    ]
    fig.legend(
        handles=legend_elements,
        ncol=6,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
        title="Strategy  (marker shape = family: ◆ baseline  ● window  ■ H2O  ▲ SnapKV  ✚ ExpAttn)",
        title_fontsize=7.5,
    )

    fig.suptitle(
        f"Pareto: real accuracy vs simulated latency — all 16 hardware configs, {LONG_CTX//1024}K token context\n"
        "(panels sorted fastest → slowest; x-axis scale differs per panel)",
        fontsize=10,
    )
    fig.tight_layout()
    out4 = OUTPUT_DIR / "pareto_all_hardware.png"
    fig.savefig(out4, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out4}")
