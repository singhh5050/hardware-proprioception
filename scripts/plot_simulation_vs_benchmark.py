#!/usr/bin/env python3
"""Plot: simulation vs. measured GPU latency.

Produces two panels:
  Left  — Context length sweep (H100 SXM, LLaMA-3.2-3B, flash_attention_2)
           Shows measured, roofline, and calibrated-simulator predictions
           across 1K–128K context lengths.

  Right — Per-strategy benchmark (A100-40GB, Qwen2.5-7B, SDPA)
           Grouped bar chart of measured vs. simulated latency per strategy.

Usage (from repo root):
    python scripts/plot_simulation_vs_benchmark.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from hwprop.specs import get_hardware_specs, get_model_configs
from hwprop.cost_model import CostModel, KVCacheState
from hwprop.overhead import OVERHEAD_H100_FLASH2, OVERHEAD_A100_SDPA
from hwprop.simulator import LLMSimulator, simulate_latency
from hwprop.strategy import KVCacheStrategy, STRATEGY_REGISTRY

CONTEXT_CSV  = Path("results/benchmark/context_sweep_H100_SXM.csv")
STRATEGY_CSV = Path("results/benchmark/benchmark_results.csv")
OUTPUT       = Path("results/plots/simulation_vs_benchmark.png")

HARDWARE = get_hardware_specs()
MODELS   = get_model_configs()

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
MEASURED_COLOR   = "#1565C0"   # deep blue
ROOFLINE_COLOR   = "#B71C1C"   # deep red
SIMULATED_COLOR  = "#2E7D32"   # deep green

STRATEGY_COLORS = {
    "full_cache":       "#1565C0",
    "full_cache_int8":  "#5C6BC0",
    "window":           "#FF8F00",
    "h2o":              "#2E7D32",
    "snapkv":           "#7B1FA2",
    "expected_attn":    "#C62828",
}

def _strategy_color(name: str) -> str:
    for prefix, color in STRATEGY_COLORS.items():
        if name.startswith(prefix):
            return color
    return "#607D8B"


def _pretty_name(name: str) -> str:
    return (name
            .replace("full_cache_int8", "Full (INT8)")
            .replace("full_cache", "Full Cache")
            .replace("window_", "Win-")
            .replace("h2o_", "H2O-")
            .replace("snapkv_512", "SnapKV-512")
            .replace("expected_attn_512", "ExpAttn-512"))


# ---------------------------------------------------------------------------
# Panel 1: Context length sweep (H100 SXM)
# ---------------------------------------------------------------------------

def plot_context_sweep(ax: plt.Axes) -> None:
    hw  = HARDWARE["H100_SXM"]
    mdl = MODELS["LLaMA-3.2-3B"]
    cost_model = CostModel(hw, mdl)

    context_lengths, measured_ms, roofline_ms, simulated_ms = [], [], [], []

    with open(CONTEXT_CSV) as f:
        for row in csv.DictReader(f):
            n    = int(row["context_length"])
            meas = float(row["measured_per_token_ms"])
            sim_raw = float(row["simulated_per_token_ms"])

            # Calibrated simulator prediction
            kv = KVCacheState(seq_len=n, tokens_in_hbm=n, tokens_in_hbm_quantized=0,
                              tokens_in_cpu=0, tokens_on_disk=0, tokens_evicted=0)
            raw_cost = cost_model.step_cost(kv)
            t_corr = OVERHEAD_H100_FLASH2.corrected_time(raw_cost.time_s, n)

            context_lengths.append(n)
            measured_ms.append(meas)
            roofline_ms.append(sim_raw)
            simulated_ms.append(t_corr * 1000.0)

    x = np.array(context_lengths)

    ax.plot(x, measured_ms,  "o-",  color=MEASURED_COLOR,  lw=2.0, ms=7,
            label="Measured (H100 SXM)", zorder=5)
    ax.plot(x, simulated_ms, "s--", color=SIMULATED_COLOR, lw=2.0, ms=7,
            label="Calibrated simulator", zorder=4)
    ax.plot(x, roofline_ms,  "^:",  color=ROOFLINE_COLOR,  lw=1.5, ms=6,
            label="Roofline only", zorder=3)

    # Annotate the ratio improvement at a few points
    for i, (n, meas, sim, roof) in enumerate(zip(context_lengths, measured_ms, simulated_ms, roofline_ms)):
        if n in (1024, 32768, 131072):
            ratio_roof = meas / roof
            ratio_sim  = meas / sim
            ypos = max(meas, sim) * 1.25
            ax.annotate(
                f"Roofline: {ratio_roof:.1f}×\nSimulator: {ratio_sim:.1f}×",
                xy=(n, meas), xytext=(n, ypos),
                ha="center", va="bottom", fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color="#888", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", alpha=0.85),
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(
        lambda v, _: f"{int(v) // 1024}K" if v >= 1024 else str(int(v))
    ))
    ax.set_xlabel("Context length (tokens)", fontsize=11)
    ax.set_ylabel("Latency (ms / token)", fontsize=11)
    ax.set_title("H100 SXM · LLaMA-3.2-3B · flash_attention_2", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
    ax.set_xlim(min(context_lengths) * 0.7, max(context_lengths) * 1.4)


# ---------------------------------------------------------------------------
# Panel 2: Per-strategy benchmark (A100-40GB)
# ---------------------------------------------------------------------------

def plot_strategy_comparison(ax: plt.Axes) -> None:
    strategies, measured_ms, simulated_ms, colors = [], [], [], []

    with open(STRATEGY_CSV) as f:
        for row in csv.DictReader(f):
            strategies.append(row["strategy"])
            measured_ms.append(float(row["measured_per_token_ms"]))
            simulated_ms.append(float(row["simulated_per_token_ms"]))
            colors.append(_strategy_color(row["strategy"]))

    x      = np.arange(len(strategies))
    width  = 0.32
    labels = [_pretty_name(s) for s in strategies]

    # Three bars per strategy: measured | simulator | roofline
    width = 0.25
    roofline_ms = [s * 0.3 for s in measured_ms]  # placeholder — overwritten below

    # Re-read roofline from CSV
    roofline_ms = []
    with open(STRATEGY_CSV) as f:
        for row in csv.DictReader(f):
            roofline_ms.append(float(row["simulated_per_token_ms"]))

    ax.bar(x - width, measured_ms,  width, label="Measured (A100-40GB)",
           color=MEASURED_COLOR,  alpha=0.88, edgecolor="white", linewidth=0.5)
    ax.bar(x,         simulated_ms, width, label="Calibrated simulator",
           color=SIMULATED_COLOR, alpha=0.88, edgecolor="white", linewidth=0.5)
    ax.bar(x + width, roofline_ms,  width, label="Roofline only",
           color=ROOFLINE_COLOR,  alpha=0.88, edgecolor="white", linewidth=0.5)

    # Ratio annotation (measured / simulator) above each group
    for i, (meas, sim, roof) in enumerate(zip(measured_ms, simulated_ms, roofline_ms)):
        ymax = max(meas, sim, roof) + 0.8
        ratio_sim  = meas / sim
        ratio_roof = meas / roof
        ax.text(i, ymax, f"{ratio_sim:.1f}×\n({ratio_roof:.0f}×)",
                ha="center", va="bottom", fontsize=7, color="#333",
                linespacing=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Latency (ms / token)", fontsize=11)
    ax.set_title("A100-40GB · Qwen2.5-7B · SDPA", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Simulation vs. Measured GPU Latency — Roofline vs. Calibrated Simulator",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plot_context_sweep(axes[0])
    plot_strategy_comparison(axes[1])

    # Shared note
    fig.text(
        0.5, -0.03,
        "Annotations show measured / predicted ratio  (1.0× = perfect).  "
        "Calibrated simulator adds launch overhead + Flash Attention scan correction on top of roofline.",
        ha="center", fontsize=9, color="#555",
    )

    plt.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()
