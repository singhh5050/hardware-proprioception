#!/usr/bin/env python3
"""Simulate decode latency for accuracy-evaluated strategies across hardware.

Reads accuracy results from results/accuracy_results_final.jsonl, computes
per-strategy accuracy, then runs roofline latency simulation for each
strategy x 16 hardware configs x 5 offload splits.

Outputs:
  results/latency_simulation.csv   — full grid
  results/accuracy_vs_latency.png  — scatter with Pareto frontier
  results/latency_heatmap.png      — strategies x hardware heatmap
  results/offload_impact.png       — offload split impact on latency
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys

# Ensure src/ is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hwprop.specs import get_hardware_specs, get_model_configs
from hwprop.eval_pipeline import compute_strategy_latency

# ---------------------------------------------------------------------------
# Strategy definitions (must match accuracy_eval.py)
# ---------------------------------------------------------------------------
STRATEGIES = [
    {"strategy": "full_cache", "budget_tokens": None, "quantized": False},
    {"strategy": "full_cache_int8", "budget_tokens": None, "quantized": True},
    {"strategy": "window_128", "budget_tokens": 132, "quantized": False},
    {"strategy": "window_256", "budget_tokens": 260, "quantized": False},
    {"strategy": "window_512", "budget_tokens": 516, "quantized": False},
    {"strategy": "window_1024", "budget_tokens": 1028, "quantized": False},
    {"strategy": "h2o_128", "budget_tokens": 128, "quantized": False},
    {"strategy": "h2o_256", "budget_tokens": 256, "quantized": False},
    {"strategy": "h2o_512", "budget_tokens": 512, "quantized": False},
    {"strategy": "h2o_1024", "budget_tokens": 1024, "quantized": False},
    {"strategy": "snapkv_512", "budget_tokens": 512, "quantized": False},
    {"strategy": "expected_attn_512", "budget_tokens": 512, "quantized": False},
]

OFFLOAD_SPLITS = [
    (1.0, 0.0, 0.0),   # all HBM
    (0.7, 0.3, 0.0),   # 70% HBM, 30% CPU
    (0.5, 0.5, 0.0),   # 50/50 HBM/CPU
    (0.3, 0.3, 0.4),   # 30% HBM, 30% CPU, 40% disk
    (0.5, 0.0, 0.5),   # 50% HBM, 50% disk
]

# Hardware classification for faceted plots
HW_CLASSES = {
    "Datacenter GPU": [
        "A100_80GB", "H100_SXM", "H200", "B200", "B300",
        "L40S", "MI300X", "MI325X", "MI350X", "Gaudi_3",
    ],
    "Edge / Consumer": ["RTX_5090", "M4_Max", "Snapdragon_X_Elite"],
    "TPU": ["TPU_v5e", "TPU_v6e", "TPU_v7"],
}


def load_accuracy(jsonl_path: str) -> dict[str, float]:
    """Load per-strategy accuracy from JSONL results."""
    counts: dict[str, list[bool]] = {}
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            name = rec["strategy_name"]
            correct = rec.get("correct_rescored", rec.get("correct", False))
            counts.setdefault(name, []).append(bool(correct))
    return {name: sum(vals) / len(vals) * 100 for name, vals in counts.items()}


def get_median_params(jsonl_path: str) -> tuple[int, int]:
    """Get median prompt_tokens and tokens_generated from JSONL."""
    prompts, gens = [], []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            prompts.append(rec["prompt_tokens"])
            gens.append(rec["tokens_generated"])
    return int(statistics.median(prompts)), int(statistics.median(gens))


def run_simulation(
    jsonl_path: str, output_dir: str
) -> tuple[list[dict], dict[str, float]]:
    """Run full latency simulation grid."""
    accuracy = load_accuracy(jsonl_path)
    prompt_len, decode_steps = get_median_params(jsonl_path)
    print(f"Median prompt_tokens={prompt_len}, tokens_generated={decode_steps}")

    hw_configs = get_hardware_specs()
    model = get_model_configs()["Qwen2.5-7B"]

    results = []
    total = len(STRATEGIES) * len(hw_configs) * len(OFFLOAD_SPLITS)
    done = 0

    for strat in STRATEGIES:
        for hw_name, hw in hw_configs.items():
            for hbm_f, cpu_f, disk_f in OFFLOAD_SPLITS:
                done += 1
                # Skip disk splits on hardware without disk
                if disk_f > 0 and hw.disk_capacity == 0:
                    continue

                result = compute_strategy_latency(
                    strategy_name=strat["strategy"],
                    budget_tokens=strat["budget_tokens"],
                    hardware=hw,
                    model_config=model,
                    prompt_len=prompt_len,
                    decode_steps=decode_steps,
                    decision_interval=64,
                    offload_frac=cpu_f,
                    disk_frac=disk_f,
                    quantized=strat["quantized"],
                )
                result["offload_split"] = f"{hbm_f:.1f}/{cpu_f:.1f}/{disk_f:.1f}"
                result["accuracy"] = accuracy.get(strat["strategy"], 0.0)
                results.append(result)

                if done % 100 == 0:
                    print(f"  {done}/{total} simulations...")

    print(f"Completed {len(results)} simulations (skipped hw without disk)")

    # Write CSV
    csv_path = os.path.join(output_dir, "latency_simulation.csv")
    fieldnames = [
        "strategy", "hardware", "offload_split", "offload_frac", "disk_frac",
        "mean_latency_ms", "total_time_s", "prefill_time_s", "accuracy",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"Wrote {csv_path}")

    return results, accuracy


def _hw_class(hw_name: str) -> str:
    for cls, members in HW_CLASSES.items():
        if hw_name in members:
            return cls
    return "Other"


def plot_accuracy_vs_latency(results: list[dict], accuracy: dict[str, float], output_dir: str) -> None:
    """Scatter: accuracy (y) vs ms/token (x), faceted by hardware class, HBM-only."""
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Filter to HBM-only split
    hbm_only = [r for r in results if r["offload_split"] == "1.0/0.0/0.0"]

    classes = list(HW_CLASSES.keys())
    fig, axes = plt.subplots(1, len(classes), figsize=(6 * len(classes), 5.5), sharey=True)
    if len(classes) == 1:
        axes = [axes]

    colors = plt.cm.tab20(np.linspace(0, 1, len(STRATEGIES)))
    strat_colors = {s["strategy"]: colors[i] for i, s in enumerate(STRATEGIES)}

    for ax, cls in zip(axes, classes):
        cls_hw = set(HW_CLASSES[cls])
        cls_results = [r for r in hbm_only if r["hardware"] in cls_hw]

        # Group by strategy: average latency across hardware in this class
        strat_latency: dict[str, list[float]] = {}
        for r in cls_results:
            strat_latency.setdefault(r["strategy"], []).append(r["mean_latency_ms"])

        xs, ys, labels, cs = [], [], [], []
        for sname, lats in strat_latency.items():
            mean_lat = statistics.mean(lats)
            acc = accuracy.get(sname, 0.0)
            xs.append(mean_lat)
            ys.append(acc)
            labels.append(sname)
            cs.append(strat_colors[sname])

        ax.scatter(xs, ys, c=cs, s=80, zorder=3, edgecolors="black", linewidth=0.5)
        for x, y, label in zip(xs, ys, labels):
            ax.annotate(label, (x, y), fontsize=6, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points")

        # Pareto frontier (higher accuracy, lower latency = better)
        points = sorted(zip(xs, ys), key=lambda p: p[0])
        pareto_x, pareto_y = [], []
        best_acc = -1
        for px, py in points:
            if py > best_acc:
                pareto_x.append(px)
                pareto_y.append(py)
                best_acc = py
        if len(pareto_x) > 1:
            ax.plot(pareto_x, pareto_y, "r--", alpha=0.6, linewidth=1.5, label="Pareto frontier")
            ax.legend(fontsize=7)

        ax.set_title(cls, fontsize=11)
        ax.set_xlabel("Mean decode latency (ms/token)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle("Accuracy vs Decode Latency (HBM-only, Qwen2.5-7B on MATH-500)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(output_dir, "accuracy_vs_latency.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_latency_heatmap(results: list[dict], output_dir: str) -> None:
    """Heatmap: strategies (rows) x hardware (cols), HBM-only, cell color = ms/token."""
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    hbm_only = [r for r in results if r["offload_split"] == "1.0/0.0/0.0"]

    strat_names = [s["strategy"] for s in STRATEGIES]
    hw_names = list(get_hardware_specs().keys())

    # Build matrix
    lookup = {}
    for r in hbm_only:
        lookup[(r["strategy"], r["hardware"])] = r["mean_latency_ms"]

    matrix = np.zeros((len(strat_names), len(hw_names)))
    for i, sn in enumerate(strat_names):
        for j, hn in enumerate(hw_names):
            matrix[i, j] = lookup.get((sn, hn), 0)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(hw_names)))
    ax.set_xticklabels(hw_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(strat_names)))
    ax.set_yticklabels(strat_names, fontsize=8)

    # Annotate cells
    for i in range(len(strat_names)):
        for j in range(len(hw_names)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("ms/token")
    ax.set_title("Decode Latency Heatmap (HBM-only, Qwen2.5-7B)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(output_dir, "latency_heatmap.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_offload_impact(results: list[dict], output_dir: str) -> None:
    """For H100, A100, M4 Max: how offload splits affect latency."""
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    selected_hw = ["H100_SXM", "A100_80GB", "M4_Max"]
    highlight_strats = ["full_cache", "full_cache_int8", "window_512", "h2o_512", "snapkv_512"]

    fig, axes = plt.subplots(1, len(selected_hw), figsize=(6 * len(selected_hw), 5), sharey=True)
    if len(selected_hw) == 1:
        axes = [axes]

    split_labels = [f"{int(h*100)}/{int(c*100)}/{int(d*100)}" for h, c, d in OFFLOAD_SPLITS]
    colors = plt.cm.Set2(np.linspace(0, 1, len(highlight_strats)))

    for ax, hw_name in zip(axes, selected_hw):
        hw_results = [r for r in results if r["hardware"] == hw_name]

        for idx, sname in enumerate(highlight_strats):
            strat_results = [r for r in hw_results if r["strategy"] == sname]
            # Order by offload split
            split_to_lat = {r["offload_split"]: r["mean_latency_ms"] for r in strat_results}
            x_pos, lats = [], []
            for si, (h, c, d) in enumerate(OFFLOAD_SPLITS):
                key = f"{h:.1f}/{c:.1f}/{d:.1f}"
                if key in split_to_lat:
                    x_pos.append(si)
                    lats.append(split_to_lat[key])

            ax.plot(x_pos, lats, marker="o", color=colors[idx], linewidth=1.5,
                    markersize=5, label=sname)

        ax.set_xticks(range(len(split_labels)))
        ax.set_xticklabels(split_labels, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("Offload split (HBM/CPU/Disk %)")
        ax.set_title(hw_name, fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean decode latency (ms/token)")
    axes[-1].legend(fontsize=7, loc="upper left")
    fig.suptitle("Impact of Memory Offloading on Decode Latency (Qwen2.5-7B)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(output_dir, "offload_impact.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(repo_root, "results")
    jsonl_path = os.path.join(results_dir, "accuracy_results_final.jsonl")

    if not os.path.exists(jsonl_path):
        print(f"ERROR: {jsonl_path} not found")
        sys.exit(1)

    results, accuracy = run_simulation(jsonl_path, results_dir)
    plot_accuracy_vs_latency(results, accuracy, results_dir)
    plot_latency_heatmap(results, results_dir)
    plot_offload_impact(results, results_dir)
    print("\nDone. Outputs in results/")


if __name__ == "__main__":
    main()
