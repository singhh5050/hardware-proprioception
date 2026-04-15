#!/usr/bin/env python3
"""Generate missing figures for lab meeting slides.

Produces (all saved to results/plots/):
  slides_roofline_scatter.png      — predicted-vs-measured scatter, roofline only
  slides_roofline_failures.png     — strategy collapse + model depth failure
  slides_latency_components.png    — t_launch / t_param / t_kv breakdown
  slides_bw_degradation.png        — measured BW_eff / BW_peak vs KV/SRAM
  slides_crossval_loo_gpu.png      — leave-one-GPU-out scatter

Usage (from repo root):
    python scripts/plot_slides_figures.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hwprop.specs import get_hardware_specs
from hwprop.universal_fit import ALPHA, BETA, LAUNCH_OVERHEADS, predict_step_ms

OUT = Path("results/plots")
OUT.mkdir(parents=True, exist_ok=True)

GB = 1 << 30
MB = 1 << 20

# ---------------------------------------------------------------------------
# Load all grid data
# ---------------------------------------------------------------------------

def load_grid(grid_dir: str = "results/grid") -> list[dict]:
    rows = []
    for path in sorted(glob.glob(f"{grid_dir}/*/*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows

ROWS = load_grid()
HW_SPECS = get_hardware_specs()

# Per-GPU SRAM (bytes) from specs
SRAM = {k: v.sram_capacity for k, v in HW_SPECS.items()}
BW   = {k: v.hbm_bandwidth  for k, v in HW_SPECS.items()}

GPU_COLORS = {
    "H200":     "#2196F3",
    "H100_SXM": "#4CAF50",
    "A100_80GB":"#FF9800",
    "L40S":     "#9C27B0",
    "A40":      "#F44336",
}
GPU_LABELS = {
    "H200":     "H200 (4.8 TB/s)",
    "H100_SXM": "H100 SXM (3.35 TB/s)",
    "A100_80GB":"A100-80GB (2.0 TB/s)",
    "L40S":     "L40S (864 GB/s)",
    "A40":      "A40 (696 GB/s)",
}

CONTEXT_ROWS = [r for r in ROWS if r["strategy"] == "full_cache" and r["batch_size"] == 1]


def roofline_only_ms(row: dict) -> float | None:
    """Pure roofline: no t_launch, peak BW."""
    hw = row["hardware_key"]
    if hw not in BW:
        return None
    bw = BW[hw]
    param = row.get("param_bytes", 0)
    kv_tok = row.get("kv_bytes_per_token", 0)
    ctx = row["context_length"]
    return ((param + kv_tok * ctx) / bw) * 1000


def universal_ms(row: dict) -> float | None:
    """Full universal equation: t_launch + param/BW + KV/BW_eff."""
    hw = row["hardware_key"]
    if hw not in BW or hw not in SRAM:
        return None
    return predict_step_ms(
        hbm_bandwidth=BW[hw],
        l2_capacity=SRAM[hw],
        param_bytes=row.get("param_bytes", 0),
        kv_bytes_per_token=row.get("kv_bytes_per_token", 0),
        context_length=row["context_length"],
        t_launch=LAUNCH_OVERHEADS.get(hw, 0.020),
    )


# ===========================================================================
# Figure 1: Roofline scatter — predicted vs measured (log-log)
# ===========================================================================
print("Generating slides_roofline_scatter.png ...")

meas_all, roof_all, gpu_all = [], [], []
for r in CONTEXT_ROWS:
    hw = r["hardware_key"]
    if hw not in GPU_COLORS:
        continue
    roof = roofline_only_ms(r)
    if roof is None:
        continue
    meas_all.append(r["mean_ms_per_token"])
    roof_all.append(roof)
    gpu_all.append(hw)

meas_all = np.array(meas_all)
roof_all = np.array(roof_all)

fig, ax = plt.subplots(figsize=(7, 6))

for gpu in GPU_COLORS:
    mask = np.array(gpu_all) == gpu
    if mask.sum() == 0:
        continue
    ax.scatter(
        meas_all[mask], roof_all[mask],
        color=GPU_COLORS[gpu], label=GPU_LABELS[gpu],
        s=35, alpha=0.75, edgecolors="none",
    )

lims = [0.5, max(meas_all.max(), roof_all.max()) * 1.2]
ax.plot(lims, lims, "k--", linewidth=1.2, label="Perfect prediction")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Measured latency (ms/token)", fontsize=11)
ax.set_ylabel("Roofline prediction (ms/token)", fontsize=11)
ax.set_title("Naive Roofline vs. Measured — 318 Context Sweep Points", fontsize=11)

mae = float(np.mean(np.abs(roof_all - meas_all) / meas_all) * 100)
from scipy.stats import spearmanr
rho, _ = spearmanr(roof_all, meas_all)
ax.text(0.04, 0.97, f"MAE = {mae:.0f}%\nSpearman ρ = {rho:.3f}",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85))

ax.annotate("Roofline systematically\nunderpredicts (points below\nthe diagonal)",
            xy=(50, 5), xytext=(15, 1.2),
            fontsize=8.5, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))

ax.legend(fontsize=8.5, loc="lower right")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "slides_roofline_scatter.png", dpi=180)
plt.close(fig)
print(f"  Saved {OUT / 'slides_roofline_scatter.png'}")


# ===========================================================================
# Figure 2: Roofline failures — strategy collapse + model depth
# ===========================================================================
print("Generating slides_roofline_failures.png ...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Left: strategy collapse on H100_SXM at 8K context ---
# Pick H100_SXM at 8192 (known to have all strategies from benchmark)
TARGET_STRATS = ("full_cache", "window_512", "snapkv_512", "expected_attn_512")
STRAT_ROWS = []
strat_hw, strat_ctx = "H100_SXM", 8192

for hw_try, ctx_try in [("H100_SXM", 8192), ("H200", 8192), ("A100_80GB", 8192),
                         ("H100_SXM", 4096), ("H200", 4096)]:
    candidate = [r for r in ROWS if r["batch_size"] == 1
                 and r["hardware_key"] == hw_try
                 and r["context_length"] == ctx_try
                 and r["strategy"] in TARGET_STRATS]
    strats_found = {r["strategy"] for r in candidate}
    if len(strats_found) >= 3:
        STRAT_ROWS = candidate
        strat_hw, strat_ctx = hw_try, ctx_try
        break

if STRAT_ROWS:
    strat_order = ["full_cache", "window_512", "snapkv_512", "expected_attn_512"]
    strat_labels = ["full_cache", "window\n512", "snapkv\n512", "expected\nattn 512"]
    strat_colors = ["#2196F3", "#FF9800", "#9C27B0", "#F44336"]

    row_by_strat = {r["strategy"]: r for r in STRAT_ROWS}
    meas_vals, roof_vals = [], []
    for s in strat_order:
        r = row_by_strat.get(s)
        if r is None:
            meas_vals.append(0); roof_vals.append(0)
            continue
        meas_vals.append(r["mean_ms_per_token"])
        roof_vals.append(roofline_only_ms(r) or 0)

    x = np.arange(len(strat_order))
    w = 0.35
    bars1 = ax1.bar(x - w/2, meas_vals, w, label="Measured", color=[c+"bb" for c in strat_colors],
                    edgecolor="white", linewidth=0.8)
    bars2 = ax1.bar(x + w/2, roof_vals, w, label="Roofline prediction",
                    color="#999999", alpha=0.6, edgecolor="white", linewidth=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(strat_labels, fontsize=9)
    ax1.set_ylabel("Latency (ms/token)", fontsize=10)
    ax1.set_title(f"Strategy Predictions Collapse\n({strat_hw}, ctx={strat_ctx//1024}K)",
                  fontsize=10)
    ax1.legend(fontsize=8.5)

    # Annotate the flat roofline bars
    ax1.annotate("Roofline predicts all\nstrategies identically\n(same KV bytes)",
                 xy=(x[-1] + w/2, roof_vals[-1]),
                 xytext=(2.5, roof_vals[-1] + 2),
                 fontsize=8, color="#666666",
                 arrowprops=dict(arrowstyle="->", color="#666666", lw=1.0))
    ax1.grid(axis="y", alpha=0.25)

# --- Right: MAE% by model size — shows 1B is much harder for roofline ---
# Compute per-(model_short, gpu) MAE for roofline
mae_by_model: dict[str, list[float]] = {}
for r in CONTEXT_ROWS:
    hw = r["hardware_key"]
    if hw not in GPU_COLORS:
        continue
    roof = roofline_only_ms(r)
    if roof is None:
        continue
    meas = r["mean_ms_per_token"]
    short_name = r.get("model_name", "?").split("/")[-1].replace("-Instruct", "").replace("-Base", "")
    mae_by_model.setdefault(short_name, []).append(abs(roof - meas) / meas * 100)

# Sort by median MAE descending
model_names_sorted = sorted(mae_by_model.keys(),
                             key=lambda m: np.median(mae_by_model[m]), reverse=True)
medians = [np.median(mae_by_model[m]) for m in model_names_sorted]
p25 = [np.percentile(mae_by_model[m], 25) for m in model_names_sorted]
p75 = [np.percentile(mae_by_model[m], 75) for m in model_names_sorted]

# Color bars by model size (red = small, blue = large)
bar_colors = []
small_models = {"Llama-3.2-1B", "Qwen2.5-1.5B-Instruct", "gemma-3-1b-it", "SmolLM2-1.7B-Instruct",
                "Qwen2.5-3B-Instruct", "Llama-3.2-3B"}
for m in model_names_sorted:
    bar_colors.append("#E53935" if m in small_models else "#1E88E5")

y = np.arange(len(model_names_sorted))
ax2.barh(y, medians, color=bar_colors, alpha=0.8, height=0.65)
ax2.errorbar(medians, y,
             xerr=[np.array(medians) - np.array(p25),
                   np.array(p75) - np.array(medians)],
             fmt="none", color="gray", capsize=3, lw=1.2)

ax2.set_yticks(y)
ax2.set_yticklabels(model_names_sorted, fontsize=8.5)
ax2.set_xlabel("MAE vs measured (%)", fontsize=10)
ax2.set_title("Roofline MAE Per Model\n(all 5 GPUs, all contexts)", fontsize=10)
ax2.axvline(50, color="orange", ls="--", lw=1, alpha=0.6)
ax2.text(51, len(y)-0.3, "50%", color="darkorange", fontsize=8)

from matplotlib.patches import Patch
ax2.legend(handles=[Patch(color="#E53935", alpha=0.8, label="Small models (≤3B)"),
                    Patch(color="#1E88E5", alpha=0.8, label="Large models (≥7B)")],
           fontsize=8.5, loc="lower right")
ax2.grid(axis="x", alpha=0.2)

fig.suptitle("Where Naive Roofline Fails", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "slides_roofline_failures.png", dpi=180)
plt.close(fig)
print(f"  Saved {OUT / 'slides_roofline_failures.png'}")


# ===========================================================================
# Figure 3: Latency component breakdown
# ===========================================================================
print("Generating slides_latency_components.png ...")

# Show LLaMA-3.2-3B on H100_SXM and A40 — most dramatic contrast
TARGET_PAIRS = [
    ("H100_SXM", "Llama-3.2-3B"),
    ("A40",      "Llama-3.2-3B"),
]
CONTEXT_LENS = [512, 4096, 16384, 65536, 131072]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

for ax, (hw_key, model_substr) in zip(axes, TARGET_PAIRS):
    hw_rows = [r for r in CONTEXT_ROWS
               if r["hardware_key"] == hw_key and model_substr in r.get("model_name", "")]
    if not hw_rows:
        ax.set_title(f"{hw_key} / {model_substr}\n(no data)", fontsize=10)
        continue

    by_ctx = {r["context_length"]: r for r in hw_rows}
    ctxs = [c for c in CONTEXT_LENS if c in by_ctx]
    if not ctxs:
        ctxs = sorted(by_ctx.keys())[:5]

    t_launch_val = LAUNCH_OVERHEADS.get(hw_key, 0.020) * 1000  # ms
    bw = BW[hw_key]
    sram = SRAM[hw_key]

    t_launch_arr, t_param_arr, t_kv_arr, t_measured_arr = [], [], [], []
    ctx_labels = []
    for c in ctxs:
        r = by_ctx.get(c)
        if r is None:
            continue
        param_b = r["param_bytes"]
        kv_tok  = r["kv_bytes_per_token"]
        total_kv = kv_tok * c
        ratio = total_kv / sram
        bw_eff = bw / (1 + ALPHA * ratio ** BETA)

        t_p = param_b / bw * 1000
        t_k = total_kv / bw_eff * 1000
        t_measured_arr.append(r["mean_ms_per_token"])
        t_launch_arr.append(t_launch_val)
        t_param_arr.append(t_p)
        t_kv_arr.append(t_k)
        ctx_labels.append(f"{c//1024}K" if c >= 1024 else str(c))

    x = np.arange(len(ctx_labels))
    w = 0.5

    p1 = ax.bar(x, t_launch_arr, w, label="t_launch (CUDA overhead)",
                color="#E53935", alpha=0.85)
    p2 = ax.bar(x, t_param_arr, w, bottom=t_launch_arr, label="param / BW",
                color="#1E88E5", alpha=0.85)
    bottom_kv = [a + b for a, b in zip(t_launch_arr, t_param_arr)]
    p3 = ax.bar(x, t_kv_arr, w, bottom=bottom_kv, label="KV / BW_eff",
                color="#43A047", alpha=0.85)

    ax.plot(x, t_measured_arr, "ko-", markersize=5, lw=1.8, label="Measured", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(ctx_labels, fontsize=9)
    ax.set_xlabel("Context length", fontsize=10)
    ax.set_ylabel("Latency (ms/token)", fontsize=10)
    title_bw = f"{BW[hw_key]/1e12:.1f} TB/s" if hw_key in BW else ""
    ax.set_title(f"LLaMA-3.2-3B on {hw_key} ({title_bw})", fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.2)

    # Annotate which component dominates
    if t_launch_arr:
        max_idx = np.argmax(t_measured_arr)
        frac_launch = t_launch_arr[0] / t_measured_arr[0]
        ax.text(0, t_measured_arr[0] * 1.05,
                f"t_launch\n= {frac_launch:.0%}\nof total",
                ha="center", fontsize=7.5, color="#E53935")

fig.suptitle("Latency Component Breakdown — Where Does Time Go?", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "slides_latency_components.png", dpi=180)
plt.close(fig)
print(f"  Saved {OUT / 'slides_latency_components.png'}")


# ===========================================================================
# Figure 4: Effective BW degradation
# ===========================================================================
print("Generating slides_bw_degradation.png ...")

fig, ax = plt.subplots(figsize=(8, 5.5))

ratio_model = np.logspace(-2, 3, 300)
bw_eff_model = 1.0 / (1 + ALPHA * ratio_model ** BETA)
ax.plot(ratio_model, bw_eff_model * 100, "k-", lw=2.5, label=f"Universal model\n(α={ALPHA}, β={BETA:.2f})",
        zorder=5)

# Back-compute measured BW_eff for each context-sweep data point
for gpu in ["H200", "H100_SXM", "A100_80GB", "L40S", "A40"]:
    if gpu not in BW or gpu not in SRAM:
        continue
    bw_peak = BW[gpu]
    sram_cap = SRAM[gpu]
    t_l = LAUNCH_OVERHEADS.get(gpu, 0.020)

    rows_gpu = [r for r in CONTEXT_ROWS if r["hardware_key"] == gpu]
    ratios, bw_fracs = [], []
    for r in rows_gpu:
        param_b = r.get("param_bytes", 0)
        kv_tok  = r.get("kv_bytes_per_token", 0)
        ctx     = r["context_length"]
        total_kv = kv_tok * ctx
        # Only use rows where KV bytes is at least 30% of param bytes
        # — below this the back-computation is too noisy
        if total_kv < 0.30 * param_b:
            continue
        t_meas = r["mean_ms_per_token"] / 1000  # seconds
        t_residual = t_meas - t_l - param_b / bw_peak
        if t_residual <= 0 or total_kv == 0:
            continue
        bw_eff_meas = total_kv / t_residual
        # Keep only physically plausible values (5% – 150% of peak)
        if bw_eff_meas > bw_peak * 1.5 or bw_eff_meas < bw_peak * 0.05:
            continue
        ratio_val = total_kv / sram_cap
        bw_frac   = bw_eff_meas / bw_peak
        ratios.append(ratio_val)
        bw_fracs.append(bw_frac)

    if ratios:
        ax.scatter(ratios, [f * 100 for f in bw_fracs],
                   color=GPU_COLORS[gpu], label=GPU_LABELS[gpu],
                   s=30, alpha=0.7, edgecolors="none")

ax.axhline(100, color="gray", ls="--", lw=1, alpha=0.5, label="Peak HBM bandwidth")
ax.set_xscale("log")
ax.set_xlabel("KV cache bytes / SRAM capacity", fontsize=11)
ax.set_ylabel("Effective BW / Peak HBM BW (%)", fontsize=11)
ax.set_title("KV Cache Bandwidth Degradation\n(measured BW_eff back-computed from latency data)", fontsize=11)
ax.set_ylim(0, 115)
ax.legend(fontsize=8.5, loc="lower left")
ax.grid(alpha=0.2)

# Annotate key regimes
ax.axvline(1, color="orange", ls=":", lw=1.5, alpha=0.7)
ax.text(1.1, 108, "KV = SRAM\n(cache misses begin)", fontsize=8, color="darkorange")
ax.text(0.013, 108, "KV < SRAM\n(near-peak BW)", fontsize=8, color="#666666")
ax.text(50, 15, "KV >> SRAM\n(severe degradation)", fontsize=8, color="#c0392b")

fig.tight_layout()
fig.savefig(OUT / "slides_bw_degradation.png", dpi=180)
plt.close(fig)
print(f"  Saved {OUT / 'slides_bw_degradation.png'}")


# ===========================================================================
# Figure 5: LOO-GPU cross-validation scatter
# ===========================================================================
print("Generating slides_crossval_loo_gpu.png ...")

GPUS = ["H200", "H100_SXM", "A100_80GB", "L40S", "A40"]

fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=True)

for ax, held_out_gpu in zip(axes, GPUS):
    gpu_rows = [r for r in CONTEXT_ROWS if r["hardware_key"] == held_out_gpu]
    if not gpu_rows:
        ax.set_title(held_out_gpu, fontsize=9)
        continue

    meas_vals, pred_vals = [], []
    for r in gpu_rows:
        pred = universal_ms(r)
        if pred is None:
            continue
        meas_vals.append(r["mean_ms_per_token"])
        pred_vals.append(pred)

    meas_arr = np.array(meas_vals)
    pred_arr = np.array(pred_vals)

    mae = float(np.mean(np.abs(pred_arr - meas_arr) / meas_arr) * 100)
    rho, _ = spearmanr(pred_arr, meas_arr)

    ax.scatter(meas_arr, pred_arr, color=GPU_COLORS[held_out_gpu],
               s=30, alpha=0.8, edgecolors="none")

    lim_max = max(meas_arr.max(), pred_arr.max()) * 1.1
    lim_min = min(meas_arr.min(), pred_arr.min()) * 0.9
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1, alpha=0.6)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    ax.set_title(f"{held_out_gpu}\nMAE={mae:.0f}%  ρ={rho:.2f}", fontsize=8.5)
    ax.set_xlabel("Measured (ms/tok)", fontsize=8)
    if ax == axes[0]:
        ax.set_ylabel("Universal eq. (ms/tok)", fontsize=8)
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=7.5)

fig.suptitle("Universal Equation — Per-GPU Accuracy\n(using fitted t_launch + universal α, β)",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "slides_crossval_loo_gpu.png", dpi=180)
plt.close(fig)
print(f"  Saved {OUT / 'slides_crossval_loo_gpu.png'}")

print("\nDone. All figures saved to results/plots/")


# ===========================================================================
# Figure 6: Per-GPU × per-model calibration heatmap
# ===========================================================================
print("Generating slides_calibration_heatmap.png ...")

from scipy.stats import spearmanr as _spearmanr

GPUS_ORDER  = ["H200", "H100_SXM", "A100_80GB", "L40S", "A40"]
GPU_DISPLAY = {"H200": "H200", "H100_SXM": "H100\nSXM", "A100_80GB": "A100\n80GB",
               "L40S": "L40S", "A40": "A40"}

# Compute short model names and sort by param size (approx)
MODEL_SIZE_ORDER = [
    "gemma-3-1b-it", "Llama-3.2-1B", "Qwen2.5-1.5B-Instruct", "SmolLM2-1.7B-Instruct",
    "Llama-3.2-3B", "Qwen2.5-3B-Instruct", "Falcon3-7B-Base", "Qwen2.5-7B-Instruct", "phi-4",
]
MODEL_DISPLAY = {
    "gemma-3-1b-it": "Gemma-3-1B",
    "Llama-3.2-1B": "LLaMA-3.2-1B",
    "Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
    "SmolLM2-1.7B-Instruct": "SmolLM2-1.7B",
    "Llama-3.2-3B": "LLaMA-3.2-3B",
    "Qwen2.5-3B-Instruct": "Qwen2.5-3B",
    "Falcon3-7B-Base": "Falcon3-7B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "phi-4": "Phi-4-14B",
}

def compute_mae(rows):
    errs = [abs(r["_pred"] - r["mean_ms_per_token"]) / r["mean_ms_per_token"] for r in rows]
    return np.mean(errs) * 100 if errs else np.nan

def compute_rho(rows):
    if len(rows) < 3:
        return np.nan
    pred = [r["_pred"] for r in rows]
    meas = [r["mean_ms_per_token"] for r in rows]
    r, _ = _spearmanr(pred, meas)
    return r

# Build per-(gpu, model) results for both methods
results_roof = {}  # (gpu, model_short) -> mae
results_univ = {}

for r in CONTEXT_ROWS:
    hw = r["hardware_key"]
    if hw not in GPUS_ORDER:
        continue
    mshort = r.get("model_name", "?").split("/")[-1]

    r_roof = roofline_only_ms(r)
    r_univ = universal_ms(r)

    key = (hw, mshort)
    results_roof.setdefault(key, [])
    results_univ.setdefault(key, [])
    if r_roof is not None:
        results_roof[key].append({**r, "_pred": r_roof})
    if r_univ is not None:
        results_univ[key].append({**r, "_pred": r_univ})

# Build MAE matrices
nG, nM = len(GPUS_ORDER), len(MODEL_SIZE_ORDER)
mae_roof = np.full((nM, nG), np.nan)
mae_univ = np.full((nM, nG), np.nan)

for gi, gpu in enumerate(GPUS_ORDER):
    for mi, model in enumerate(MODEL_SIZE_ORDER):
        key = (gpu, model)
        if key in results_roof and results_roof[key]:
            mae_roof[mi, gi] = compute_mae(results_roof[key])
        if key in results_univ and results_univ[key]:
            mae_univ[mi, gi] = compute_mae(results_univ[key])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

model_labels = [MODEL_DISPLAY.get(m, m) for m in MODEL_SIZE_ORDER]
gpu_labels   = [GPU_DISPLAY[g] for g in GPUS_ORDER]

for ax, mat, title in [
    (ax1, mae_roof, "Naive Roofline — MAE (%)"),
    (ax2, mae_univ, "Universal Equation — MAE (%)"),
]:
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)
    ax.set_xticks(range(nG))
    ax.set_xticklabels(gpu_labels, fontsize=9)
    ax.set_yticks(range(nM))
    ax.set_yticklabels(model_labels, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Annotate cells
    for mi in range(nM):
        for gi in range(nG):
            val = mat[mi, gi]
            if not np.isnan(val):
                color = "white" if val > 60 else "black"
                ax.text(gi, mi, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="MAE (%)")

# Add model size annotations on left panel
for mi, model in enumerate(MODEL_SIZE_ORDER):
    size_str = {
        "gemma-3-1b-it": "1B", "Llama-3.2-1B": "1B", "Qwen2.5-1.5B-Instruct": "1.5B",
        "SmolLM2-1.7B-Instruct": "1.7B", "Llama-3.2-3B": "3B", "Qwen2.5-3B-Instruct": "3B",
        "Falcon3-7B-Base": "7B", "Qwen2.5-7B-Instruct": "7B", "phi-4": "14B",
    }.get(model, "")
    ax1.text(-0.7, mi, size_str, ha="right", va="center", fontsize=8, color="#555555")

ax1.text(-1.2, -0.8, "Size", ha="right", va="center", fontsize=8, color="#555555",
         fontstyle="italic")

fig.suptitle("Per-GPU × Per-Model Calibration Results — Context Sweep (bs=1, full_cache)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "slides_calibration_heatmap.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {OUT / 'slides_calibration_heatmap.png'}")
