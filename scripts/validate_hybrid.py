"""Validate hybrid prediction strategy: V3 for FA2, LLMSimulator for SDPA.

For each benchmark dataset:
  - FA2  hardware → V3 model (universal constants, t_launch from 1K point)
  - SDPA hardware → LLMSimulator (calibrated OverheadProfile)

Reports MAE and Spearman rank correlation for each dataset,
and a combined summary across all benchmarks.

Usage:
    python scripts/validate_hybrid.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hwprop.specs import get_hardware_specs, get_model_configs
from hwprop.v3_model import predict_step_ms, fit_launch_overhead, ALPHA, BETA, GAMMA
from hwprop.simulator import simulate_latency


# ---------------------------------------------------------------------------
# Benchmark datasets
# We pick the canonical single-batch, 128-step sweeps only.
# Batch sweeps use only one context point so they don't test context scaling.
# ---------------------------------------------------------------------------
DATASETS = [
    # (label, csv_path, hw_key, model_key, attn_impl, bs)
    ("H100 / LLaMA-3.2-3B / FA2",
     "results/benchmark/context_sweep_H100_SXM.csv",
     "H100_SXM", "LLaMA-3.2-3B", "flash_attention_2", 1),

    ("H100 / Qwen2.5-7B / FA2  [cross-model]",
     "results/benchmark_cross_model/context_sweep_H100_SXM.csv",
     "H100_SXM", "Qwen2.5-7B", "flash_attention_2", 1),

    ("H200 / LLaMA-3.2-3B / FA2",
     "results/benchmark_H200/context_sweep_H200.csv",
     "H200", "LLaMA-3.2-3B", "flash_attention_2", 1),

    ("RTX 5090 / LLaMA-3.2-3B / FA2",
     "results/benchmark_RTX_5090/context_sweep_RTX_5090.csv",
     "RTX_5090", "LLaMA-3.2-3B", "flash_attention_2", 1),

    ("A100-80GB / Qwen2.5-7B / FA2  [cross-hw+model]",
     "results/benchmark_cross_model_a100/context_sweep_A100_80GB.csv",
     "A100_80GB", "Qwen2.5-7B", "flash_attention_2", 1),

    ("GH200 / LLaMA-3.2-3B / SDPA",
     "results/benchmark/context_sweep_GH200.csv",
     "GH200", "LLaMA-3.2-3B", "sdpa", 1),

    ("A100-40GB / LLaMA-3.2-3B / SDPA",
     "results/benchmark_A100_40GB/context_sweep_A100_40GB.csv",
     "A100_40GB", "LLaMA-3.2-3B", "sdpa", 1),
]


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_v3(hw_key: str, model_key: str, rows: list[dict]) -> np.ndarray:
    """V3 prediction for FA2 hardware. Derives t_launch from shortest-context row."""
    hw_catalog = get_hardware_specs()
    model_cfg = get_model_configs()[model_key]
    hw = hw_catalog[hw_key]

    # Derive t_launch from the shortest-context measurement
    short_row = min(rows, key=lambda r: int(r["context_length"]))
    short_ctx = int(short_row["context_length"])
    short_ms = float(short_row["measured_per_token_ms"])
    t_launch = fit_launch_overhead(hw, model_cfg, short_ms, short_ctx)

    preds = []
    for r in rows:
        ctx = int(r["context_length"])
        preds.append(predict_step_ms(hw, model_cfg, ctx, t_launch=t_launch))
    return np.array(preds)


def predict_llmsim(hw_key: str, model_key: str, rows: list[dict], bs: int) -> np.ndarray:
    """LLMSimulator prediction for SDPA hardware using calibrated profiles."""
    preds = []
    for r in rows:
        ctx = int(r["context_length"])
        decode_steps = int(r["decode_steps"])
        result = simulate_latency(
            hardware=hw_key,
            model=model_key,
            prompt_len=ctx,
            decode_steps=decode_steps,
            batch_size=bs,
        )
        preds.append(result.mean_per_token_ms)
    return np.array(preds)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mae_pct(pred: np.ndarray, meas: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - meas) / meas) * 100)

def spearman(pred: np.ndarray, meas: np.ndarray) -> float:
    r, _ = spearmanr(pred, meas)
    return float(r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Hybrid Validation: V3 (FA2) vs LLMSimulator (SDPA)")
    print("=" * 80)

    all_meas, all_pred_hybrid, all_pred_llmsim = [], [], []
    summary_rows = []

    for label, csv_path, hw_key, model_key, attn_impl, bs in DATASETS:
        path = Path(csv_path)
        if not path.exists():
            print(f"\nSKIP (file not found): {label}")
            continue

        with open(path) as f:
            rows = list(csv.DictReader(f))

        if not rows:
            continue

        meas = np.array([float(r["measured_per_token_ms"]) for r in rows])
        ctx_lens = [int(r["context_length"]) for r in rows]

        # Hybrid prediction
        if attn_impl == "flash_attention_2":
            pred_hybrid = predict_v3(hw_key, model_key, rows)
            method = "V3"
        else:
            pred_hybrid = predict_llmsim(hw_key, model_key, rows, bs)
            method = "LLMSim"

        # LLMSimulator prediction for comparison
        pred_llmsim = predict_llmsim(hw_key, model_key, rows, bs)

        mae_h = mae_pct(pred_hybrid, meas)
        mae_l = mae_pct(pred_llmsim, meas)
        sp_h  = spearman(pred_hybrid, meas)
        sp_l  = spearman(pred_llmsim, meas)

        print(f"\n{'─' * 80}")
        print(f"{label}")
        print(f"  Method: {method}  |  n={len(rows)} pts  |  ctx {min(ctx_lens):,}–{max(ctx_lens):,}")
        print(f"  {'ctx':>8}  {'measured':>10}  {'hybrid':>10}  {'err%':>7}  {'llmsim':>10}  {'err%':>7}")
        for r, m, ph, pl in zip(rows, meas, pred_hybrid, pred_llmsim):
            eh = (ph - m) / m * 100
            el = (pl - m) / m * 100
            print(f"  {int(r['context_length']):>8,}  {m:>10.3f}  {ph:>10.3f}  {eh:>+7.1f}%  {pl:>10.3f}  {el:>+7.1f}%")
        print(f"  MAE     — hybrid: {mae_h:5.1f}%   llmsim: {mae_l:5.1f}%  {'✓ hybrid wins' if mae_h < mae_l else '✓ llmsim wins'}")
        print(f"  Spearman— hybrid: {sp_h:5.3f}   llmsim: {sp_l:5.3f}")

        summary_rows.append((label, method, mae_h, mae_l, sp_h, sp_l, len(rows)))
        all_meas.extend(meas)
        all_pred_hybrid.extend(pred_hybrid)
        all_pred_llmsim.extend(pred_llmsim)

    # Overall summary
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"  {'Dataset':<48} {'method':>7} {'MAE_hyb':>9} {'MAE_sim':>9} {'Sp_hyb':>8} {'Sp_sim':>8}")
    print(f"  {'─'*48} {'─'*7} {'─'*9} {'─'*9} {'─'*8} {'─'*8}")
    for label, method, mae_h, mae_l, sp_h, sp_l, n in summary_rows:
        short = label[:47]
        print(f"  {short:<48} {method:>7} {mae_h:>8.1f}% {mae_l:>8.1f}% {sp_h:>8.3f} {sp_l:>8.3f}")

    all_meas = np.array(all_meas)
    all_pred_hybrid = np.array(all_pred_hybrid)
    all_pred_llmsim = np.array(all_pred_llmsim)

    print(f"\n  Overall (all {len(all_meas)} points):")
    print(f"    Hybrid  — MAE: {mae_pct(all_pred_hybrid, all_meas):.1f}%   Spearman: {spearman(all_pred_hybrid, all_meas):.4f}")
    print(f"    LLMSim  — MAE: {mae_pct(all_pred_llmsim, all_meas):.1f}%   Spearman: {spearman(all_pred_llmsim, all_meas):.4f}")


if __name__ == "__main__":
    main()
