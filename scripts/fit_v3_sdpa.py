"""Fit V3 SDPA extension constants with leave-one-out validation.

Extends the V3 latency model to SDPA attention by adding a superlinear
attention term that captures SDPA's context-scaling overhead:

    t = t_launch + param_bytes/BW_hbm + kv_bytes/BW_hbm
        + delta * N^phi * kv_head_layers

For FA2 this term is zero (BW degradation captures context scaling).
For SDPA the quadratic-ish term dominates at long context.

LOO procedure:
    1. Fit (delta, phi) on GH200, derive t_launch from 1K point
       → validate on A100-40GB
    2. Fit (delta, phi) on A100-40GB, derive t_launch from 1K point
       → validate on GH200
    3. If LOO errors are comparable to in-sample, combine both datasets
       for final constants.

Usage:
    python scripts/fit_v3_sdpa.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

# Make hwprop importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hwprop.specs import get_hardware_specs, get_model_configs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

SDPA_DATA = {
    "GH200": Path("results/benchmark/context_sweep_GH200.csv"),
    "A100_40GB": Path("results/benchmark_A100_40GB/context_sweep_A100_40GB.csv"),
}

MODEL_KEY = "LLaMA-3.2-3B"


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# V3 SDPA prediction
# ---------------------------------------------------------------------------

def predict_sdpa_ms(
    hw_name: str,
    context_lens: list[int],
    t_launch: float,
    delta: float,
    phi: float,
    hw_catalog: dict,
    model_cfg,
) -> np.ndarray:
    """Predict per-step decode latency (ms) under SDPA.

    t = t_launch + param_bytes/BW + kv_bytes/BW + delta * N^phi * kv_head_layers
    """
    hw = hw_catalog[hw_name]
    bw = hw.hbm_bandwidth
    param_time = model_cfg.param_bytes / bw
    kv_per_tok = model_cfg.kv_bytes_per_token  # all layers
    kv_head_layers = model_cfg.num_kv_heads * model_cfg.num_layers

    preds = []
    for N in context_lens:
        kv_time = (N * kv_per_tok) / bw
        attn_time = delta * (N ** phi) * kv_head_layers
        preds.append((t_launch + param_time + kv_time + attn_time) * 1000)
    return np.array(preds)


def fit_t_launch(
    hw_name: str,
    measured_1k_ms: float,
    delta: float,
    phi: float,
    hw_catalog: dict,
    model_cfg,
    short_ctx: int = 1024,
) -> float:
    """Back out t_launch from a single short-context measurement."""
    hw = hw_catalog[hw_name]
    bw = hw.hbm_bandwidth
    param_time = model_cfg.param_bytes / bw
    kv_per_tok = model_cfg.kv_bytes_per_token
    kv_head_layers = model_cfg.num_kv_heads * model_cfg.num_layers

    kv_time = (short_ctx * kv_per_tok) / bw
    attn_time = delta * (short_ctx ** phi) * kv_head_layers

    return measured_1k_ms / 1000 - param_time - kv_time - attn_time


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_delta_phi(
    hw_name: str,
    rows: list[dict],
    hw_catalog: dict,
    model_cfg,
) -> tuple[float, float, float]:
    """Fit (t_launch, delta, phi) jointly from a context sweep.

    Returns (t_launch, delta, phi).
    """
    hw = hw_catalog[hw_name]
    bw = hw.hbm_bandwidth
    param_time = model_cfg.param_bytes / bw
    kv_per_tok = model_cfg.kv_bytes_per_token
    kv_head_layers = model_cfg.num_kv_heads * model_cfg.num_layers

    ctx_lens = np.array([int(r["context_length"]) for r in rows], dtype=float)
    measured = np.array([float(r["measured_per_token_ms"]) for r in rows])  # ms

    def model(N, t_launch, delta, phi):
        kv_time = (N * kv_per_tok) / bw
        attn_time = delta * (N ** phi) * kv_head_layers
        return (t_launch + param_time + kv_time + attn_time) * 1000

    p0 = [0.020, 1e-15, 2.0]  # initial guess: 20ms launch, small delta, phi~2
    bounds = ([0.0, 0.0, 0.5], [0.10, 1e-10, 3.5])
    popt, _ = curve_fit(model, ctx_lens, measured, p0=p0, bounds=bounds, maxfev=10000)
    return float(popt[0]), float(popt[1]), float(popt[2])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def mae(predicted: np.ndarray, measured: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - measured) / measured) * 100)


def print_comparison(label: str, ctx_lens, measured, predicted):
    print(f"\n  {label}")
    print(f"  {'ctx':>8}  {'measured':>10}  {'predicted':>10}  {'err%':>7}")
    for N, m, p in zip(ctx_lens, measured, predicted):
        err = (p - m) / m * 100
        print(f"  {int(N):>8,}  {m:>10.3f}  {p:>10.3f}  {err:>+7.1f}%")
    print(f"  MAE: {mae(predicted, measured):.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    hw_catalog = get_hardware_specs()
    model_cfg = get_model_configs()[MODEL_KEY]

    datasets = {name: load_csv(path) for name, path in SDPA_DATA.items()}

    hw_names = list(datasets.keys())

    print("=" * 70)
    print("V3 SDPA Extension — Leave-One-Out Validation")
    print(f"Model: {MODEL_KEY}")
    print("=" * 70)

    loo_results = {}

    for train_hw in hw_names:
        val_hw = [h for h in hw_names if h != train_hw][0]
        train_rows = datasets[train_hw]
        val_rows = datasets[val_hw]

        # Fit on train_hw
        t_launch, delta, phi = fit_delta_phi(train_hw, train_rows, hw_catalog, model_cfg)

        print(f"\n{'─' * 70}")
        print(f"Train: {train_hw}  →  Validate: {val_hw}")
        print(f"  Fitted:  t_launch={t_launch*1000:.2f}ms  delta={delta:.4e}  phi={phi:.4f}")

        # In-sample
        train_ctxs = np.array([int(r["context_length"]) for r in train_rows], dtype=float)
        train_meas = np.array([float(r["measured_per_token_ms"]) for r in train_rows])
        train_pred = predict_sdpa_ms(train_hw, list(train_ctxs.astype(int)),
                                     t_launch, delta, phi, hw_catalog, model_cfg)
        print_comparison(f"In-sample  ({train_hw})", train_ctxs, train_meas, train_pred)

        # Out-of-sample: fit t_launch for val_hw from its 1K-context measurement
        val_1k_row = min(val_rows, key=lambda r: int(r["context_length"]))
        val_1k_ms = float(val_1k_row["measured_per_token_ms"])
        val_1k_ctx = int(val_1k_row["context_length"])
        t_launch_val = fit_t_launch(val_hw, val_1k_ms, delta, phi,
                                    hw_catalog, model_cfg, short_ctx=val_1k_ctx)

        val_ctxs = np.array([int(r["context_length"]) for r in val_rows], dtype=float)
        val_meas = np.array([float(r["measured_per_token_ms"]) for r in val_rows])
        val_pred = predict_sdpa_ms(val_hw, list(val_ctxs.astype(int)),
                                   t_launch_val, delta, phi, hw_catalog, model_cfg)
        print(f"\n  t_launch for {val_hw} (from 1K measurement): {t_launch_val*1000:.2f}ms")
        print_comparison(f"Out-of-sample ({val_hw})", val_ctxs, val_meas, val_pred)

        loo_results[train_hw] = {
            "t_launch": t_launch, "delta": delta, "phi": phi,
            "in_sample_mae": mae(train_pred, train_meas),
            "loo_mae": mae(val_pred, val_meas),
            "t_launch_val": t_launch_val,
        }

    # Combined fit
    print(f"\n{'─' * 70}")
    print("Combined fit (both GPUs)")
    all_rows_labeled = [(hw, r) for hw, rows in datasets.items() for r in rows]

    # Build combined arrays
    combined_ctx = np.array([int(r["context_length"]) for _, r in all_rows_labeled], dtype=float)
    combined_meas = np.array([float(r["measured_per_token_ms"]) for _, r in all_rows_labeled])
    combined_hw = [hw for hw, _ in all_rows_labeled]

    # Fit delta, phi jointly; t_launch per GPU
    from scipy.optimize import minimize

    def combined_loss(params):
        t_launch_gh200, t_launch_a100, delta, phi = params
        if delta <= 0 or phi <= 0:
            return 1e10
        total = 0.0
        for hw_name, rows in datasets.items():
            t_launch = t_launch_gh200 if hw_name == "GH200" else t_launch_a100
            ctxs = [int(r["context_length"]) for r in rows]
            meas = np.array([float(r["measured_per_token_ms"]) for r in rows])
            pred = predict_sdpa_ms(hw_name, ctxs, t_launch, delta, phi, hw_catalog, model_cfg)
            total += np.sum(((pred - meas) / meas) ** 2)
        return total

    x0 = [0.020, 0.020, 1e-15, 2.0]
    result = minimize(combined_loss, x0, method="Nelder-Mead",
                      options={"maxiter": 50000, "xatol": 1e-12, "fatol": 1e-10})
    t_l_gh200, t_l_a100, delta_c, phi_c = result.x

    print(f"  t_launch GH200:   {t_l_gh200*1000:.2f}ms")
    print(f"  t_launch A100-40: {t_l_a100*1000:.2f}ms")
    print(f"  delta:  {delta_c:.4e}")
    print(f"  phi:    {phi_c:.4f}")

    for hw_name, rows in datasets.items():
        t_launch = t_l_gh200 if hw_name == "GH200" else t_l_a100
        ctxs = [int(r["context_length"]) for r in rows]
        meas = np.array([float(r["measured_per_token_ms"]) for r in rows])
        pred = predict_sdpa_ms(hw_name, ctxs, t_launch, delta_c, phi_c, hw_catalog, model_cfg)
        print_comparison(f"Combined ({hw_name})", ctxs, meas, pred)

    # Summary
    print(f"\n{'=' * 70}")
    print("LOO Summary")
    print(f"  {'Train HW':<12} {'In-sample MAE':>15} {'LOO MAE':>10}")
    for train_hw, res in loo_results.items():
        val_hw = [h for h in hw_names if h != train_hw][0]
        print(f"  {train_hw:<12} {res['in_sample_mae']:>14.2f}%  {res['loo_mae']:>9.2f}%")

    print(f"\nConclusion:")
    loo_maes = [r["loo_mae"] for r in loo_results.values()]
    if max(loo_maes) < 10:
        print(f"  LOO MAE {max(loo_maes):.1f}% — constants generalize across NVIDIA SDPA hardware.")
        print(f"  Safe to use combined constants (delta={delta_c:.4e}, phi={phi_c:.4f}).")
    else:
        print(f"  LOO MAE {max(loo_maes):.1f}% — constants may not generalize. Need more SDPA hardware.")

    print(f"\nSuggested addition to v3_model.py:")
    print(f"  SDPA_DELTA = {delta_c:.6e}")
    print(f"  SDPA_PHI   = {phi_c:.4f}")
    print(f"  KNOWN_LAUNCH_OVERHEADS['GH200']    = {t_l_gh200:.5f}  # {t_l_gh200*1000:.1f}ms")
    print(f"  KNOWN_LAUNCH_OVERHEADS['A100_40GB'] = {t_l_a100:.5f}  # {t_l_a100*1000:.1f}ms")


if __name__ == "__main__":
    main()
