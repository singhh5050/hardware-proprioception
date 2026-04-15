"""Validate simulator against grid benchmark results (results/grid/).

Reads JSONL files produced by benchmark_full_sweep.py and compares
simulate_latency() predictions against measured per-token latency.

Reports MAE and Spearman rank correlation per:
  - (model, hardware, sweep_type)
  - aggregated across all models on a hardware
  - aggregated across all hardware for a model

The key correlation signal:
  - context sweep: does the simulator correctly predict how latency scales?
  - strategy sweep: does the simulator correctly rank eviction strategies?
  - batch sweep: does the simulator correctly scale with batch size?

Usage:
    python scripts/validate_grid.py
    python scripts/validate_grid.py --grid-dir results/grid
    python scripts/validate_grid.py --sweep context     # only context sweep
    python scripts/validate_grid.py --hw H100_SXM       # only one GPU
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hwprop.specs import HardwareSpec, ModelConfig, get_hardware_specs
from hwprop.simulator import simulate_latency


# ---------------------------------------------------------------------------
# HuggingFace model name → hwprop model config key (or None = build from JSONL)
# ---------------------------------------------------------------------------
HF_TO_HWPROP: dict[str, str] = {
    "meta-llama/Llama-3.2-3B":          "LLaMA-3.2-3B",
    "meta-llama/Llama-3.2-1B":          "LLaMA-3.2-1B",
    "meta-llama/Llama-3.1-8B":          "LLaMA-3.1-8B",
    "Qwen/Qwen2.5-7B-Instruct":         "Qwen2.5-7B",
    "Qwen/Qwen2.5-14B-Instruct":        "Qwen2.5-14B",
    "Qwen/Qwen2.5-3B-Instruct":         "Qwen2.5-3B",
    "Qwen/Qwen2.5-1.5B-Instruct":       "Qwen2.5-1.5B",
    "microsoft/phi-4":                  "Phi-4-14B",
    "google/gemma-3-1b-it":             "Gemma-3-1B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "SmolLM2-1.7B",
    "tiiuae/Falcon3-7B-Base":           "Falcon3-7B",
}


def model_short_name(hf_name: str) -> str:
    return hf_name.split("/")[-1]


def build_model_config_from_row(row: dict) -> ModelConfig | None:
    """Build a ModelConfig from JSONL fields recorded by benchmark_full_sweep.py.

    The benchmark records: param_bytes, kv_bytes_per_token, num_layers,
    num_kv_heads, head_dim.  We reconstruct the config using these directly,
    using dummy values for fields that only affect computed properties we
    already have measured values for.
    """
    try:
        param_bytes     = int(row["param_bytes"])
        kv_per_tok      = int(row["kv_bytes_per_token"])
        num_layers      = int(row["num_layers"])
        num_kv_heads    = int(row["num_kv_heads"])
        head_dim        = int(row["head_dim"])
    except (KeyError, TypeError, ValueError):
        return None

    # Back-compute num_params from param_bytes (assuming bf16)
    bytes_per_param = 2
    num_params = param_bytes // bytes_per_param

    # Back-compute d_model from kv_bytes_per_token:
    # kv_bytes_per_token = num_layers * 2 * num_kv_heads * head_dim * bytes_per_param
    kv_per_tok_computed = num_layers * 2 * num_kv_heads * head_dim * bytes_per_param
    if abs(kv_per_tok_computed - kv_per_tok) > 16:
        # Architecture doesn't match standard formula — skip
        return None

    # Estimate d_model: assume standard GQA ratio num_heads = num_kv_heads * gqa_factor
    # We don't need d_model precisely — just need it consistent with param_bytes.
    # Use a dummy that won't affect kv_bytes_per_token or param_bytes properties.
    # We override param_bytes and kv_bytes_per_token via a subclass trick below.
    name = model_short_name(row.get("model_name", "unknown"))

    return _DirectModelConfig(
        name=name,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        _param_bytes_override=param_bytes,
        _kv_bytes_per_token_override=kv_per_tok,
    )


@dataclass
class _DirectModelConfig:
    """Lightweight ModelConfig built directly from benchmark JSONL fields.

    Skips the full architectural reconstruction — only stores what the
    simulator actually needs: param_bytes, kv_bytes_per_token, num_layers,
    num_kv_heads, head_dim, name.
    """
    name: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    _param_bytes_override: int
    _kv_bytes_per_token_override: int

    @property
    def param_bytes(self) -> int:
        return self._param_bytes_override

    @property
    def kv_bytes_per_token(self) -> int:
        return self._kv_bytes_per_token_override

    def kv_bytes_per_token_per_layer(self) -> int:
        return self._kv_bytes_per_token_override // self.num_layers

    # Stubs for anything the simulator might reach for
    @property
    def num_params(self) -> int:
        return self._param_bytes_override // 2

    @property
    def bytes_per_param(self) -> int:
        return 2

    # num_heads: only used for FLOP counting, not bandwidth model
    @property
    def num_heads(self) -> int:
        return self.num_kv_heads  # conservative; doesn't affect BW model


# ---------------------------------------------------------------------------
# Load JSONL
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_grid(grid_dir: Path, hw_filter: str | None, sweep_filter: str | None) -> list[dict]:
    rows = []
    for jsonl_path in sorted(grid_dir.rglob("*.jsonl")):
        for row in iter_jsonl(jsonl_path):
            if hw_filter and row.get("hardware_key") != hw_filter:
                continue
            sweep = infer_sweep(row)
            if sweep_filter and sweep != sweep_filter:
                continue
            row["_sweep"] = sweep
            row["_jsonl"] = str(jsonl_path)
            rows.append(row)
    return rows


def infer_sweep(row: dict) -> str:
    """Infer sweep type from row fields."""
    bs   = int(row.get("batch_size", 1))
    strat = row.get("strategy", "full_cache")
    if bs > 1:
        return "batch"
    if strat != "full_cache":
        return "strategy"
    return "context"


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(row: dict, hw_catalog: dict) -> float | None:
    hw_key   = row.get("hardware_key")
    hf_model = row.get("model_name", "")
    ctx      = int(row.get("context_length", 0))
    decode   = int(row.get("num_decode_steps", 128))
    bs       = int(row.get("batch_size", 1))
    strategy = row.get("strategy", "full_cache")

    if hw_key not in hw_catalog:
        return None

    # Get model config: prefer hwprop catalog, fall back to JSONL reconstruction
    hwprop_key = HF_TO_HWPROP.get(hf_model)
    if hwprop_key:
        model_arg = hwprop_key
    else:
        model_cfg = build_model_config_from_row(row)
        if model_cfg is None:
            return None
        model_arg = model_cfg

    try:
        result = simulate_latency(
            hardware=hw_key,
            model=model_arg,
            strategy=strategy,
            prompt_len=ctx,
            decode_steps=decode,
            batch_size=bs,
        )
        # simulate_latency returns mean_per_token_ms (per token, not per step)
        # benchmark records mean_ms_per_token — same units
        return result.mean_per_token_ms
    except Exception as e:
        return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mae_pct(pred: np.ndarray, meas: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - meas) / meas) * 100)


def spearman(pred: np.ndarray, meas: np.ndarray) -> float:
    if len(pred) < 2:
        return float("nan")
    r, _ = spearmanr(pred, meas)
    return float(r)


def pairwise_acc(pred: np.ndarray, meas: np.ndarray) -> float:
    correct = total = 0
    for i in range(len(pred)):
        for j in range(i + 1, len(pred)):
            if abs(meas[i] - meas[j]) < 1e-6:
                continue  # skip ties
            total += 1
            if (pred[i] > pred[j]) == (meas[i] > meas[j]):
                correct += 1
    return correct / total if total > 0 else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-dir", default="results/grid")
    parser.add_argument("--sweep", choices=["context", "strategy", "batch"], default=None)
    parser.add_argument("--hw", default=None, help="Filter to one hardware key")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    grid_dir = Path(args.grid_dir)
    if not grid_dir.exists():
        print(f"Grid dir not found: {grid_dir}")
        print("No results yet — run benchmark_full_sweep.py on hardware first.")
        sys.exit(0)

    hw_catalog = get_hardware_specs()
    rows = load_grid(grid_dir, hw_filter=args.hw, sweep_filter=args.sweep)

    if not rows:
        print(f"No rows found in {grid_dir} (hw={args.hw}, sweep={args.sweep})")
        sys.exit(0)

    print(f"Loaded {len(rows)} rows from {grid_dir}")

    # Run predictions
    results = []
    skipped = 0
    for row in rows:
        meas = row.get("mean_ms_per_token")
        if meas is None:
            skipped += 1
            continue
        pred = predict(row, hw_catalog)
        if pred is None:
            skipped += 1
            continue
        results.append({
            "hw":       row["hardware_key"],
            "model":    model_short_name(row.get("model_name", "?")),
            "hf_model": row.get("model_name", "?"),
            "sweep":    row["_sweep"],
            "ctx":      int(row.get("context_length", 0)),
            "bs":       int(row.get("batch_size", 1)),
            "strategy": row.get("strategy", "full_cache"),
            "meas":     float(meas),
            "pred":     float(pred),
        })

    print(f"Predicted {len(results)} / {len(rows)} rows ({skipped} skipped — unknown HW or model)\n")

    if not results:
        print("Nothing to report.")
        return

    # Group by (hw, model, sweep)
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for r in results:
        groups[(r["hw"], r["model"], r["sweep"])].append(r)

    all_meas, all_pred = [], []
    summary = []

    sweeps_to_show = [args.sweep] if args.sweep else ["context", "strategy", "batch"]

    for sweep_type in sweeps_to_show:
        sweep_rows = [r for r in results if r["sweep"] == sweep_type]
        if not sweep_rows:
            continue

        print(f"\n{'=' * 80}")
        print(f"SWEEP: {sweep_type.upper()}")
        print(f"{'=' * 80}")

        hw_model_groups: dict[tuple, list] = defaultdict(list)
        for r in sweep_rows:
            hw_model_groups[(r["hw"], r["model"])].append(r)

        for (hw, model), group in sorted(hw_model_groups.items()):
            meas_arr = np.array([r["meas"] for r in group])
            pred_arr = np.array([r["pred"] for r in group])
            mae  = mae_pct(pred_arr, meas_arr)
            sp   = spearman(pred_arr, meas_arr)
            pw   = pairwise_acc(pred_arr, meas_arr)
            n    = len(group)

            all_meas.extend(meas_arr)
            all_pred.extend(pred_arr)
            summary.append((hw, model, sweep_type, n, mae, sp, pw))

            print(f"\n  {hw} / {model} (n={n})")
            if args.verbose:
                if sweep_type == "context":
                    print(f"  {'ctx':>8}  {'meas':>8}  {'pred':>8}  {'err%':>7}")
                    for r in sorted(group, key=lambda x: x["ctx"]):
                        err = (r["pred"] - r["meas"]) / r["meas"] * 100
                        print(f"  {r['ctx']:>8,}  {r['meas']:>8.2f}  {r['pred']:>8.2f}  {err:>+7.1f}%")
                elif sweep_type == "strategy":
                    for ctx in sorted(set(r["ctx"] for r in group)):
                        ctx_group = [r for r in group if r["ctx"] == ctx]
                        print(f"  ctx={ctx:,}")
                        for r in sorted(ctx_group, key=lambda x: x["meas"]):
                            err = (r["pred"] - r["meas"]) / r["meas"] * 100
                            print(f"    {r['strategy']:<22}  meas={r['meas']:>7.2f}  pred={r['pred']:>7.2f}  {err:>+6.1f}%")
                elif sweep_type == "batch":
                    print(f"  {'bs':>4}  {'meas':>8}  {'pred':>8}  {'err%':>7}")
                    for r in sorted(group, key=lambda x: x["bs"]):
                        err = (r["pred"] - r["meas"]) / r["meas"] * 100
                        print(f"  {r['bs']:>4}  {r['meas']:>8.2f}  {r['pred']:>8.2f}  {err:>+7.1f}%")

            print(f"  MAE={mae:.1f}%  Spearman={sp:+.4f}  Pairwise={pw:.0%}")

    # Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"  {'HW':<14} {'Model':<22} {'Sweep':<10} {'n':>4} {'MAE':>7} {'Spearman':>9} {'Pairwise':>9}")
    print(f"  {'-'*14} {'-'*22} {'-'*10} {'-'*4} {'-'*7} {'-'*9} {'-'*9}")
    for hw, model, sweep, n, mae, sp, pw in summary:
        print(f"  {hw:<14} {model:<22} {sweep:<10} {n:>4} {mae:>6.1f}% {sp:>+9.4f} {pw:>8.0%}")

    if all_meas:
        all_meas_arr = np.array(all_meas)
        all_pred_arr = np.array(all_pred)
        print(f"\n  Overall ({len(all_meas)} points):")
        print(f"    MAE={mae_pct(all_pred_arr, all_meas_arr):.1f}%  "
              f"Spearman={spearman(all_pred_arr, all_meas_arr):+.4f}  "
              f"Pairwise={pairwise_acc(all_pred_arr, all_meas_arr):.0%}")


if __name__ == "__main__":
    main()
