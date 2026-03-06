#!/usr/bin/env python3
"""Post-hoc latency simulation on real accuracy-eval rollouts.

Loads prompt/decode lengths from accuracy_results_final.jsonl, runs the
roofline cost model for each strategy x hardware x tier-split combination,
and writes latency results + a Pareto (accuracy vs latency) plot.

Usage:
    python scripts/run_latency_simulation.py
    python scripts/run_latency_simulation.py --results results/accuracy_results_final.jsonl
    python scripts/run_latency_simulation.py --output-dir results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hwprop.accuracy_eval import load_results, plot_pareto
from hwprop.eval_pipeline import compute_strategy_latency
from hwprop.specs import get_hardware_specs, get_model_configs


# ---------------------------------------------------------------------------
# Strategy metadata: budget_tokens and quantized flag for each eval strategy
# (mirrors the registry in accuracy_eval.get_strategies())
# ---------------------------------------------------------------------------
STRATEGY_META: dict[str, dict] = {
    "full_cache":        {"budget_tokens": None,  "quantized": False},
    "full_cache_int8":   {"budget_tokens": None,  "quantized": True},
    "window_128":        {"budget_tokens": 132,   "quantized": False},
    "window_256":        {"budget_tokens": 260,   "quantized": False},
    "window_512":        {"budget_tokens": 516,   "quantized": False},
    "window_1024":       {"budget_tokens": 1028,  "quantized": False},
    "h2o_128":           {"budget_tokens": 128,   "quantized": False},
    "h2o_256":           {"budget_tokens": 256,   "quantized": False},
    "h2o_512":           {"budget_tokens": 512,   "quantized": False},
    "h2o_1024":          {"budget_tokens": 1024,  "quantized": False},
    "snapkv_512":        {"budget_tokens": 512,   "quantized": False},
    "expected_attn_512": {"budget_tokens": 512,   "quantized": False},
}

# Tier splits: (hbm_frac, cpu_frac, disk_frac)
# hbm_frac = fraction of kept tokens staying in HBM (full precision)
# cpu_frac  = fraction offloaded to CPU RAM
# disk_frac = fraction offloaded to NVMe
TIER_SPLITS: list[tuple[float, float, float]] = [
    (1.0, 0.0, 0.0),   # all HBM (baseline)
    (0.7, 0.3, 0.0),   # 70% HBM, 30% CPU
    (0.5, 0.5, 0.0),   # 50% HBM, 50% CPU
    (0.3, 0.3, 0.4),   # 30% HBM, 30% CPU, 40% disk
    (0.5, 0.0, 0.5),   # 50% HBM, 50% disk
]


def compute_rollout_stats(
    results_path: str,
) -> dict[str, dict]:
    """Compute per-strategy mean prompt/decode lengths from the real rollouts.

    Returns a dict keyed by strategy_name with keys:
        mean_prompt_tokens, mean_decode_tokens, accuracy, n
    """
    results = load_results(results_path)

    by_strategy: dict[str, list] = {}
    for r in results:
        by_strategy.setdefault(r.strategy_name, []).append(r)

    stats = {}
    for strat, rows in by_strategy.items():
        n = len(rows)
        stats[strat] = {
            "mean_prompt_tokens": sum(r.prompt_tokens for r in rows) / n,
            "mean_decode_tokens": sum(r.tokens_generated for r in rows) / n,
            "accuracy": sum(r.correct for r in rows) / n,
            "n": n,
        }
    return stats


def run_simulation(
    results_path: str,
    model_name: str = "Qwen2.5-7B",
    decision_interval: int = 64,
) -> list[dict]:
    """Run roofline latency simulation across strategy x hardware x tier_split.

    Uses real mean prompt/decode lengths from the accuracy eval rollouts.
    Returns a list of dicts suitable for JSONL output.
    """
    rollout_stats = compute_rollout_stats(results_path)
    hardware_configs = get_hardware_specs()
    model_config = get_model_configs()[model_name]

    rows: list[dict] = []
    strategies = list(rollout_stats.keys())
    total = len(strategies) * len(hardware_configs) * len(TIER_SPLITS)
    count = 0

    print(f"Model: {model_name}")
    print(f"Strategies: {len(strategies)}, Hardware: {len(hardware_configs)}, Tier splits: {len(TIER_SPLITS)}")
    print(f"Total runs: {total}\n")

    for strat_name in strategies:
        if strat_name not in STRATEGY_META:
            print(f"  Skipping unknown strategy: {strat_name}")
            continue

        meta = STRATEGY_META[strat_name]
        stats = rollout_stats[strat_name]
        prompt_len = round(stats["mean_prompt_tokens"])
        decode_steps = round(stats["mean_decode_tokens"])

        for hw_name, hw in hardware_configs.items():
            for hbm_f, cpu_f, disk_f in TIER_SPLITS:
                count += 1

                # Skip disk splits on hardware with no NVMe
                if disk_f > 0 and hw.disk_capacity == 0:
                    continue

                result = compute_strategy_latency(
                    strategy_name=strat_name,
                    budget_tokens=meta["budget_tokens"],
                    hardware=hw,
                    model_config=model_config,
                    prompt_len=prompt_len,
                    decode_steps=decode_steps,
                    decision_interval=decision_interval,
                    offload_frac=cpu_f,
                    disk_frac=disk_f,
                    quantized=meta["quantized"],
                )

                row = {
                    **result,
                    "hbm_frac": hbm_f,
                    "cpu_frac": cpu_f,
                    "disk_frac": disk_f,
                    "accuracy": stats["accuracy"],
                    "n_tasks": stats["n"],
                    "prompt_len": prompt_len,
                    "decode_steps": decode_steps,
                    "model": model_name,
                }
                rows.append(row)

                if count % 100 == 0 or count == total:
                    print(f"  [{count}/{total}] {strat_name} / {hw_name} / split={hbm_f:.0%}HBM+{cpu_f:.0%}CPU+{disk_f:.0%}disk")

    return rows


def run_context_sweep(
    model_name: str = "Qwen2.5-7B",
    context_lengths: list[int] | None = None,
    prompt_len: int = 128,
    decision_interval: int = 64,
) -> list[dict]:
    """Sweep decode steps across context lengths for all strategies × hardware.

    Uses the all-HBM tier split to isolate the pure effect of context length
    on latency spread between strategies.  prompt_len is held fixed; decode_steps
    = context_length - prompt_len.
    """
    if context_lengths is None:
        context_lengths = [512, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    hardware_configs = get_hardware_specs()
    model_config = get_model_configs()[model_name]

    rows: list[dict] = []
    total = len(STRATEGY_META) * len(hardware_configs) * len(context_lengths)
    count = 0

    print(f"\nContext sweep — model: {model_name}, prompt_len: {prompt_len}")
    print(f"Context lengths: {context_lengths}")
    print(f"Total runs: {total}\n")

    for ctx_len in context_lengths:
        decode_steps = max(1, ctx_len - prompt_len)

        for strat_name, meta in STRATEGY_META.items():
            for hw_name, hw in hardware_configs.items():
                count += 1
                result = compute_strategy_latency(
                    strategy_name=strat_name,
                    budget_tokens=meta["budget_tokens"],
                    hardware=hw,
                    model_config=model_config,
                    prompt_len=prompt_len,
                    decode_steps=decode_steps,
                    decision_interval=decision_interval,
                    offload_frac=0.0,
                    disk_frac=0.0,
                    quantized=meta["quantized"],
                )
                rows.append({
                    **result,
                    "context_length": ctx_len,
                    "prompt_len": prompt_len,
                    "decode_steps": decode_steps,
                    "model": model_name,
                    "hbm_frac": 1.0,
                    "cpu_frac": 0.0,
                    "disk_frac": 0.0,
                })

        if ctx_len in (context_lengths[0], context_lengths[-1]) or count % 500 == 0:
            print(f"  ctx={ctx_len:>7} tokens  [{count}/{total}]")

    return rows


def print_summary(rows: list[dict]) -> None:
    """Print a summary table: strategy x hardware for the all-HBM split."""
    hbm_only = [r for r in rows if r["cpu_frac"] == 0.0 and r["disk_frac"] == 0.0]

    # Pivot: strategy -> hardware -> latency
    strategies = sorted({r["strategy"] for r in hbm_only})
    hw_names = sorted({r["hardware"] for r in hbm_only})

    # Header
    col_w = 14
    header = f"{'strategy':<22}" + "".join(f"{hw[:col_w]:>{col_w}}" for hw in hw_names)
    print("\n" + "=" * len(header))
    print("Mean latency per decode step (ms) — all-HBM split")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    lookup = {(r["strategy"], r["hardware"]): r["mean_latency_ms"] for r in hbm_only}
    for strat in strategies:
        acc = next(r["accuracy"] for r in hbm_only if r["strategy"] == strat)
        row_str = f"{strat:<22}"
        for hw in hw_names:
            val = lookup.get((strat, hw))
            cell = f"{val:.2f}" if val is not None else "  n/a"
            row_str += f"{cell:>{col_w}}"
        print(f"{row_str}   acc={acc:.1%}")

    print("=" * len(header))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default="results/accuracy_results_final.jsonl",
        help="Path to accuracy results JSONL",
    )
    parser.add_argument(
        "--model",
        default="Qwen2.5-7B",
        help="Model key from get_model_configs() to use for roofline math",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to write latency JSONL and Pareto plot",
    )
    parser.add_argument(
        "--decision-interval",
        type=int,
        default=64,
        help="Eviction interval (tokens between cache decisions)",
    )
    parser.add_argument(
        "--context-lengths",
        default="512,2048,4096,8192,16384,32768,65536,131072",
        help="Comma-separated context lengths for the sweep (tokens)",
    )
    parser.add_argument(
        "--sweep-prompt-len",
        type=int,
        default=128,
        help="Fixed prompt length used for the context sweep",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latency_path = output_dir / "latency_simulation.jsonl"
    pareto_path = output_dir / "pareto.png"

    # Run simulation
    rows = run_simulation(
        results_path=args.results,
        model_name=args.model,
        decision_interval=args.decision_interval,
    )

    # Save JSONL
    with latency_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nSaved {len(rows)} rows to {latency_path}")

    # Print summary table
    print_summary(rows)

    # Pareto plot: accuracy (real) vs latency (simulated), all-HBM split only
    hbm_only_rows = [r for r in rows if r["cpu_frac"] == 0.0 and r["disk_frac"] == 0.0]
    accuracy_results = load_results(args.results)
    plot_pareto(accuracy_results, hbm_only_rows, str(pareto_path))
    print(f"Saved Pareto plot to {pareto_path}")

    # Context length sweep
    context_lengths = [int(x) for x in args.context_lengths.split(",") if x.strip()]
    sweep_rows = run_context_sweep(
        model_name=args.model,
        context_lengths=context_lengths,
        prompt_len=args.sweep_prompt_len,
        decision_interval=args.decision_interval,
    )
    sweep_path = output_dir / "latency_context_sweep.jsonl"
    with sweep_path.open("w") as f:
        for row in sweep_rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nSaved {len(sweep_rows)} rows to {sweep_path}")


if __name__ == "__main__":
    main()
