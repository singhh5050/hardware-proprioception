#!/usr/bin/env python3
"""Runner script for real accuracy evaluation with KV cache strategies.

Usage:
    python eval_accuracy.py [--model MODEL] [--num-tasks N] [--seed SEED]
                            [--decision-interval N] [--max-new-tokens N]
                            [--strategies NAMES] [--output-dir DIR] [--device DEV]
                            [--skip-latency] [--latency-hardware NAMES]

Examples:
    # Quick smoke test (3 tasks, 2 strategies)
    python eval_accuracy.py --num-tasks 3 --strategies full_cache,window_512

    # Full run (200 tasks, all 12 strategies)
    python eval_accuracy.py --num-tasks 200

    # Skip latency simulation
    python eval_accuracy.py --num-tasks 50 --skip-latency
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MATH accuracy evaluation with KV cache strategies"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Math-7B-Instruct",
        help="Model name or path (default: Qwen/Qwen2.5-Math-7B-Instruct)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=200,
        help="Number of MATH problems to evaluate (default: 200)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for task sampling"
    )
    parser.add_argument(
        "--decision-interval",
        type=int,
        default=64,
        help="Compression interval for DecodingPress (default: 64)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max tokens to generate per problem (default: 2048)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated strategy names (default: all 12)",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_outputs_accuracy",
        help="Output directory (default: eval_outputs_accuracy)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device map for model loading (default: auto)",
    )
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Skip post-hoc latency simulation",
    )
    parser.add_argument(
        "--latency-hardware",
        type=str,
        default=None,
        help="Comma-separated hardware names for latency sweep (default: all 16)",
    )
    parser.add_argument(
        "--math-dataset",
        default="HuggingFaceH4/MATH-500",
        help="HuggingFace dataset name for MATH problems",
    )
    return parser.parse_args()


def print_summary_table(results):
    """Print a formatted accuracy summary table."""
    from hwprop.accuracy_eval import AccuracyResult

    by_strategy: dict[str, list[AccuracyResult]] = {}
    for r in results:
        by_strategy.setdefault(r.strategy_name, []).append(r)

    print("\n" + "=" * 75)
    print(f"{'Strategy':<22} {'Accuracy':>10} {'Correct':>10} {'Total':>8} {'Avg Tokens':>12}")
    print("-" * 75)

    for name, strat_results in by_strategy.items():
        correct = sum(1 for r in strat_results if r.correct)
        total = len(strat_results)
        acc = correct / total if total > 0 else 0
        avg_tokens = sum(r.tokens_generated for r in strat_results) / total if total > 0 else 0
        print(f"{name:<22} {acc:>9.1%} {correct:>10} {total:>8} {avg_tokens:>12.0f}")

    print("=" * 75)


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from hwprop.accuracy_eval import (
        get_strategies,
        load_math_tasks,
        plot_accuracy_by_strategy,
        plot_accuracy_vs_budget,
        plot_pareto,
        run_accuracy_eval,
        save_results,
    )

    # 1. Load MATH tasks
    print(f"Loading {args.num_tasks} MATH tasks (seed={args.seed})...")
    tasks = load_math_tasks(
        num_tasks=args.num_tasks,
        seed=args.seed,
        dataset_name=args.math_dataset,
    )
    print(f"Loaded {len(tasks)} tasks")

    # 2. Build strategies
    all_strategies = get_strategies(decision_interval=args.decision_interval)
    if args.strategies:
        selected = [s.strip() for s in args.strategies.split(",")]
        strategies = {}
        for name in selected:
            if name not in all_strategies:
                print(f"WARNING: Unknown strategy '{name}'. Available: {list(all_strategies.keys())}")
                continue
            strategies[name] = all_strategies[name]
        if not strategies:
            print("ERROR: No valid strategies selected.")
            return 1
    else:
        strategies = all_strategies

    print(f"Strategies: {list(strategies.keys())}")

    # 3. Run accuracy eval
    t0 = time.time()
    results = run_accuracy_eval(
        model_name=args.model,
        tasks=tasks,
        strategies=strategies,
        decision_interval=args.decision_interval,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    elapsed = time.time() - t0
    print(f"\nAccuracy eval completed in {elapsed:.0f}s")

    # 4. Save results
    jsonl_path = os.path.join(args.output_dir, "accuracy_results.jsonl")
    save_results(results, jsonl_path)

    # 5. Print summary
    print_summary_table(results)

    # 6. Generate accuracy plots
    try:
        plot_accuracy_by_strategy(
            results,
            os.path.join(args.output_dir, "accuracy_by_strategy.png"),
        )
        plot_accuracy_vs_budget(
            results,
            os.path.join(args.output_dir, "accuracy_vs_budget.png"),
        )
    except Exception as e:
        print(f"WARNING: Plot generation failed: {e}")

    # 7. Latency sweep (optional)
    latency_results = []
    if not args.skip_latency:
        try:
            from hwprop.eval_pipeline import compute_latency_sweep
            from hwprop.specs import get_hardware_specs, get_model_configs

            hw_specs = get_hardware_specs()
            if args.latency_hardware:
                selected_hw = [s.strip() for s in args.latency_hardware.split(",")]
                hw_configs = {k: v for k, v in hw_specs.items() if k in selected_hw}
            else:
                hw_configs = hw_specs

            # Find model config matching the eval model
            model_configs = get_model_configs()
            # Use Qwen2.5-14B as closest proxy for 7B (architecture match)
            model_key = "Qwen2.5-14B"
            for k in model_configs:
                if "qwen" in k.lower() and ("7b" in k.lower() or "8b" in k.lower()):
                    model_key = k
                    break
            model_config = model_configs[model_key]

            # Build strategies with budgets for latency
            strategies_with_budgets = []
            for name, strat in strategies.items():
                strategies_with_budgets.append({
                    "strategy": name,
                    "budget_tokens": strat.budget_tokens,
                    "quantized": strat.quantized,
                })

            print(f"\nRunning latency sweep ({len(hw_configs)} hardware configs)...")
            latency_results = compute_latency_sweep(
                strategies_with_budgets=strategies_with_budgets,
                hardware_configs=hw_configs,
                model_config=model_config,
                prompt_len=256,
                decode_steps=512,
            )

            # Save latency results
            csv_path = os.path.join(args.output_dir, "latency_results.csv")
            if latency_results:
                fieldnames = list(latency_results[0].keys())
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(latency_results)
                print(f"Saved {len(latency_results)} latency results to {csv_path}")
        except Exception as e:
            print(f"WARNING: Latency sweep failed: {e}")

    # 8. Pareto plot (if we have both accuracy and latency)
    if latency_results:
        try:
            plot_pareto(
                results,
                latency_results,
                os.path.join(args.output_dir, "pareto_plot.png"),
            )
        except Exception as e:
            print(f"WARNING: Pareto plot failed: {e}")

    print(f"\nAll outputs saved to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
