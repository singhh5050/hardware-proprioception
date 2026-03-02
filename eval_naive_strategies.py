#!/usr/bin/env python3
"""Run eval-only baselines and plot naive strategy tradeoffs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from hwprop.eval_pipeline import (
    generate_tasks,
    generate_tasks_from_math_dataset,
    get_naive_strategies,
    plot_budget_sweep,
    plot_results,
    run_budget_sweep,
    run_eval,
)
from hwprop.specs import get_hardware_specs, get_model_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="H100_SXM", help="Hardware key from get_hardware_specs()")
    parser.add_argument("--model", default="LLaMA-3.1-8B", help="Model key from get_model_configs()")
    parser.add_argument("--task-set", default="mixed", choices=["countdown", "math", "mixed", "math_dataset"])
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--budget-s", type=float, default=0.05, help="Per-episode budget in seconds")
    parser.add_argument("--decision-interval", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="eval_outputs")
    parser.add_argument("--math-split", default="test", help="HF split for MATH dataset")
    parser.add_argument("--math-config", default="", help="HF config for MATH dataset (optional)")
    parser.add_argument("--math-dataset-name", default="qwedsacf/competition_math", help="HF dataset name")
    parser.add_argument(
        "--budget-sweep",
        default="",
        help="Comma-separated budgets in seconds, e.g. '0.005,0.01,0.02,0.05'",
    )
    return parser.parse_args()


def _parse_budget_sweep(value: str) -> list[float]:
    if not value.strip():
        return []
    budgets: list[float] = []
    for part in value.split(","):
        s = part.strip()
        if not s:
            continue
        b = float(s)
        if b <= 0:
            raise ValueError(f"Budget values must be positive, got {b}")
        budgets.append(b)
    return budgets


def main() -> None:
    args = parse_args()
    hardware = get_hardware_specs()[args.hardware]
    model = get_model_configs()[args.model]

    if args.task_set == "math_dataset":
        tasks = generate_tasks_from_math_dataset(
            args.num_tasks,
            split=args.math_split,
            config=args.math_config,
            dataset_name=args.math_dataset_name,
            seed=args.seed,
        )
    else:
        rng = np.random.default_rng(args.seed)
        tasks = generate_tasks(args.task_set, args.num_tasks, rng)
    strategies = get_naive_strategies()
    results = run_eval(
        hardware=hardware,
        model=model,
        tasks=tasks,
        strategies=strategies,
        budget_s=args.budget_s,
        decision_interval=args.decision_interval,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "naive_strategy_results.csv"
    plot_path = output_dir / "naive_strategy_tradeoff.png"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "task_set",
                "mean_quality",
                "mean_latency_ms",
                "mean_budget_overshoot_frac",
                "mean_hbm_pressure",
                "mean_retention",
                "solved_rate",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "strategy": r.strategy,
                    "task_set": r.task_set,
                    "mean_quality": f"{r.mean_quality:.4f}",
                    "mean_latency_ms": f"{r.mean_latency_ms:.4f}",
                    "mean_budget_overshoot_frac": f"{r.mean_budget_overshoot_frac:.4f}",
                    "mean_hbm_pressure": f"{r.mean_hbm_pressure:.4f}",
                    "mean_retention": f"{r.mean_retention:.4f}",
                    "solved_rate": f"{r.solved_rate:.4f}",
                }
            )

    plot_results(results, str(plot_path))

    budgets_s = _parse_budget_sweep(args.budget_sweep)
    if budgets_s:
        sweep = run_budget_sweep(
            hardware=hardware,
            model=model,
            tasks=tasks,
            budgets_s=budgets_s,
            strategies=strategies,
            decision_interval=args.decision_interval,
        )
        sweep_csv_path = output_dir / "budget_sweep_results.csv"
        sweep_plot_path = output_dir / "budget_sweep_quality.png"

        with sweep_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "strategy",
                    "budget_s",
                    "mean_quality",
                    "mean_latency_ms",
                    "solved_rate",
                ],
            )
            writer.writeheader()
            for r in sweep:
                writer.writerow(
                    {
                        "strategy": r.strategy,
                        "budget_s": f"{r.budget_s:.6f}",
                        "mean_quality": f"{r.mean_quality:.4f}",
                        "mean_latency_ms": f"{r.mean_latency_ms:.4f}",
                        "solved_rate": f"{r.solved_rate:.4f}",
                    }
                )

        plot_budget_sweep(sweep, str(sweep_plot_path))
        print(f"Wrote sweep metrics: {sweep_csv_path}")
        print(f"Wrote sweep plot:    {sweep_plot_path}")

    print(f"Wrote metrics: {csv_path}")
    print(f"Wrote plot:    {plot_path}")


if __name__ == "__main__":
    main()
