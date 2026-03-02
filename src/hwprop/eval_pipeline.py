"""Eval-only pipeline for comparing naive KV-cache strategies.

This module intentionally avoids training loops. It runs fixed strategies on
synthetic task traces ("countdown" and "math"), collects cost/quality metrics,
and can plot aggregate tradeoffs.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Callable

import numpy as np

from hwprop.oracle import CostOracle, KVAction
from hwprop.specs import HardwareSpec, ModelConfig


@dataclass(frozen=True)
class EvalTask:
    """Task descriptor for an eval episode.

    ``required_context_frac`` captures how much historical context the task
    depends on: countdown usually lower, multi-step math higher.
    """

    name: str
    prompt_len: int
    decode_steps: int
    required_context_frac: float


@dataclass(frozen=True)
class EvalResult:
    """Aggregate metrics for one strategy across many tasks."""

    strategy: str
    task_set: str
    mean_quality: float
    mean_latency_ms: float
    mean_budget_overshoot_frac: float
    mean_hbm_pressure: float
    mean_retention: float
    solved_rate: float


@dataclass(frozen=True)
class BudgetSweepResult:
    """One strategy result at one budget point."""

    strategy: str
    budget_s: float
    mean_quality: float
    mean_latency_ms: float
    solved_rate: float


StrategyFn = Callable[[CostOracle, EvalTask], KVAction]


def generate_tasks(
    task_set: str,
    num_tasks: int,
    rng: np.random.Generator | None = None,
) -> list[EvalTask]:
    """Generate synthetic eval tasks for ``countdown``, ``math``, or ``mixed``."""
    if rng is None:
        rng = np.random.default_rng(0)

    if task_set not in {"countdown", "math", "mixed"}:
        raise ValueError(f"Unknown task_set: {task_set}")

    tasks: list[EvalTask] = []
    for _ in range(num_tasks):
        if task_set == "mixed":
            kind = "countdown" if rng.random() < 0.5 else "math"
        else:
            kind = task_set

        if kind == "countdown":
            tasks.append(
                EvalTask(
                    name="countdown",
                    prompt_len=int(rng.integers(16, 64)),
                    decode_steps=int(rng.integers(64, 192)),
                    required_context_frac=float(rng.uniform(0.2, 0.4)),
                )
            )
        else:
            tasks.append(
                EvalTask(
                    name="math",
                    prompt_len=int(rng.integers(64, 192)),
                    decode_steps=int(rng.integers(96, 256)),
                    required_context_frac=float(rng.uniform(0.6, 0.85)),
                )
            )
    return tasks


def _estimate_token_len(text: str) -> int:
    words = re.findall(r"\S+", text or "")
    return max(1, len(words))


def _context_frac_from_math_level(level_text: str | None) -> float:
    if not level_text:
        return 0.75
    m = re.search(r"(\d+)", level_text)
    if m is None:
        return 0.75
    level = max(1, min(5, int(m.group(1))))
    # Level 1->5 maps to 0.60->0.84
    return 0.60 + 0.06 * (level - 1)


def _build_math_dataset_tasks_from_records(
    records: list[dict],
    num_tasks: int,
    rng: np.random.Generator,
) -> list[EvalTask]:
    if not records:
        raise ValueError("No records provided from MATH dataset")

    choose_n = min(num_tasks, len(records))
    indices = rng.choice(len(records), size=choose_n, replace=False)
    tasks: list[EvalTask] = []
    for idx in indices:
        row = records[int(idx)]
        problem = str(row.get("problem", ""))
        solution = str(row.get("solution", ""))
        level = row.get("level")

        prompt_tokens = _estimate_token_len(problem)
        solution_tokens = _estimate_token_len(solution)

        # Approximate generation length from reference solution length.
        decode_steps = max(32, min(384, int(solution_tokens * 0.8)))
        required_context = _context_frac_from_math_level(
            str(level) if level is not None else None
        )

        tasks.append(
            EvalTask(
                name="math_dataset",
                prompt_len=prompt_tokens,
                decode_steps=decode_steps,
                required_context_frac=required_context,
            )
        )
    return tasks


def generate_tasks_from_math_dataset(
    num_tasks: int,
    *,
    split: str = "test",
    config: str | None = None,
    dataset_name: str = "qwedsacf/competition_math",
    seed: int = 0,
) -> list[EvalTask]:
    """Generate tasks from the Hugging Face MATH dataset.

    Uses prompt/solution text lengths as token-count proxies for this
    analytical pipeline.
    """
    # Keep HF caches writable in sandboxed/restricted environments.
    hf_home = os.path.join(os.getcwd(), ".hf_home")
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "datasets is required for MATH dataset loading. "
            "Install with `pip install datasets` or `pip install -e \".[eval]\"`."
        ) from exc

    config_val = None if config in (None, "", "none", "None") else config
    try:
        if config_val is None:
            ds = load_dataset(dataset_name, split=split)
        else:
            ds = load_dataset(dataset_name, config_val, split=split)
    except ValueError as exc:
        # Some community datasets expose only "train". If "test" is requested,
        # transparently fall back to "train" to keep the eval pipeline usable.
        if split == "test" and "Unknown split \"test\"" in str(exc):
            if config_val is None:
                ds = load_dataset(dataset_name, split="train")
            else:
                ds = load_dataset(dataset_name, config_val, split="train")
        else:
            raise RuntimeError(
                f"Failed to load MATH dataset '{dataset_name}' (config='{config_val}', split='{split}'). "
                "Check network/auth access to Hugging Face, or pass a local dataset path via "
                "--math-dataset-name."
            ) from exc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to load MATH dataset '{dataset_name}' (config='{config_val}', split='{split}'). "
            "Check network/auth access to Hugging Face, or pass a local dataset path via "
            "--math-dataset-name."
        ) from exc
    records = [dict(ds[i]) for i in range(len(ds))]
    rng = np.random.default_rng(seed)
    return _build_math_dataset_tasks_from_records(records, num_tasks, rng)


def get_naive_strategies() -> dict[str, StrategyFn]:
    """Baseline strategies for quick tradeoff studies."""

    def keep_all(_oracle: CostOracle, _task: EvalTask) -> KVAction:
        return KVAction(keep_frac=1.0, quant_frac=0.0, offload_frac=0.0, disk_frac=0.0)

    def quantize_all(_oracle: CostOracle, _task: EvalTask) -> KVAction:
        return KVAction(keep_frac=0.0, quant_frac=1.0, offload_frac=0.0, disk_frac=0.0)

    def cpu_offload_heavy(_oracle: CostOracle, _task: EvalTask) -> KVAction:
        return KVAction(keep_frac=0.2, quant_frac=0.1, offload_frac=0.7, disk_frac=0.0)

    def disk_offload_heavy(_oracle: CostOracle, _task: EvalTask) -> KVAction:
        return KVAction(keep_frac=0.2, quant_frac=0.1, offload_frac=0.0, disk_frac=0.7)

    def balanced(_oracle: CostOracle, _task: EvalTask) -> KVAction:
        return KVAction(keep_frac=0.5, quant_frac=0.2, offload_frac=0.2, disk_frac=0.1)

    def keep_recent_window(oracle: CostOracle, _task: EvalTask) -> KVAction:
        # Keep roughly a 256-token window in full precision as seq grows.
        active = max(1, oracle.kv_state.active_tokens)
        keep_frac = min(1.0, 256.0 / active)
        # Put most of the remainder in CPU, some quantized in HBM.
        rem = max(0.0, 1.0 - keep_frac)
        quant = 0.2 * rem
        cpu = 0.8 * rem
        return KVAction(keep_frac=keep_frac, quant_frac=quant, offload_frac=cpu, disk_frac=0.0)

    return {
        "keep_all": keep_all,
        "quantize_all": quantize_all,
        "cpu_offload_heavy": cpu_offload_heavy,
        "disk_offload_heavy": disk_offload_heavy,
        "balanced": balanced,
        "keep_recent_window": keep_recent_window,
    }


def _step_quality_proxy(info, oracle: CostOracle, task: EvalTask) -> float:
    """Quality proxy from tier placement + budget pressure.

    This is a modeling proxy for eval-only comparisons (not real model accuracy).
    """
    st = oracle.kv_state
    weighted_context = (
        1.00 * st.tokens_in_hbm
        + 0.92 * st.tokens_in_hbm_quantized
        + 0.70 * st.tokens_in_cpu
        + 0.35 * st.tokens_on_disk
    )
    required = max(1.0, task.required_context_frac * max(1, st.seq_len))
    context_score = min(1.0, weighted_context / required)
    budget_penalty = min(0.5, info.budget_overshoot_frac)
    return max(0.0, context_score - budget_penalty)


def run_eval(
    hardware: HardwareSpec,
    model: ModelConfig,
    tasks: list[EvalTask],
    strategies: dict[str, StrategyFn] | None = None,
    *,
    budget_s: float = 0.05,
    decision_interval: int = 32,
    batch_size: int = 1,
) -> list[EvalResult]:
    """Run eval-only episodes and return aggregate metrics per strategy."""
    if not tasks:
        raise ValueError("tasks must be non-empty")

    policy_map = strategies if strategies is not None else get_naive_strategies()
    results: list[EvalResult] = []

    for strategy_name, policy in policy_map.items():
        qualities: list[float] = []
        latencies_ms: list[float] = []
        overshoots: list[float] = []
        hbm_pressures: list[float] = []
        retentions: list[float] = []
        solved = 0

        for task in tasks:
            oracle = CostOracle(
                hardware=hardware,
                model=model,
                budget_s=budget_s,
                max_seq_len=task.prompt_len + task.decode_steps,
                decision_interval=decision_interval,
                batch_size=batch_size,
            )
            info = oracle.reset(prompt_len=task.prompt_len)

            step_qualities: list[float] = []
            for _ in range(task.decode_steps):
                action = policy(oracle, task) if oracle.is_decision_step else None
                info = oracle.step(action)
                step_qualities.append(_step_quality_proxy(info, oracle, task))

            episode_quality = float(np.mean(step_qualities)) if step_qualities else 0.0
            qualities.append(episode_quality)
            latencies_ms.append((oracle.spent_s / max(1, task.decode_steps)) * 1_000.0)
            overshoots.append(info.budget_overshoot_frac)
            hbm_pressures.append(info.hbm_pressure)
            retentions.append(info.retention)
            solved += int(episode_quality >= 0.8)

        task_set_name = tasks[0].name if len({t.name for t in tasks}) == 1 else "mixed"
        results.append(
            EvalResult(
                strategy=strategy_name,
                task_set=task_set_name,
                mean_quality=float(np.mean(qualities)),
                mean_latency_ms=float(np.mean(latencies_ms)),
                mean_budget_overshoot_frac=float(np.mean(overshoots)),
                mean_hbm_pressure=float(np.mean(hbm_pressures)),
                mean_retention=float(np.mean(retentions)),
                solved_rate=float(solved / len(tasks)),
            )
        )

    return results


def run_budget_sweep(
    hardware: HardwareSpec,
    model: ModelConfig,
    tasks: list[EvalTask],
    budgets_s: list[float],
    strategies: dict[str, StrategyFn] | None = None,
    *,
    decision_interval: int = 32,
    batch_size: int = 1,
) -> list[BudgetSweepResult]:
    """Run eval across multiple budgets for each strategy."""
    if not budgets_s:
        raise ValueError("budgets_s must be non-empty")
    out: list[BudgetSweepResult] = []
    for budget in budgets_s:
        results = run_eval(
            hardware=hardware,
            model=model,
            tasks=tasks,
            strategies=strategies,
            budget_s=float(budget),
            decision_interval=decision_interval,
            batch_size=batch_size,
        )
        for r in results:
            out.append(
                BudgetSweepResult(
                    strategy=r.strategy,
                    budget_s=float(budget),
                    mean_quality=r.mean_quality,
                    mean_latency_ms=r.mean_latency_ms,
                    solved_rate=r.solved_rate,
                )
            )
    return out


def plot_results(results: list[EvalResult], output_path: str) -> None:
    """Save a quality-vs-latency plot for strategy comparison."""
    try:
        # Keep matplotlib usable in sandboxed/non-GUI environments.
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting. Install with `pip install matplotlib`."
        ) from exc

    if not results:
        raise ValueError("results must be non-empty")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    xs = np.array([r.mean_latency_ms for r in results], dtype=float)
    ys = np.array([r.mean_quality for r in results], dtype=float)
    x_span = float(xs.max() - xs.min()) if len(xs) > 1 else 1.0
    y_span = float(ys.max() - ys.min()) if len(ys) > 1 else 1.0
    crowd_x = max(1e-4, 0.015 * x_span)
    crowd_y = max(1e-4, 0.015 * y_span)
    offset_x = max(2e-4, 0.02 * x_span)
    offset_y = max(2e-4, 0.02 * y_span)

    used = set()
    clusters: list[list[int]] = []
    for i in range(len(results)):
        if i in used:
            continue
        cluster = [i]
        used.add(i)
        for j in range(i + 1, len(results)):
            if j in used:
                continue
            if abs(xs[j] - xs[i]) <= crowd_x and abs(ys[j] - ys[i]) <= crowd_y:
                cluster.append(j)
                used.add(j)
        clusters.append(cluster)

    plot_x = xs.copy()
    plot_y = ys.copy()
    for cluster in clusters:
        if len(cluster) == 1:
            continue
        n = len(cluster)
        for rank, idx in enumerate(cluster):
            angle = (2.0 * np.pi * rank) / n
            plot_x[idx] = xs[idx] + np.cos(angle) * offset_x
            plot_y[idx] = ys[idx] + np.sin(angle) * offset_y
            ax.plot(
                [xs[idx], plot_x[idx]],
                [ys[idx], plot_y[idx]],
                linestyle=":",
                linewidth=1.0,
                color="gray",
                alpha=0.55,
            )

    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
    for i, r in enumerate(results):
        marker = markers[i % len(markers)]
        ax.scatter(
            plot_x[i],
            plot_y[i],
            s=85,
            alpha=0.92,
            marker=marker,
            label=r.strategy,
        )

    ax.set_title("Naive Strategy Tradeoff: Quality vs Latency")
    ax.set_xlabel("Mean latency per decode step (ms)")
    ax.set_ylabel("Mean quality (proxy)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_budget_sweep(results: list[BudgetSweepResult], output_path: str) -> None:
    """Save quality-vs-budget curves for each strategy."""
    try:
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting. Install with `pip install matplotlib`."
        ) from exc

    if not results:
        raise ValueError("results must be non-empty")

    grouped: dict[str, list[BudgetSweepResult]] = {}
    for r in results:
        grouped.setdefault(r.strategy, []).append(r)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for strategy, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda x: x.budget_s)
        x = [r.budget_s for r in rows_sorted]
        y = [r.mean_quality for r in rows_sorted]
        ax.plot(x, y, marker="o", linewidth=1.8, markersize=4, label=strategy)

    ax.set_xscale("log")
    ax.set_title("Budget Sweep: Mean Quality vs Budget")
    ax.set_xlabel("Budget per episode (seconds, log scale)")
    ax.set_ylabel("Mean quality (proxy)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
