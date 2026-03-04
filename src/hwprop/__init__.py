"""hwprop — Analytical roofline-based cost oracle for LLM inference on diverse hardware."""

__version__ = "0.3.0"

from hwprop.specs import MemoryTier, HardwareSpec, ModelConfig, get_hardware_specs, get_model_configs
from hwprop.cost_model import CostModel, StepCost, KVCacheState
from hwprop.oracle import CostOracle, CostInfo, KVAction
from hwprop.sampling import sample_synthetic_hardware
from hwprop.eval_pipeline import (
    EvalTask,
    EvalResult,
    BudgetSweepResult,
    generate_tasks,
    generate_tasks_from_math_dataset,
    get_naive_strategies,
    run_eval,
    run_budget_sweep,
    plot_results,
    plot_budget_sweep,
    strategy_to_kv_update,
    compute_strategy_latency,
    compute_latency_sweep,
)

__all__ = [
    "MemoryTier",
    "HardwareSpec",
    "ModelConfig",
    "get_hardware_specs",
    "get_model_configs",
    "CostModel",
    "StepCost",
    "KVCacheState",
    "CostOracle",
    "CostInfo",
    "KVAction",
    "sample_synthetic_hardware",
    "EvalTask",
    "EvalResult",
    "BudgetSweepResult",
    "generate_tasks",
    "generate_tasks_from_math_dataset",
    "get_naive_strategies",
    "run_eval",
    "run_budget_sweep",
    "plot_results",
    "plot_budget_sweep",
    "strategy_to_kv_update",
    "compute_strategy_latency",
    "compute_latency_sweep",
]
