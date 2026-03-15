"""hwprop — Analytical roofline-based cost oracle for LLM inference on diverse hardware."""

__version__ = "0.3.0"

from hwprop.specs import MemoryTier, HardwareSpec, ModelConfig, get_hardware_specs, get_model_configs
from hwprop.cost_model import CostModel, StepCost, KVCacheState
from hwprop.oracle import CostOracle, CostInfo, KVAction
from hwprop.sampling import sample_synthetic_hardware
from hwprop.eval_pipeline import (
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
    "strategy_to_kv_update",
    "compute_strategy_latency",
    "compute_latency_sweep",
]
