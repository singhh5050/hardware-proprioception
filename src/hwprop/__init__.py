"""hwprop — Analytical roofline-based cost oracle for LLM inference on diverse hardware."""

__version__ = "0.4.0"

from hwprop.specs import MemoryTier, HardwareSpec, ModelConfig, get_hardware_specs, get_model_configs
from hwprop.cost_model import CostModel, StepCost, KVCacheState
from hwprop.oracle import CostOracle, CostInfo, KVAction
from hwprop.sampling import sample_synthetic_hardware
from hwprop.eval_pipeline import (
    strategy_to_kv_update,
    compute_strategy_latency,
    compute_latency_sweep,
)
from hwprop.overhead import OverheadProfile, OVERHEAD_H100_FLASH2, OVERHEAD_A100_SDPA
from hwprop.strategy import KVCacheStrategy, EvictionEngine, STRATEGY_REGISTRY, get_strategy
from hwprop.simulator import LLMSimulator, SimStepCost, SimResult, simulate_latency

__all__ = [
    # specs
    "MemoryTier",
    "HardwareSpec",
    "ModelConfig",
    "get_hardware_specs",
    "get_model_configs",
    # cost model
    "CostModel",
    "StepCost",
    "KVCacheState",
    # oracle
    "CostOracle",
    "CostInfo",
    "KVAction",
    # sampling
    "sample_synthetic_hardware",
    # eval pipeline (backward compat)
    "strategy_to_kv_update",
    "compute_strategy_latency",
    "compute_latency_sweep",
    # overhead
    "OverheadProfile",
    "OVERHEAD_H100_FLASH2",
    "OVERHEAD_A100_SDPA",
    # strategy
    "KVCacheStrategy",
    "EvictionEngine",
    "STRATEGY_REGISTRY",
    "get_strategy",
    # simulator
    "LLMSimulator",
    "SimStepCost",
    "SimResult",
    "simulate_latency",
]
