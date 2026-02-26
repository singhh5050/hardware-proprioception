"""hwprop — Analytical roofline-based cost oracle for LLM inference on diverse hardware."""

__version__ = "0.2.0"

from hwprop.specs import HardwareSpec, ModelConfig, get_hardware_specs, get_model_configs
from hwprop.cost_model import CostModel, StepCost, KVCacheState
from hwprop.oracle import CostOracle, CostInfo, KVAction
from hwprop.sampling import sample_synthetic_hardware

__all__ = [
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
]
