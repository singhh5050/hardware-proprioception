"""hwprop — Analytical roofline-based cost model for LLM inference on diverse hardware."""

__version__ = "0.1.0"

from hwprop.specs import HardwareSpec, ModelConfig, get_hardware_specs, get_model_configs
from hwprop.decisions import KVDecision, StepCost
from hwprop.simulator import HardwareSimulator
from hwprop.sampling import sample_synthetic_hardware

__all__ = [
    "HardwareSpec",
    "ModelConfig",
    "get_hardware_specs",
    "get_model_configs",
    "KVDecision",
    "StepCost",
    "HardwareSimulator",
    "sample_synthetic_hardware",
]
