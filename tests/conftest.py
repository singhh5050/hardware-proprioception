"""Shared test fixtures."""

import pytest

from hwprop.specs import get_hardware_specs, get_model_configs
from hwprop.cost_model import CostModel, KVCacheState
from hwprop.oracle import CostOracle


@pytest.fixture
def h100():
    return get_hardware_specs()["H100_SXM"]


@pytest.fixture
def a100():
    return get_hardware_specs()["A100_80GB"]


@pytest.fixture
def l40s():
    return get_hardware_specs()["L40S"]


@pytest.fixture
def mi300x():
    return get_hardware_specs()["MI300X"]


@pytest.fixture
def m4_max():
    return get_hardware_specs()["M4_Max"]


@pytest.fixture
def tiny_1b():
    return get_model_configs()["Tiny-1B"]


@pytest.fixture
def llama_8b():
    return get_model_configs()["LLaMA-3.1-8B"]


@pytest.fixture
def cost_model_h100_llama(h100, llama_8b):
    return CostModel(h100, llama_8b)


@pytest.fixture
def oracle_h100_llama(h100, llama_8b):
    return CostOracle(h100, llama_8b, budget_s=0.1)
