"""Shared test fixtures."""

import pytest

from hwprop.specs import get_hardware_specs, get_model_configs
from hwprop.simulator import HardwareSimulator


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
def llama3_8b():
    return get_model_configs()["LLaMA-3.1-8B"]


@pytest.fixture
def llama3_70b():
    return get_model_configs()["LLaMA-3.1-70B"]


@pytest.fixture
def sim_h100_llama3(h100, llama3_8b):
    return HardwareSimulator(h100, llama3_8b)


@pytest.fixture
def sim_l40s_llama3(l40s, llama3_8b):
    return HardwareSimulator(l40s, llama3_8b)
