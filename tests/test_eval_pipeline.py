"""Tests for eval-only naive strategy pipeline."""

import numpy as np

from hwprop.eval_pipeline import (
    _build_math_dataset_tasks_from_records,
    _context_frac_from_math_level,
    _estimate_token_len,
    generate_tasks,
    get_naive_strategies,
    run_budget_sweep,
    run_eval,
)
from hwprop.specs import get_hardware_specs, get_model_configs


def test_generate_tasks_count_and_names():
    tasks = generate_tasks("mixed", 12, np.random.default_rng(42))
    assert len(tasks) == 12
    assert {t.name for t in tasks}.issubset({"countdown", "math"})


def test_run_eval_returns_one_result_per_strategy():
    hw = get_hardware_specs()["H100_SXM"]
    model = get_model_configs()["LLaMA-3.1-8B"]
    tasks = generate_tasks("countdown", 6, np.random.default_rng(0))
    results = run_eval(hw, model, tasks, budget_s=0.05, decision_interval=16)
    assert len(results) == len(get_naive_strategies())
    for r in results:
        assert 0.0 <= r.mean_quality <= 1.0
        assert r.mean_latency_ms > 0.0
        assert 0.0 <= r.solved_rate <= 1.0


def test_naive_strategies_produce_nonidentical_profiles():
    hw = get_hardware_specs()["H100_SXM"]
    model = get_model_configs()["LLaMA-3.1-8B"]
    tasks = generate_tasks("math", 8, np.random.default_rng(123))
    results = run_eval(hw, model, tasks, budget_s=0.02, decision_interval=8)
    qualities = {round(r.mean_quality, 5) for r in results}
    latencies = {round(r.mean_latency_ms, 5) for r in results}
    # Baselines should not collapse to one identical point.
    assert len(qualities) > 1
    assert len(latencies) > 1


def test_math_level_to_context_fraction_mapping():
    assert _context_frac_from_math_level("Level 1") == 0.60
    assert _context_frac_from_math_level("Level 5") == 0.84
    assert _context_frac_from_math_level(None) == 0.75


def test_build_tasks_from_math_like_records():
    records = [
        {
            "problem": "Compute 1+1.",
            "solution": "2",
            "level": "Level 1",
        },
        {
            "problem": "Find x if x^2 + 3x + 2 = 0.",
            "solution": "Factor and solve to get x = -1 or -2.",
            "level": "Level 4",
        },
    ]
    tasks = _build_math_dataset_tasks_from_records(records, 2, np.random.default_rng(7))
    assert len(tasks) == 2
    assert {t.name for t in tasks} == {"math_dataset"}
    for t in tasks:
        assert t.prompt_len >= 1
        assert 32 <= t.decode_steps <= 384
        assert 0.60 <= t.required_context_frac <= 0.84


def test_estimate_token_len_nonempty():
    assert _estimate_token_len("a b c") == 3
    assert _estimate_token_len("") == 1


def test_run_budget_sweep_shape():
    hw = get_hardware_specs()["H100_SXM"]
    model = get_model_configs()["LLaMA-3.1-8B"]
    tasks = generate_tasks("countdown", 4, np.random.default_rng(9))
    budgets = [0.01, 0.05, 0.1]
    sweep = run_budget_sweep(hw, model, tasks, budgets, decision_interval=16)
    assert len(sweep) == len(budgets) * len(get_naive_strategies())
    assert {round(r.budget_s, 6) for r in sweep} == {0.01, 0.05, 0.1}
