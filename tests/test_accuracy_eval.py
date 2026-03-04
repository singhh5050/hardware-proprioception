"""Tests for hwprop.accuracy_eval — answer extraction, scoring, strategy registry."""

import pytest

from hwprop.accuracy_eval import (
    AccuracyResult,
    MathTask,
    StrategyConfig,
    extract_boxed_answer,
    get_strategies,
    is_correct,
    normalize_math_answer,
)


# ---------------------------------------------------------------------------
# extract_boxed_answer
# ---------------------------------------------------------------------------
class TestExtractBoxedAnswer:
    def test_simple_integer(self):
        assert extract_boxed_answer(r"The answer is $\boxed{42}$.") == "42"

    def test_fraction(self):
        assert extract_boxed_answer(r"So $\boxed{\frac{1}{2}}$") == r"\frac{1}{2}"

    def test_nested_braces(self):
        text = r"We get $\boxed{\frac{a+b}{c+d}}$"
        assert extract_boxed_answer(text) == r"\frac{a+b}{c+d}"

    def test_multiple_boxed_takes_last(self):
        text = r"First $\boxed{3}$ then $\boxed{7}$"
        assert extract_boxed_answer(text) == "7"

    def test_no_boxed(self):
        assert extract_boxed_answer("No boxed answer here") is None

    def test_empty_boxed(self):
        assert extract_boxed_answer(r"\boxed{}") == ""

    def test_deeply_nested(self):
        text = r"\boxed{\sqrt{\frac{a^{2}+b^{2}}{c}}}"
        result = extract_boxed_answer(text)
        assert result == r"\sqrt{\frac{a^{2}+b^{2}}{c}}"

    def test_unclosed_brace(self):
        assert extract_boxed_answer(r"\boxed{42") is None


# ---------------------------------------------------------------------------
# normalize_math_answer
# ---------------------------------------------------------------------------
class TestNormalizeMathAnswer:
    def test_strips_whitespace(self):
        assert normalize_math_answer("  42  ") == "42"

    def test_removes_text_wrapper(self):
        assert normalize_math_answer(r"\text{cm}") == "cm"

    def test_removes_mathrm_wrapper(self):
        assert normalize_math_answer(r"\mathrm{kg}") == "kg"

    def test_removes_left_right(self):
        result = normalize_math_answer(r"\left(\frac{1}{2}\right)")
        assert r"\left" not in result
        assert r"\right" not in result

    def test_normalizes_internal_whitespace(self):
        assert normalize_math_answer("a  +   b") == "a + b"

    def test_removes_trailing_period(self):
        assert normalize_math_answer("42.") == "42"

    def test_removes_dollar_signs(self):
        assert normalize_math_answer("$x^2$") == "x^2"


# ---------------------------------------------------------------------------
# is_correct
# ---------------------------------------------------------------------------
class TestIsCorrect:
    def test_correct_simple(self):
        assert is_correct(r"The answer is $\boxed{42}$.", "42") is True

    def test_incorrect(self):
        assert is_correct(r"The answer is $\boxed{43}$.", "42") is False

    def test_no_boxed_answer(self):
        assert is_correct("The answer is 42.", "42") is False

    def test_whitespace_tolerance(self):
        assert is_correct(r"\boxed{ 42 }", "42") is True

    def test_fraction_match(self):
        assert is_correct(r"\boxed{\frac{1}{2}}", r"\frac{1}{2}") is True

    def test_text_wrapper_match(self):
        assert is_correct(r"\boxed{\text{cm}}", "cm") is True


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------
class TestStrategyRegistry:
    def test_has_12_strategies(self):
        strategies = get_strategies()
        assert len(strategies) == 12

    def test_strategy_names(self):
        strategies = get_strategies()
        expected = {
            "full_cache", "full_cache_int4",
            "window_128", "window_256", "window_512", "window_1024",
            "h2o_128", "h2o_256", "h2o_512", "h2o_1024",
            "snapkv_512", "expected_attn_512",
        }
        assert set(strategies.keys()) == expected

    def test_full_cache_has_no_press(self):
        s = get_strategies()["full_cache"]
        assert s.press_factory is None
        assert s.budget_tokens is None

    def test_full_cache_int4_is_quantized(self):
        s = get_strategies()["full_cache_int4"]
        assert s.press_factory is None
        assert s.quantized is True

    def test_window_strategies_have_press(self):
        strategies = get_strategies()
        for name in ["window_128", "window_256", "window_512", "window_1024"]:
            s = strategies[name]
            assert s.press_factory is not None, f"{name} should have press_factory"
            assert s.budget_tokens is not None, f"{name} should have budget_tokens"

    def test_h2o_strategies_have_press(self):
        strategies = get_strategies()
        for name in ["h2o_128", "h2o_256", "h2o_512", "h2o_1024"]:
            s = strategies[name]
            assert s.press_factory is not None, f"{name} should have press_factory"

    def test_window_budget_tokens(self):
        strategies = get_strategies()
        assert strategies["window_128"].budget_tokens == 132   # 128 + 4 sinks
        assert strategies["window_256"].budget_tokens == 260   # 256 + 4 sinks
        assert strategies["window_512"].budget_tokens == 516   # 512 + 4 sinks
        assert strategies["window_1024"].budget_tokens == 1028  # 1024 + 4 sinks

    def test_h2o_budget_tokens(self):
        strategies = get_strategies()
        assert strategies["h2o_128"].budget_tokens == 128
        assert strategies["h2o_256"].budget_tokens == 256
        assert strategies["h2o_512"].budget_tokens == 512
        assert strategies["h2o_1024"].budget_tokens == 1024

    def test_custom_decision_interval(self):
        strategies = get_strategies(decision_interval=32)
        assert len(strategies) == 12


# ---------------------------------------------------------------------------
# AccuracyResult serialization
# ---------------------------------------------------------------------------
class TestAccuracyResult:
    def test_to_dict_roundtrip(self):
        r = AccuracyResult(
            strategy_name="full_cache",
            task_id="math_0001",
            generated_text="The answer is \\boxed{42}.",
            extracted_answer="42",
            ground_truth_answer="42",
            correct=True,
            tokens_generated=50,
            prompt_tokens=100,
            cache_size_at_end=150,
            peak_memory_mb=1024.0,
        )
        d = r.to_dict()
        r2 = AccuracyResult.from_dict(d)
        assert r == r2

    def test_from_dict_ignores_extra_keys(self):
        d = {
            "strategy_name": "full_cache",
            "task_id": "math_0001",
            "generated_text": "test",
            "extracted_answer": None,
            "ground_truth_answer": "42",
            "correct": False,
            "tokens_generated": 10,
            "prompt_tokens": 50,
            "cache_size_at_end": 60,
            "peak_memory_mb": 512.0,
            "extra_key": "should be ignored",
        }
        r = AccuracyResult.from_dict(d)
        assert r.strategy_name == "full_cache"


# ---------------------------------------------------------------------------
# MathTask
# ---------------------------------------------------------------------------
class TestMathTask:
    def test_frozen(self):
        t = MathTask(
            task_id="math_0001",
            problem="What is 2+2?",
            ground_truth_solution="$2+2=4$",
            ground_truth_answer="4",
        )
        with pytest.raises(AttributeError):
            t.problem = "changed"


# ---------------------------------------------------------------------------
# Latency replay helpers (from eval_pipeline)
# ---------------------------------------------------------------------------
class TestStrategyToKvUpdate:
    def test_full_cache_keeps_all(self):
        from hwprop.eval_pipeline import strategy_to_kv_update
        result = strategy_to_kv_update("full_cache", None, 500)
        assert result["tokens_kept"] == 500
        assert result["tokens_evicted"] == 0
        assert result["is_quantized"] is False

    def test_full_cache_int4_marks_quantized(self):
        from hwprop.eval_pipeline import strategy_to_kv_update
        result = strategy_to_kv_update("full_cache_int4", None, 500, quantized=True)
        assert result["tokens_kept"] == 500
        assert result["is_quantized"] is True

    def test_window_evicts_over_budget(self):
        from hwprop.eval_pipeline import strategy_to_kv_update
        result = strategy_to_kv_update("window_512", 516, 1000)
        assert result["tokens_kept"] == 516
        assert result["tokens_evicted"] == 484

    def test_window_no_eviction_under_budget(self):
        from hwprop.eval_pipeline import strategy_to_kv_update
        result = strategy_to_kv_update("window_512", 516, 200)
        assert result["tokens_kept"] == 200
        assert result["tokens_evicted"] == 0


class TestComputeStrategyLatency:
    def test_returns_expected_keys(self, h100, llama_8b):
        from hwprop.eval_pipeline import compute_strategy_latency
        result = compute_strategy_latency(
            strategy_name="full_cache",
            budget_tokens=None,
            hardware=h100,
            model_config=llama_8b,
            prompt_len=64,
            decode_steps=64,
        )
        assert "strategy" in result
        assert "mean_latency_ms" in result
        assert "total_time_s" in result
        assert "prefill_time_s" in result
        assert result["mean_latency_ms"] > 0

    def test_window_faster_than_full_cache(self, h100, llama_8b):
        from hwprop.eval_pipeline import compute_strategy_latency
        full = compute_strategy_latency(
            "full_cache", None, h100, llama_8b, 256, 512
        )
        window = compute_strategy_latency(
            "window_128", 132, h100, llama_8b, 256, 512
        )
        # Window should have lower or equal mean latency (less KV to read)
        assert window["mean_latency_ms"] <= full["mean_latency_ms"]


class TestComputeLatencySweep:
    def test_sweep_produces_results(self, h100, llama_8b):
        from hwprop.eval_pipeline import compute_latency_sweep
        strategies = [
            {"strategy": "full_cache", "budget_tokens": None},
            {"strategy": "window_512", "budget_tokens": 516},
        ]
        hw = {"H100_SXM": h100}
        results = compute_latency_sweep(
            strategies, hw, llama_8b,
            prompt_len=64, decode_steps=64,
            offload_splits=[(1.0, 0.0, 0.0)],
        )
        assert len(results) == 2
        assert all("mean_latency_ms" in r for r in results)
