"""Tests for the CostOracle RL interface."""

import numpy as np
import pytest

from hwprop.oracle import CostOracle, KVAction, CostInfo


class TestBudgetTracking:
    def test_spent_accumulates(self, oracle_h100_llama):
        oracle_h100_llama.reset(prompt_len=0)
        for _ in range(10):
            oracle_h100_llama.step()
        assert oracle_h100_llama.spent_s > 0

    def test_budget_ok_flips_when_exceeded(self, h100, llama_8b):
        """With a tiny budget, budget_ok should flip to False."""
        oracle = CostOracle(h100, llama_8b, budget_s=1e-6)  # ~1 microsecond
        oracle.reset(prompt_len=128)
        # Prefill alone should exceed 1us budget
        info = oracle.step()
        assert info.budget_ok is False

    def test_within_budget_after_reset(self, h100, llama_8b):
        """Always within budget immediately after reset (no prompt)."""
        oracle = CostOracle(h100, llama_8b, budget_s=1.0)
        oracle.reset(prompt_len=0)
        assert oracle.within_budget is True


class TestKVStateGrows:
    def test_each_step_adds_one_token(self, oracle_h100_llama):
        oracle_h100_llama.reset(prompt_len=100)
        assert oracle_h100_llama.kv_state.seq_len == 100

        for i in range(1, 11):
            oracle_h100_llama.step()
            assert oracle_h100_llama.kv_state.seq_len == 100 + i
            assert oracle_h100_llama.kv_state.tokens_in_hbm == 100 + i


class TestActions:
    def test_action_at_decision_boundary(self, h100, llama_8b):
        """Action at decision boundary should redistribute KV tokens."""
        oracle = CostOracle(h100, llama_8b, budget_s=1.0, decision_interval=4)
        oracle.reset(prompt_len=100)

        # Steps 0-3: first 4 tokens (step 0 is a decision step)
        # Step 0 is decision boundary — apply action to redistribute the 100 prompt tokens
        action = KVAction(keep_frac=0.5, quant_frac=0.3, offload_frac=0.1, disk_frac=0.0)
        info = oracle.step(action)
        assert info.is_decision_step is True

        # After action + new token: should have redistributed + 1 new token in HBM
        kv = oracle.kv_state
        # 100 active tokens redistributed: 50 keep + 30 quant + 10 offload + 10 evict
        # Then +1 new token in HBM
        assert kv.tokens_in_hbm == 51  # round(100*0.5) + 1
        assert kv.tokens_in_hbm_quantized == 30  # round(100*0.3)
        assert kv.tokens_in_cpu == 10  # round(100*0.1)

    def test_action_ignored_off_boundary(self, h100, llama_8b):
        """Action at non-decision step is a no-op."""
        oracle = CostOracle(h100, llama_8b, budget_s=1.0, decision_interval=64)
        oracle.reset(prompt_len=100)

        # Step 0 is decision boundary — skip it with no action
        oracle.step()  # step 0, now step_count=1

        # Step 1 is NOT a decision boundary
        action = KVAction(keep_frac=0.5, quant_frac=0.3, offload_frac=0.1, disk_frac=0.0)
        oracle.step(action)  # action should be ignored

        # All tokens should still be in HBM (100 prompt + 2 decode)
        assert oracle.kv_state.tokens_in_hbm == 102
        assert oracle.kv_state.tokens_in_hbm_quantized == 0
        assert oracle.kv_state.tokens_in_cpu == 0

    def test_eviction_reduces_hbm_pressure(self, h100, llama_8b):
        """Evicting tokens should lower hbm_pressure."""
        oracle = CostOracle(h100, llama_8b, budget_s=1.0, decision_interval=1)
        oracle.reset(prompt_len=1000)

        # Step without action — get baseline pressure
        info_before = oracle.step()

        # Step with aggressive eviction
        action = KVAction(keep_frac=0.3, quant_frac=0.0, offload_frac=0.0, disk_frac=0.0)
        info_after = oracle.step(action)

        assert info_after.hbm_pressure < info_before.hbm_pressure

    def test_rounding_never_negative_eviction(self, h100, llama_8b):
        """Fractions that round up shouldn't corrupt state with negative eviction."""
        oracle = CostOracle(h100, llama_8b, budget_s=1.0, decision_interval=1)
        oracle.reset(prompt_len=3)  # 3 tokens: round(3*0.34)=1, round(3*0.33)=1, round(3*0.33)=1 → sum=3

        # Fractions that sum to 1.0 but round() on 3 tokens gives 1+1+1=3 exactly.
        # Try fractions where round individually overshoots.
        action = KVAction(keep_frac=0.34, quant_frac=0.34, offload_frac=0.34, disk_frac=0.0)
        oracle.step(action)

        kv = oracle.kv_state
        assert kv.tokens_evicted >= 0
        assert kv.tokens_in_hbm >= 0
        assert kv.tokens_in_hbm_quantized >= 0
        assert kv.tokens_in_cpu >= 0
        assert kv.tokens_on_disk >= 0

    def test_rounding_large_fracs_clamped(self, h100, llama_8b):
        """Fractions summing well over 1.0 should still produce valid state."""
        oracle = CostOracle(h100, llama_8b, budget_s=1.0, decision_interval=1)
        oracle.reset(prompt_len=100)

        # Absurd fractions: keep=0.9, quant=0.9, offload=0.9 (sum=2.7)
        # _apply_action uses sequential clamping so it can't go negative
        action = KVAction(keep_frac=0.9, quant_frac=0.9, offload_frac=0.9, disk_frac=0.0)
        oracle.step(action)

        kv = oracle.kv_state
        assert kv.tokens_evicted >= 0
        total = (kv.tokens_in_hbm + kv.tokens_in_hbm_quantized
                 + kv.tokens_in_cpu + kv.tokens_on_disk + kv.tokens_evicted)
        # total should equal seq_len (101 = 100 prompt + 1 decode)
        assert total == kv.seq_len

    def test_disk_tier_in_action(self, h100, llama_8b):
        """Verify disk_frac populates tokens_on_disk."""
        oracle = CostOracle(h100, llama_8b, budget_s=1.0, decision_interval=1)
        oracle.reset(prompt_len=100)

        action = KVAction(keep_frac=0.5, quant_frac=0.1, offload_frac=0.1, disk_frac=0.2)
        oracle.step(action)

        kv = oracle.kv_state
        assert kv.tokens_on_disk == round(100 * 0.2)  # 20
        assert kv.tokens_in_hbm == round(100 * 0.5) + 1  # 50 + 1 new


class TestObservation:
    def test_observation_vector_shape(self, oracle_h100_llama):
        oracle_h100_llama.reset(prompt_len=128)
        for _ in range(10):
            oracle_h100_llama.step()
        obs = oracle_h100_llama.observation()
        assert obs.shape == (4,)
        assert obs.dtype == np.float32

    def test_observation_values_in_range(self, oracle_h100_llama):
        oracle_h100_llama.reset(prompt_len=128)
        for _ in range(10):
            oracle_h100_llama.step()
        obs = oracle_h100_llama.observation()
        # budget_remaining_frac in [0, 1]
        assert 0.0 <= obs[0] <= 1.0
        # hbm_pressure > 0 (at least params loaded)
        assert obs[1] > 0
        # seq_position_frac in [0, 1]
        assert 0.0 <= obs[2] <= 1.0
        # retention in [0, 1]
        assert 0.0 <= obs[3] <= 1.0

    def test_observation_str_format(self, oracle_h100_llama):
        oracle_h100_llama.reset(prompt_len=128)
        for _ in range(10):
            oracle_h100_llama.step()
        s = oracle_h100_llama.observation_str()
        assert "Budget:" in s
        assert "HBM:" in s
        assert "Position:" in s
        assert "Retention:" in s


class TestReset:
    def test_reset_clears_state(self, oracle_h100_llama):
        """Reset restores to initial state."""
        oracle_h100_llama.reset(prompt_len=128)
        for _ in range(50):
            oracle_h100_llama.step()

        # Now reset
        oracle_h100_llama.reset(prompt_len=64)
        assert oracle_h100_llama.kv_state.seq_len == 64
        assert oracle_h100_llama.kv_state.tokens_in_hbm == 64
        assert oracle_h100_llama.kv_state.tokens_in_hbm_quantized == 0
        assert oracle_h100_llama.kv_state.tokens_in_cpu == 0
        assert oracle_h100_llama.kv_state.tokens_on_disk == 0
        assert oracle_h100_llama.kv_state.tokens_evicted == 0
        assert oracle_h100_llama.step_count == 0


class TestCurriculumBudget:
    def test_tighter_budget_overflows_sooner(self, h100, llama_8b):
        """Same generation with smaller budget → over-budget sooner."""
        loose = CostOracle(h100, llama_8b, budget_s=1.0)
        tight = CostOracle(h100, llama_8b, budget_s=0.01)

        loose.reset(prompt_len=128)
        tight.reset(prompt_len=128)

        loose_over = None
        tight_over = None
        for i in range(500):
            info_loose = loose.step()
            info_tight = tight.step()
            if loose_over is None and not info_loose.budget_ok:
                loose_over = i
            if tight_over is None and not info_tight.budget_ok:
                tight_over = i

        # Tight budget should overflow before loose (or loose never overflows)
        assert tight_over is not None, "Tight budget should overflow within 500 steps"
        if loose_over is not None:
            assert tight_over < loose_over


class TestKVActionFromTensor:
    def test_valid_fractions_passthrough(self):
        t = np.array([0.6, 0.2, 0.1, 0.05])
        a = KVAction.from_tensor(t)
        assert a.keep_frac == pytest.approx(0.6)
        assert a.quant_frac == pytest.approx(0.2)
        assert a.offload_frac == pytest.approx(0.1)
        assert a.disk_frac == pytest.approx(0.05)

    def test_normalises_when_sum_exceeds_one(self):
        t = np.array([0.5, 0.5, 0.5, 0.5])  # sum = 2.0
        a = KVAction.from_tensor(t)
        total = a.keep_frac + a.quant_frac + a.offload_frac + a.disk_frac
        assert total == pytest.approx(1.0, abs=1e-6)
        # Proportions preserved (all equal)
        assert a.keep_frac == pytest.approx(a.quant_frac)

    def test_clips_negatives(self):
        t = np.array([-0.5, 0.8, -0.1, 0.0])
        a = KVAction.from_tensor(t)
        assert a.keep_frac == pytest.approx(0.0)
        assert a.quant_frac == pytest.approx(0.8)
        assert a.offload_frac == pytest.approx(0.0)
        assert a.disk_frac == pytest.approx(0.0)

    def test_longer_tensor_uses_first_four(self):
        t = np.array([0.5, 0.3, 0.1, 0.05, 999.0, 888.0])
        a = KVAction.from_tensor(t)
        assert a.keep_frac == pytest.approx(0.5)
        assert a.disk_frac == pytest.approx(0.05)


class TestKVActionFromText:
    def test_standard_format(self):
        a = KVAction.from_text("[KV: keep=0.8 quant=0.1 offload=0.05 disk=0.0]")
        assert a.keep_frac == pytest.approx(0.8)
        assert a.quant_frac == pytest.approx(0.1)
        assert a.offload_frac == pytest.approx(0.05)
        assert a.disk_frac == pytest.approx(0.0)

    def test_with_commas_and_whitespace(self):
        a = KVAction.from_text("keep = 0.7, quant = 0.2, offload = 0.1, disk = 0.0")
        assert a.keep_frac == pytest.approx(0.7)

    def test_normalises_if_over_one(self):
        a = KVAction.from_text("keep=0.6 quant=0.6 offload=0.6 disk=0.6")
        total = a.keep_frac + a.quant_frac + a.offload_frac + a.disk_frac
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_without_disk_defaults_to_zero(self):
        a = KVAction.from_text("[KV: keep=0.8 quant=0.1 offload=0.05]")
        assert a.keep_frac == pytest.approx(0.8)
        assert a.quant_frac == pytest.approx(0.1)
        assert a.offload_frac == pytest.approx(0.05)
        assert a.disk_frac == pytest.approx(0.0)

    def test_invalid_text_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            KVAction.from_text("I think we should keep most of the cache")


class TestBudgetOvershoot:
    def test_zero_when_within_budget(self, h100, llama_8b):
        oracle = CostOracle(h100, llama_8b, budget_s=10.0)
        info = oracle.reset(prompt_len=128)
        assert info.budget_overshoot_frac == pytest.approx(0.0)

    def test_positive_when_over_budget(self, h100, llama_8b):
        oracle = CostOracle(h100, llama_8b, budget_s=1e-6)
        oracle.reset(prompt_len=128)
        info = oracle.step()
        assert info.budget_overshoot_frac > 0.0
        assert info.budget_ok is False

    def test_overshoot_grows_monotonically(self, h100, llama_8b):
        """Once over budget, overshoot should only increase."""
        oracle = CostOracle(h100, llama_8b, budget_s=0.005)
        oracle.reset(prompt_len=128)

        prev_overshoot = 0.0
        went_over = False
        for _ in range(200):
            info = oracle.step()
            if not info.budget_ok:
                went_over = True
            if went_over:
                assert info.budget_overshoot_frac >= prev_overshoot
                prev_overshoot = info.budget_overshoot_frac

        assert went_over, "Should have gone over budget"
