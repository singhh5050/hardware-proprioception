"""Tests for the LLMSimulator, OverheadProfile, and KVCacheStrategy."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from hwprop.specs import HardwareSpec, ModelConfig, get_hardware_specs, get_model_configs, GB, TFLOPS, TB
from hwprop.cost_model import KVCacheState
from hwprop.overhead import OverheadProfile, OVERHEAD_H100_FLASH2, OVERHEAD_A100_SDPA, OVERHEAD_A100_SDPA_64
from hwprop.strategy import KVCacheStrategy, EvictionEngine, STRATEGY_REGISTRY, get_strategy
from hwprop.simulator import LLMSimulator, SimStepCost, SimResult, simulate_latency

HARDWARE = get_hardware_specs()
MODELS = get_model_configs()

H100 = HARDWARE["H100_SXM"]
A100 = HARDWARE["A100_40GB"]
LLAMA_3B = MODELS["LLaMA-3.2-3B"]
QWEN_7B = MODELS["Qwen2.5-7B"]

CONTEXT_SWEEP_CSV = Path(__file__).parent.parent / "results/benchmark/context_sweep_H100_SXM.csv"
BENCHMARK_CSV = Path(__file__).parent.parent / "results/benchmark/benchmark_results.csv"


# ===========================================================================
# OverheadProfile tests
# ===========================================================================

class TestOverheadProfile:
    def test_corrected_time_is_larger_than_roofline(self):
        t = OVERHEAD_H100_FLASH2.corrected_time(0.002, active_tokens=1024)
        assert t > 0.002, "corrected time should exceed roofline"

    def test_corrected_time_grows_with_context(self):
        t1k  = OVERHEAD_H100_FLASH2.corrected_time(0.002, active_tokens=1024)
        t128k = OVERHEAD_H100_FLASH2.corrected_time(0.002, active_tokens=128 * 1024)
        assert t128k > t1k, "longer context should produce higher corrected time"

    def test_overhead_breakdown_sums_to_one(self):
        bd = OVERHEAD_H100_FLASH2.overhead_breakdown(0.002, active_tokens=4096)
        total = sum(bd.values())
        assert abs(total - 1.0) < 1e-6

    def test_for_hardware_returns_valid_profile(self):
        prof = OverheadProfile.for_hardware(H100)
        assert prof.roofline_efficiency > 0
        assert prof.launch_overhead_s > 0
        assert prof.attn_scan_coeff == 0.0   # effective bandwidth model; no scan coeff
        assert prof.kv_bandwidth_alpha > 0   # physics-based bandwidth degradation
        assert prof.alloc_coeff >= 0

    def test_for_hardware_uses_impl_specific_alpha(self):
        """FA2 and SDPA should get different alpha/beta values."""
        prof_fa2  = OverheadProfile.for_hardware(H100, attn_impl="flash_attention_2")
        prof_sdpa = OverheadProfile.for_hardware(H100, attn_impl="sdpa")
        assert prof_fa2.kv_bandwidth_alpha  > prof_sdpa.kv_bandwidth_alpha
        assert prof_fa2.kv_bandwidth_beta   < prof_sdpa.kv_bandwidth_beta

    def test_for_hardware_eviction_score_scales_with_hbm_bw(self):
        """Eviction scoring is memory-bandwidth bound: coeff ∝ 1/hbm_bw."""
        fast_bw = HardwareSpec(
            name="fast_bw", hbm_capacity=80*GB, hbm_bandwidth=4.0*TB,
            cpu_ram_capacity=512*GB, cpu_gpu_bandwidth=128*GB,
            fp16_flops=990*TFLOPS, int8_flops=1979*TFLOPS, fp32_flops=67*TFLOPS,
            sram_capacity=50*(1<<20), interconnect_bandwidth=900*GB,
        )
        slow_bw = HardwareSpec(
            name="slow_bw", hbm_capacity=80*GB, hbm_bandwidth=2.0*TB,
            cpu_ram_capacity=512*GB, cpu_gpu_bandwidth=128*GB,
            fp16_flops=312*TFLOPS, int8_flops=624*TFLOPS, fp32_flops=20*TFLOPS,
            sram_capacity=50*(1<<20), interconnect_bandwidth=600*GB,
        )
        assert (OverheadProfile.for_hardware(slow_bw).eviction_score_coeff >
                OverheadProfile.for_hardware(fast_bw).eviction_score_coeff)

    def test_calibrate_from_synthetic_data(self):
        """calibrate() should recover known parameters from synthetic rows."""
        from hwprop.cost_model import CostModel

        cost_model = CostModel(H100, LLAMA_3B)

        # Build synthetic benchmark rows using a known profile
        target = OverheadProfile(
            name="synthetic",
            roofline_efficiency=0.5,
            launch_overhead_s=0.010,
            attn_scan_coeff=1.0e-4,
            alloc_coeff=3.0e-5,
        )
        rows = []
        for context_len in [1024, 4096, 16384, 65536]:
            kv = KVCacheState(
                seq_len=context_len,
                tokens_in_hbm=context_len,
                tokens_in_hbm_quantized=0,
                tokens_in_cpu=0,
                tokens_on_disk=0,
                tokens_evicted=0,
            )
            raw = cost_model.step_cost(kv)
            t_measured = target.corrected_time(raw.time_s, context_len)
            rows.append({
                "measured_per_token_ms": t_measured * 1000.0,
                "simulated_per_token_ms": raw.time_s * 1000.0,
                "context_length": context_len,
                "sram_capacity": H100.sram_capacity,
            })

        fitted = OverheadProfile.calibrate(rows, name="fitted")
        # Individual coefficients may not be unique (collinearity between t_roofline and N/sram),
        # but predictions should match closely on the training data.
        for row in rows:
            n = row["context_length"]
            t_rf = row["simulated_per_token_ms"] / 1000.0
            t_meas = row["measured_per_token_ms"] / 1000.0
            t_pred = fitted.corrected_time(t_rf, n)
            assert abs(t_pred - t_meas) / t_meas < 0.05, (
                f"Fitted prediction {t_pred*1000:.2f}ms vs measured {t_meas*1000:.2f}ms "
                f"at context={n}"
            )
        # Launch overhead should be reasonably close since it's not collinear
        assert abs(fitted.launch_overhead_s - target.launch_overhead_s) < 0.003

    @pytest.mark.skipif(not CONTEXT_SWEEP_CSV.exists(), reason="benchmark CSV not found")
    def test_calibrated_h100_reduces_error(self):
        """OVERHEAD_H100_FLASH2 should reduce median ratio error vs pure roofline."""
        rows = []
        with open(CONTEXT_SWEEP_CSV) as f:
            for r in csv.DictReader(f):
                rows.append({
                    "measured_per_token_ms": float(r["measured_per_token_ms"]),
                    "simulated_per_token_ms": float(r["simulated_per_token_ms"]),
                    "context_length": int(r["context_length"]),
                    "sram_capacity": H100.sram_capacity,
                })

        from hwprop.cost_model import CostModel
        cost_model = CostModel(H100, LLAMA_3B)

        kv_head_layers = LLAMA_3B.num_kv_heads * LLAMA_3B.num_layers
        corrected_ratios = []
        for row in rows:
            n = row["context_length"]
            kv = KVCacheState(seq_len=n, tokens_in_hbm=n, tokens_in_hbm_quantized=0,
                              tokens_in_cpu=0, tokens_on_disk=0, tokens_evicted=0)
            raw = cost_model.step_cost(kv)
            t_corr = OVERHEAD_H100_FLASH2.corrected_time(raw.time_s, n, kv_head_layers)
            t_meas = row["measured_per_token_ms"] / 1000.0
            corrected_ratios.append(t_meas / t_corr)

        roofline_ratios = [r["measured_per_token_ms"] / r["simulated_per_token_ms"] for r in rows]
        import statistics
        median_corrected = statistics.median(corrected_ratios)
        median_roofline = statistics.median(roofline_ratios)
        # Corrected model should be closer to 1.0 than pure roofline
        assert abs(median_corrected - 1.0) < abs(median_roofline - 1.0), (
            f"Corrected median ratio {median_corrected:.2f} not closer to 1.0 than "
            f"roofline {median_roofline:.2f}"
        )


# ===========================================================================
# KVCacheStrategy tests
# ===========================================================================

class TestKVCacheStrategy:
    def test_full_cache_no_budget(self):
        s = KVCacheStrategy.full_cache()
        assert s.effective_budget(1000) is None

    def test_window_effective_budget(self):
        s = KVCacheStrategy.window(512)
        assert s.effective_budget(1000) == 512

    def test_budget_frac(self):
        s = KVCacheStrategy(name="frac", eviction_policy="window", budget_frac=0.5)
        assert s.effective_budget(1000) == 500

    def test_budget_min_of_tokens_and_frac(self):
        s = KVCacheStrategy(name="both", eviction_policy="window", budget_tokens=300, budget_frac=0.5)
        assert s.effective_budget(1000) == 300  # min(300, 500)
        assert s.effective_budget(400) == 200   # min(300, 200)

    def test_invalid_tier_fracs(self):
        with pytest.raises(ValueError, match="tier fractions"):
            KVCacheStrategy(name="bad", hbm_frac=0.7, cpu_frac=0.7)

    def test_invalid_quant_bits(self):
        with pytest.raises(ValueError, match="quant_bits"):
            KVCacheStrategy(name="bad", quant_bits=16)

    def test_registry_has_all_12_strategies(self):
        expected = {
            "full_cache", "full_cache_int8",
            "window_128", "window_256", "window_512", "window_1024",
            "h2o_128", "h2o_256", "h2o_512", "h2o_1024",
            "snapkv_512", "expected_attn_512",
        }
        assert expected.issubset(STRATEGY_REGISTRY.keys())

    def test_get_strategy_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown strategy"):
            get_strategy("nonexistent_strategy")


# ===========================================================================
# EvictionEngine tests
# ===========================================================================

class TestEvictionEngine:
    def _make_kv(self, n: int) -> KVCacheState:
        return KVCacheState(
            seq_len=n, tokens_in_hbm=n,
            tokens_in_hbm_quantized=0, tokens_in_cpu=0,
            tokens_on_disk=0, tokens_evicted=0,
        )

    def test_full_cache_no_eviction(self):
        kv = self._make_kv(1000)
        new_kv = EvictionEngine.apply(kv, KVCacheStrategy.full_cache())
        assert new_kv.active_tokens == 1000
        assert new_kv.tokens_evicted == 0

    def test_window_eviction(self):
        kv = self._make_kv(2000)
        strategy = KVCacheStrategy.window(512)
        new_kv = EvictionEngine.apply(kv, strategy)
        # sinks (4) + window_size (512) - sinks = 512 tokens retained total
        assert new_kv.active_tokens <= 512
        assert new_kv.tokens_evicted > 0

    def test_window_no_eviction_when_under_budget(self):
        kv = self._make_kv(100)
        strategy = KVCacheStrategy.window(512)
        new_kv = EvictionEngine.apply(kv, strategy)
        assert new_kv.active_tokens == 100
        assert new_kv.tokens_evicted == 0

    def test_quantization_moves_to_quantized_tier(self):
        kv = self._make_kv(500)
        strategy = KVCacheStrategy.full_cache_int8()
        new_kv = EvictionEngine.apply(kv, strategy)
        assert new_kv.tokens_in_hbm == 0
        assert new_kv.tokens_in_hbm_quantized == 500

    def test_cpu_offload_tier_distribution(self):
        kv = self._make_kv(1000)
        strategy = KVCacheStrategy(
            name="offload_test",
            eviction_policy="none",
            hbm_frac=0.5, cpu_frac=0.5,
        )
        new_kv = EvictionEngine.apply(kv, strategy)
        assert new_kv.tokens_in_hbm == 500
        assert new_kv.tokens_in_cpu == 500

    def test_disk_falls_back_to_hbm_without_disk(self):
        kv = self._make_kv(1000)
        strategy = KVCacheStrategy(
            name="disk_test",
            eviction_policy="none",
            hbm_frac=0.5, disk_frac=0.5,
        )
        new_kv = EvictionEngine.apply(kv, strategy, hardware_has_disk=False)
        assert new_kv.tokens_on_disk == 0
        assert new_kv.active_tokens == 1000

    def test_conservation(self):
        """active_tokens + tokens_evicted is conserved."""
        kv = self._make_kv(1000)
        strategy = KVCacheStrategy.window(256)
        new_kv = EvictionEngine.apply(kv, strategy)
        assert new_kv.active_tokens + new_kv.tokens_evicted == kv.active_tokens + kv.tokens_evicted


# ===========================================================================
# LLMSimulator tests
# ===========================================================================

class TestLLMSimulator:
    def test_step_cost_gt_roofline(self):
        sim = LLMSimulator(H100, LLAMA_3B, overhead=OVERHEAD_H100_FLASH2)
        kv = KVCacheState(seq_len=1024, tokens_in_hbm=1024,
                          tokens_in_hbm_quantized=0, tokens_in_cpu=0,
                          tokens_on_disk=0, tokens_evicted=0)
        sc = sim.step_cost(kv)
        assert sc.time_s > sc.roofline_time_s
        assert sc.launch_overhead_s > 0

    def test_step_cost_grows_with_context(self):
        sim = LLMSimulator(H100, LLAMA_3B, overhead=OVERHEAD_H100_FLASH2)
        def cost_at(n):
            kv = KVCacheState(seq_len=n, tokens_in_hbm=n, tokens_in_hbm_quantized=0,
                              tokens_in_cpu=0, tokens_on_disk=0, tokens_evicted=0)
            return sim.step_cost(kv).time_s
        assert cost_at(128 * 1024) > cost_at(1024)

    def test_simulate_sequence_returns_correct_shape(self):
        sim = LLMSimulator(H100, LLAMA_3B, overhead=OVERHEAD_H100_FLASH2)
        result = sim.simulate_sequence(prompt_len=256, decode_steps=64)
        assert len(result.step_costs) == 64
        assert len(result.kv_states) == 64

    def test_simulate_sequence_window_eviction_limits_kv(self):
        strategy = KVCacheStrategy.window(128)
        sim = LLMSimulator(H100, LLAMA_3B, strategy=strategy,
                           overhead=OVERHEAD_H100_FLASH2)
        result = sim.simulate_sequence(prompt_len=512, decode_steps=256)
        # After eviction, active tokens should stay near the window budget
        final_state = result.kv_states[-1]
        # Window is 128, but we add 1 new token each step before eviction kicks in
        assert final_state.active_tokens <= 200  # generous bound

    def test_window_compression_vs_full_cache_latency(self):
        """Compressed strategy should produce lower or equal mean latency than full cache
        at long contexts (where KV dominates)."""
        prof = OVERHEAD_H100_FLASH2
        full = LLMSimulator(H100, LLAMA_3B, overhead=prof,
                            strategy=KVCacheStrategy.full_cache())
        compressed = LLMSimulator(H100, LLAMA_3B, overhead=prof,
                                  strategy=KVCacheStrategy.window(256))
        r_full = full.simulate_sequence(prompt_len=2048, decode_steps=128)
        r_comp = compressed.simulate_sequence(prompt_len=2048, decode_steps=128)
        assert r_comp.mean_per_token_ms <= r_full.mean_per_token_ms * 1.05  # within 5%

    def test_theoretical_hardware(self):
        """Simulator should work with any HardwareSpec, including theoretical ones."""
        future_gpu = HardwareSpec(
            name="future_10x_bw",
            hbm_capacity=80 * GB,
            hbm_bandwidth=33.5 * TB,
            cpu_ram_capacity=512 * GB,
            cpu_gpu_bandwidth=128 * GB,
            fp16_flops=990 * TFLOPS,
            int8_flops=1979 * TFLOPS,
            fp32_flops=67 * TFLOPS,
            sram_capacity=50 * (1 << 20),
            interconnect_bandwidth=900 * GB,
            disk_capacity=2 * TB,
            disk_bandwidth=5 * GB,
        )
        sim = LLMSimulator(future_gpu, LLAMA_3B)
        result = sim.simulate_sequence(prompt_len=256, decode_steps=32)
        assert result.mean_per_token_ms > 0
        assert result.hardware_name == "future_10x_bw"

    def test_higher_bandwidth_faster(self):
        """10x HBM bandwidth GPU should be faster than baseline in roofline-dominated regime."""
        base = LLMSimulator(H100, LLAMA_3B, overhead=OverheadProfile.for_hardware(H100))
        fast_hw = HardwareSpec(
            name="fast_bw",
            hbm_capacity=80 * GB,
            hbm_bandwidth=33.5 * TB,   # 10x H100
            cpu_ram_capacity=512 * GB,
            cpu_gpu_bandwidth=128 * GB,
            fp16_flops=990 * TFLOPS,
            int8_flops=1979 * TFLOPS,
            fp32_flops=67 * TFLOPS,
            sram_capacity=50 * (1 << 20),
            interconnect_bandwidth=900 * GB,
        )
        fast = LLMSimulator(fast_hw, LLAMA_3B, overhead=OverheadProfile.for_hardware(fast_hw))
        r_base = base.simulate_sequence(prompt_len=1024, decode_steps=64)
        r_fast = fast.simulate_sequence(prompt_len=1024, decode_steps=64)
        # Fast should be at least somewhat faster
        assert r_fast.mean_per_token_ms < r_base.mean_per_token_ms


# ===========================================================================
# simulate_latency() convenience function
# ===========================================================================

class TestSimulateLatency:
    def test_string_hardware_and_model(self):
        result = simulate_latency("H100_SXM", "LLaMA-3.2-3B",
                                  prompt_len=128, decode_steps=32)
        assert isinstance(result, SimResult)
        assert result.mean_per_token_ms > 0

    def test_string_strategy(self):
        result = simulate_latency("H100_SXM", "LLaMA-3.2-3B",
                                  strategy="window_512",
                                  prompt_len=256, decode_steps=32)
        assert result.strategy_name == "window_512"

    def test_none_strategy_is_full_cache(self):
        result = simulate_latency("H100_SXM", "LLaMA-3.2-3B",
                                  strategy=None, prompt_len=128, decode_steps=16)
        assert result.strategy_name == "full_cache"

    def test_unknown_hardware_raises(self):
        with pytest.raises(KeyError, match="Unknown hardware"):
            simulate_latency("NonExistentGPU", "LLaMA-3.2-3B")

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError, match="Unknown model"):
            simulate_latency("H100_SXM", "NonExistentModel")

    def test_h100_uses_flash2_profile(self):
        result = simulate_latency("H100_SXM", "LLaMA-3.2-3B",
                                  prompt_len=128, decode_steps=16)
        assert result.overhead_name == OVERHEAD_H100_FLASH2.name

    def test_a100_uses_sdpa_profile(self):
        result_short = simulate_latency("A100_40GB", "Qwen2.5-7B",
                                        prompt_len=256, decode_steps=64)
        assert result_short.overhead_name == OVERHEAD_A100_SDPA_64.name
        result_long = simulate_latency("A100_40GB", "Qwen2.5-7B",
                                       prompt_len=256, decode_steps=128)
        assert result_long.overhead_name == OVERHEAD_A100_SDPA.name

    def test_custom_overhead(self):
        custom = OverheadProfile(
            name="custom_test",
            roofline_efficiency=0.9,
            launch_overhead_s=0.001,
            attn_scan_coeff=1e-5,
            alloc_coeff=1e-6,
        )
        result = simulate_latency("H100_SXM", "LLaMA-3.2-3B",
                                  overhead=custom, prompt_len=64, decode_steps=8)
        assert result.overhead_name == "custom_test"

    @pytest.mark.skipif(not CONTEXT_SWEEP_CSV.exists(), reason="benchmark CSV not found")
    def test_h100_flash2_within_2x_on_benchmark(self):
        """OVERHEAD_H100_FLASH2 should predict within 2x of measured for most context lengths."""
        rows = []
        with open(CONTEXT_SWEEP_CSV) as f:
            for r in csv.DictReader(f):
                rows.append(r)

        within_2x = 0
        for row in rows:
            context_len = int(row["context_length"])
            decode_steps = int(row["decode_steps"])
            measured_ms = float(row["measured_per_token_ms"])

            result = simulate_latency(
                "H100_SXM", "LLaMA-3.2-3B",
                prompt_len=context_len,
                decode_steps=decode_steps,
            )
            ratio = measured_ms / result.mean_per_token_ms
            if 0.5 <= ratio <= 2.0:
                within_2x += 1

        # At least 50% of predictions within 2x (goal: improve over roofline's ~0%)
        assert within_2x >= len(rows) // 2, (
            f"Only {within_2x}/{len(rows)} context lengths within 2x of measured"
        )
