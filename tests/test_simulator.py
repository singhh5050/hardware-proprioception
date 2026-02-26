"""Tests for the HardwareSimulator core roofline model."""

import numpy as np
import pytest

from hwprop.specs import get_hardware_specs, get_model_configs, GB, TB, TFLOPS
from hwprop.decisions import KVDecision
from hwprop.simulator import HardwareSimulator


class TestMinDecodeLatency:
    def test_llama3_8b_on_h100(self, sim_h100_llama3, llama3_8b, h100):
        """LLaMA-3 8B bf16 on H100: ~16GB / 3.35 TB/s ≈ 4.8ms."""
        latency = sim_h100_llama3.min_decode_latency()
        expected = llama3_8b.param_bytes / h100.hbm_bandwidth
        assert latency == pytest.approx(expected, rel=1e-6)
        # Sanity: should be in the ~3-10ms range
        assert 0.001 < latency < 0.020

    def test_is_pure_param_load(self, sim_h100_llama3, llama3_8b, h100):
        """min_decode_latency = param_bytes / HBM_BW exactly."""
        assert sim_h100_llama3.min_decode_latency() == pytest.approx(
            llama3_8b.param_bytes / h100.hbm_bandwidth
        )


class TestAttentionMemoryBound:
    def test_attention_always_memory_bound(self, h100, llama3_8b):
        """For batch sizes 1-256, attention load time > attention compute time."""
        sim = HardwareSimulator(h100, llama3_8b)
        seq_len = 2048
        for bs in [1, 4, 16, 64, 128, 256]:
            decisions = KVDecision.all_in_hbm(llama3_8b.num_layers, seq_len)
            cost = sim.compute_decode_step_cost(decisions, bs, seq_len)
            # Attention KV load time is part of kv_hbm_load_time.
            # Attention compute time should be much less than total HBM time.
            assert cost.attn_compute_time < cost.kv_hbm_load_time, (
                f"BS={bs}: attn_compute={cost.attn_compute_time:.6f} >= "
                f"kv_hbm_load={cost.kv_hbm_load_time:.6f}"
            )


class TestBcritTransition:
    def test_mlp_transitions_at_bcrit(self, h100, llama3_8b):
        """MLP switches from memory-bound to compute-bound around B_crit."""
        sim = HardwareSimulator(h100, llama3_8b)
        b_crit = h100.critical_batch_size_fp16

        # Well below B_crit: memory-bound (mlp_compute ≈ mlp_memory)
        decisions = KVDecision.all_in_hbm(llama3_8b.num_layers, 1)
        cost_low = sim.compute_decode_step_cost(decisions, batch_size=1, original_seq_len=1)
        # At BS=1, MLP is memory-bound: the MLP compute time should equal memory time
        mlp_memory_time = (
            llama3_8b.num_layers * llama3_8b.mlp_params_per_layer * llama3_8b.bytes_per_param
            / h100.hbm_bandwidth
        )
        mlp_compute_time_bs1 = (
            2.0 * 1 * llama3_8b.num_layers * llama3_8b.mlp_params_per_layer
            / h100.fp16_flops
        )
        # At BS=1, memory time should dominate
        assert mlp_memory_time > mlp_compute_time_bs1

        # Well above B_crit: compute-bound
        bs_high = int(b_crit * 2)
        mlp_compute_time_high = (
            2.0 * bs_high * llama3_8b.num_layers * llama3_8b.mlp_params_per_layer
            / h100.fp16_flops
        )
        assert mlp_compute_time_high > mlp_memory_time

    def test_h100_bcrit_value(self, h100):
        """H100 B_crit ≈ 990 TFLOPS / 3.35 TB/s ≈ 296."""
        b_crit = h100.critical_batch_size_fp16
        assert 200 < b_crit < 400, f"B_crit = {b_crit:.0f}"


class TestKVScaling:
    def test_2x_seq_len_approx_2x_kv_load(self, sim_h100_llama3, llama3_8b):
        """Doubling seq_len should roughly double KV load time."""
        cost_1k = sim_h100_llama3.compute_baseline_cost(1024)
        cost_2k = sim_h100_llama3.compute_baseline_cost(2048)
        ratio = cost_2k.kv_hbm_load_time / cost_1k.kv_hbm_load_time
        assert 1.9 < ratio < 2.1, f"Ratio = {ratio:.3f}"

    def test_step_time_increases_with_seq_len(self, sim_h100_llama3):
        """Step time should increase with sequence length."""
        times = [
            sim_h100_llama3.compute_baseline_cost(s).wall_clock_time
            for s in [512, 1024, 2048, 4096, 8192]
        ]
        for i in range(len(times) - 1):
            assert times[i] < times[i + 1]


class TestEviction:
    def test_eviction_reduces_cost(self, sim_h100_llama3, llama3_8b):
        """50% eviction should reduce cost vs baseline."""
        seq_len = 4096
        baseline = sim_h100_llama3.compute_baseline_cost(seq_len)
        evicted = KVDecision.uniform_evict(llama3_8b.num_layers, seq_len, keep_frac=0.5)
        cost = sim_h100_llama3.compute_decode_step_cost(evicted, 1, seq_len)

        assert cost.wall_clock_time < baseline.wall_clock_time
        assert cost.retention_ratio == pytest.approx(0.5, abs=0.01)

    def test_eviction_reduces_memory(self, sim_h100_llama3, llama3_8b):
        """Eviction should reduce HBM usage."""
        seq_len = 4096
        baseline = sim_h100_llama3.compute_baseline_cost(seq_len)
        evicted = KVDecision.uniform_evict(llama3_8b.num_layers, seq_len, keep_frac=0.5)
        cost = sim_h100_llama3.compute_decode_step_cost(evicted, 1, seq_len)

        assert cost.hbm_used < baseline.hbm_used


class TestCPUOffload:
    def test_cpu_offload_slower_than_hbm(self, h100, llama3_8b):
        """CPU offload should be slower than keeping everything in HBM."""
        sim = HardwareSimulator(h100, llama3_8b)
        seq_len = 4096
        L = llama3_8b.num_layers

        baseline = sim.compute_baseline_cost(seq_len)

        # Offload 50% to CPU
        half = seq_len // 2
        offloaded = KVDecision(
            tokens_in_hbm_full=np.full(L, half, dtype=np.int64),
            tokens_in_hbm_quantized=np.zeros(L, dtype=np.int64),
            tokens_in_cpu=np.full(L, half, dtype=np.int64),
            tokens_evicted=np.zeros(L, dtype=np.int64),
        )
        cost = sim.compute_decode_step_cost(offloaded, 1, seq_len)

        # CPU transfer adds latency
        assert cost.kv_cpu_transfer_time > 0
        assert cost.wall_clock_time > baseline.wall_clock_time


class TestQuantization:
    def test_quantized_kv_saves_memory(self, sim_h100_llama3, llama3_8b):
        """Quantized KV cache should use less HBM than full precision."""
        seq_len = 4096
        L = llama3_8b.num_layers

        baseline = sim_h100_llama3.compute_baseline_cost(seq_len)

        # All quantized
        quantized = KVDecision(
            tokens_in_hbm_full=np.zeros(L, dtype=np.int64),
            tokens_in_hbm_quantized=np.full(L, seq_len, dtype=np.int64),
            tokens_in_cpu=np.zeros(L, dtype=np.int64),
            tokens_evicted=np.zeros(L, dtype=np.int64),
        )
        cost = sim_h100_llama3.compute_decode_step_cost(quantized, 1, seq_len)

        assert cost.hbm_used < baseline.hbm_used
        # Quantized = INT8, so roughly half the KV bytes
        kv_ratio = cost.kv_hbm_load_time / baseline.kv_hbm_load_time
        assert 0.4 < kv_ratio < 0.6, f"KV load ratio = {kv_ratio:.3f}"


class TestDifferentHardware:
    def test_h100_faster_than_l40s(self, h100, l40s, llama3_8b):
        """Same model on H100 should be faster than L40S."""
        sim_h100 = HardwareSimulator(h100, llama3_8b)
        sim_l40s = HardwareSimulator(l40s, llama3_8b)

        seq_len = 2048
        cost_h100 = sim_h100.compute_baseline_cost(seq_len)
        cost_l40s = sim_l40s.compute_baseline_cost(seq_len)

        assert cost_h100.wall_clock_time < cost_l40s.wall_clock_time


class TestHBMOverflow:
    def test_overflow_flag_set(self, llama3_8b):
        """HBM overflow flag should be set when KV exceeds capacity."""
        # Use a tiny HBM device
        from hwprop.specs import HardwareSpec, GB, TFLOPS
        tiny_hw = HardwareSpec(
            name="tiny",
            hbm_capacity=1 * GB,
            hbm_bandwidth=100 * GB,
            cpu_ram_capacity=64 * GB,
            cpu_gpu_bandwidth=32 * GB,
            fp16_flops=50 * TFLOPS,
            int8_flops=100 * TFLOPS,
            fp32_flops=10 * TFLOPS,
            sram_capacity=4 * (1 << 20),
            interconnect_bandwidth=0,
        )
        sim = HardwareSimulator(tiny_hw, llama3_8b)
        # Very long sequence to overflow 1 GB HBM
        cost = sim.compute_baseline_cost(seq_len=100_000)
        assert cost.hbm_overflow is True

    def test_no_overflow_normal_case(self, sim_h100_llama3):
        """Normal usage shouldn't overflow H100's 80GB."""
        cost = sim_h100_llama3.compute_baseline_cost(seq_len=2048)
        assert cost.hbm_overflow is False


class TestConvenienceMethods:
    def test_theoretical_min_less_than_baseline(self, sim_h100_llama3):
        """Theoretical min should be <= baseline wall clock."""
        seq_len = 2048
        t_min = sim_h100_llama3.theoretical_min_step_time(seq_len)
        baseline = sim_h100_llama3.compute_baseline_cost(seq_len)
        assert t_min <= baseline.wall_clock_time

    def test_max_throughput_is_inverse(self, sim_h100_llama3):
        seq_len = 2048
        t_min = sim_h100_llama3.theoretical_min_step_time(seq_len)
        max_tps = sim_h100_llama3.theoretical_max_throughput(seq_len)
        assert max_tps == pytest.approx(1.0 / t_min, rel=1e-6)

    def test_prefill_time_positive(self, sim_h100_llama3):
        assert sim_h100_llama3.prefill_time(2048) > 0

    def test_max_batch_size_positive(self, sim_h100_llama3):
        bs = sim_h100_llama3.max_batch_size(2048)
        assert bs >= 1

    def test_general_step_time_close_to_baseline(self, sim_h100_llama3):
        """general_step_time should be in the same ballpark as baseline."""
        seq_len = 2048
        quick = sim_h100_llama3.general_step_time(seq_len)
        baseline = sim_h100_llama3.compute_baseline_cost(seq_len).wall_clock_time
        # Within 2x of each other
        ratio = quick / baseline
        assert 0.5 < ratio < 2.0, f"Ratio = {ratio:.3f}"


class TestCrossValidation:
    def test_llama3_8b_h100_decode_ballpark(self, sim_h100_llama3):
        """LLaMA-3 8B on H100 batch=1 decode: ~5ms (within 2x of known benchmarks)."""
        cost = sim_h100_llama3.compute_baseline_cost(seq_len=2048)
        # vLLM benchmarks and Scaling Book suggest ~5ms range
        assert 0.002 < cost.wall_clock_time < 0.020, (
            f"Wall clock = {cost.wall_clock_time * 1000:.2f}ms"
        )

    def test_llama31_8b_kv_cache_at_8192(self, llama3_8b):
        """LLaMA-3.1-8B KV at 8192 ctx: exactly 1 GiB.

        Per layer: 2 (K+V) * 8 (kv_heads) * 128 (head_dim) * 2 (bf16) = 4096 B/tok
        Total: 4096 * 32 layers = 131072 B/tok = 128 KiB/tok
        At 8192 tokens: 131072 * 8192 = 1 GiB
        """
        kv_bytes = llama3_8b.kv_cache_bytes(8192)
        kv_gb = kv_bytes / (1 << 30)
        assert kv_gb == pytest.approx(1.0, rel=1e-6), f"KV cache = {kv_gb:.4f} GiB"
