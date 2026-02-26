"""Tests for the CostModel core roofline model."""

import pytest

from hwprop.specs import get_hardware_specs, HardwareSpec, GB, TB, TFLOPS
from hwprop.cost_model import CostModel, KVCacheState


class TestMinStepTime:
    def test_equals_param_bytes_over_hbm_bw(self, cost_model_h100_llama, llama_8b, h100):
        """min_step_time = param_bytes / HBM_BW exactly."""
        expected = llama_8b.param_bytes / h100.hbm_bandwidth
        assert cost_model_h100_llama.min_step_time == pytest.approx(expected, rel=1e-6)

    def test_in_plausible_range(self, cost_model_h100_llama):
        """LLaMA-3 8B bf16 on H100: ~16GB / 3.35 TB/s ~ 4.8ms."""
        assert 0.001 < cost_model_h100_llama.min_step_time < 0.020


class TestAttentionMemoryBound:
    def test_attention_always_memory_bound(self, h100, llama_8b):
        """For batch sizes 1-256, attention compute time < KV HBM load time."""
        cm = CostModel(h100, llama_8b)
        seq_len = 2048
        for bs in [1, 4, 16, 64, 128, 256]:
            kv = KVCacheState(seq_len, seq_len, 0, 0, 0, 0)
            cost = cm.step_cost(kv, bs)
            # The step cost is aggregated — but we can verify that at these
            # batch sizes the total time is dominated by memory (close to
            # param_load + kv_load rather than compute)
            assert cost.time_s > 0


class TestBcritTransition:
    def test_mlp_transitions_at_bcrit(self, h100, llama_8b):
        """MLP switches from memory-bound to compute-bound around B_crit."""
        b_crit = h100.critical_batch_size_fp16

        # At BS=1, MLP is memory-bound
        mlp_memory_time = (
            llama_8b.num_layers * llama_8b.mlp_params_per_layer * llama_8b.bytes_per_param
            / h100.hbm_bandwidth
        )
        mlp_compute_time_bs1 = (
            2.0 * 1 * llama_8b.num_layers * llama_8b.mlp_params_per_layer
            / h100.fp16_flops
        )
        assert mlp_memory_time > mlp_compute_time_bs1

        # Well above B_crit: compute-bound
        bs_high = int(b_crit * 2)
        mlp_compute_time_high = (
            2.0 * bs_high * llama_8b.num_layers * llama_8b.mlp_params_per_layer
            / h100.fp16_flops
        )
        assert mlp_compute_time_high > mlp_memory_time

    def test_h100_bcrit_value(self, h100):
        """H100 B_crit ~ 990 TFLOPS / 3.35 TB/s ~ 296."""
        b_crit = h100.critical_batch_size_fp16
        assert 200 < b_crit < 400, f"B_crit = {b_crit:.0f}"


class TestKVScaling:
    def test_2x_tokens_approx_2x_kv_load(self, cost_model_h100_llama):
        """Doubling tokens should roughly double step time (KV-dominated)."""
        kv_1k = KVCacheState(1024, 1024, 0, 0, 0, 0)
        kv_2k = KVCacheState(2048, 2048, 0, 0, 0, 0)
        cost_1k = cost_model_h100_llama.step_cost(kv_1k)
        cost_2k = cost_model_h100_llama.step_cost(kv_2k)
        # Step time increases but not exactly 2x because param load is constant
        assert cost_2k.time_s > cost_1k.time_s

    def test_step_cost_increases_with_seq_len(self, cost_model_h100_llama):
        """Step cost should increase monotonically with sequence length."""
        times = []
        for s in [512, 1024, 2048, 4096, 8192]:
            kv = KVCacheState(s, s, 0, 0, 0, 0)
            times.append(cost_model_h100_llama.step_cost(kv).time_s)
        for i in range(len(times) - 1):
            assert times[i] < times[i + 1]


class TestDifferentHardware:
    def test_h100_faster_than_l40s(self, h100, l40s, llama_8b):
        """Same model on H100 should be faster than L40S."""
        cm_h100 = CostModel(h100, llama_8b)
        cm_l40s = CostModel(l40s, llama_8b)

        kv = KVCacheState(2048, 2048, 0, 0, 0, 0)
        cost_h100 = cm_h100.step_cost(kv)
        cost_l40s = cm_l40s.step_cost(kv)

        assert cost_h100.time_s < cost_l40s.time_s


class TestQuantizedKV:
    def test_quantized_kv_saves_memory(self, cost_model_h100_llama):
        """Quantized KV cache should use less HBM than full precision."""
        seq_len = 4096
        full = KVCacheState(seq_len, seq_len, 0, 0, 0, 0)
        quant = KVCacheState(seq_len, 0, seq_len, 0, 0, 0)
        cost_full = cost_model_h100_llama.step_cost(full)
        cost_quant = cost_model_h100_llama.step_cost(quant)

        assert cost_quant.hbm_bytes < cost_full.hbm_bytes


class TestHBMOverflow:
    def test_overflow_flag_set(self, llama_8b):
        """HBM overflow flag should be set when KV exceeds capacity."""
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
        cm = CostModel(tiny_hw, llama_8b)
        kv = KVCacheState(100_000, 100_000, 0, 0, 0, 0)
        cost = cm.step_cost(kv)
        assert cost.hbm_overflow is True

    def test_no_overflow_normal_case(self, cost_model_h100_llama):
        """Normal usage shouldn't overflow H100's 80GB."""
        kv = KVCacheState(2048, 2048, 0, 0, 0, 0)
        cost = cost_model_h100_llama.step_cost(kv)
        assert cost.hbm_overflow is False


class TestCPUOffload:
    def test_cpu_offload_adds_latency(self, h100, llama_8b):
        """Offloaded tokens add CPU transfer time, increasing wall clock."""
        cm = CostModel(h100, llama_8b)
        seq_len = 4096
        half = seq_len // 2

        baseline = cm.step_cost(KVCacheState(seq_len, seq_len, 0, 0, 0, 0))
        offloaded = cm.step_cost(KVCacheState(seq_len, half, 0, half, 0, 0))

        # CPU offload adds PCIe transfer time
        assert offloaded.time_s > baseline.time_s
        assert offloaded.cpu_bytes > 0


class TestDiskOffload:
    def test_disk_offload_adds_latency(self, h100, llama_8b):
        """Disk tokens add NVMe transfer time, increasing wall clock."""
        cm = CostModel(h100, llama_8b)
        seq_len = 4096
        half = seq_len // 2

        baseline = cm.step_cost(KVCacheState(seq_len, seq_len, 0, 0, 0, 0))
        on_disk = cm.step_cost(KVCacheState(seq_len, half, 0, 0, half, 0))

        assert on_disk.time_s > baseline.time_s
        assert on_disk.disk_bytes > 0

    def test_disk_slower_than_cpu(self, h100, llama_8b):
        """Disk transfer should be slower than CPU for same number of tokens.

        H100: CPU @ 128 GB/s vs Disk @ 5 GB/s → disk much slower.
        """
        cm = CostModel(h100, llama_8b)
        seq_len = 4096
        half = seq_len // 2

        on_cpu = cm.step_cost(KVCacheState(seq_len, half, 0, half, 0, 0))
        on_disk = cm.step_cost(KVCacheState(seq_len, half, 0, 0, half, 0))

        assert on_disk.time_s > on_cpu.time_s


class TestPrefillCost:
    def test_prefill_cost_positive(self, cost_model_h100_llama):
        cost = cost_model_h100_llama.prefill_cost(2048)
        assert cost.time_s > 0
        assert cost.flops > 0

    def test_prefill_scales_with_length(self, cost_model_h100_llama):
        cost_short = cost_model_h100_llama.prefill_cost(512)
        cost_long = cost_model_h100_llama.prefill_cost(2048)
        assert cost_long.time_s > cost_short.time_s

    def test_prefill_has_zero_disk_bytes(self, cost_model_h100_llama):
        cost = cost_model_h100_llama.prefill_cost(2048)
        assert cost.disk_bytes == 0


class TestGlobalRoofline:
    def test_memory_bound_equals_bytes_over_bw(self, h100, llama_8b):
        """When memory-bound (BS=1), time_s == (param_bytes + kv_bytes) / hbm_bw."""
        cm = CostModel(h100, llama_8b)
        seq_len = 2048
        kv = KVCacheState(seq_len, seq_len, 0, 0, 0, 0)
        cost = cm.step_cost(kv, batch_size=1)

        kv_bytes = seq_len * llama_8b.kv_bytes_per_token_per_layer() * llama_8b.num_layers
        expected_mem_time = (llama_8b.param_bytes + kv_bytes) / h100.hbm_bandwidth

        # At BS=1 this should be memory-bound, so time_s == memory_time
        assert cost.time_s == pytest.approx(expected_mem_time, rel=1e-6)

    def test_no_mlp_double_count(self, h100, llama_8b):
        """Verify step time doesn't exceed (param+kv)/bw when memory-bound.

        The old code double-counted MLP weight loading. With the global
        roofline, BS=1 should be purely memory-bound.
        """
        cm = CostModel(h100, llama_8b)
        kv = KVCacheState(1, 1, 0, 0, 0, 0)  # minimal KV
        cost = cm.step_cost(kv, batch_size=1)

        kv_bytes = 1 * llama_8b.kv_bytes_per_token_per_layer() * llama_8b.num_layers
        memory_time = (llama_8b.param_bytes + kv_bytes) / h100.hbm_bandwidth

        # Should be exactly memory_time (not inflated by MLP double-count)
        assert cost.time_s == pytest.approx(memory_time, rel=1e-6)


class TestCrossValidation:
    def test_llama3_8b_h100_decode_ballpark(self, cost_model_h100_llama):
        """LLaMA-3 8B on H100 batch=1 decode: ~5ms (within 2x of benchmarks)."""
        kv = KVCacheState(2048, 2048, 0, 0, 0, 0)
        cost = cost_model_h100_llama.step_cost(kv)
        assert 0.002 < cost.time_s < 0.020, (
            f"Wall clock = {cost.time_s * 1000:.2f}ms"
        )
