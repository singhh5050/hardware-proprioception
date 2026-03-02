"""Tests for hardware and model specifications."""

import numpy as np

from hwprop.specs import MemoryTier, HardwareSpec, get_hardware_specs, get_model_configs, GB, TB, TFLOPS


class TestHardwareSpecs:
    def test_all_specs_have_positive_values(self):
        for name, spec in get_hardware_specs().items():
            assert spec.hbm_capacity > 0, f"{name} hbm_capacity"
            assert spec.hbm_bandwidth > 0, f"{name} hbm_bandwidth"
            assert spec.fp16_flops > 0, f"{name} fp16_flops"
            assert spec.int8_flops > 0, f"{name} int8_flops"
            assert spec.fp32_flops > 0, f"{name} fp32_flops"

    def test_plausible_hbm_range(self):
        for name, spec in get_hardware_specs().items():
            assert 4 * GB <= spec.hbm_capacity <= 300 * GB, (
                f"{name} HBM {spec.hbm_capacity / GB:.0f} GB out of range"
            )

    def test_plausible_flops_range(self):
        for name, spec in get_hardware_specs().items():
            assert 1 * TFLOPS <= spec.fp16_flops <= 5000 * TFLOPS, (
                f"{name} FP16 {spec.fp16_flops / TFLOPS:.0f} TFLOPS out of range"
            )

    def test_bcrit_positive(self):
        for name, spec in get_hardware_specs().items():
            assert spec.critical_batch_size_fp16 > 0, f"{name} B_crit"

    def test_to_prompt_string(self):
        spec = get_hardware_specs()["H100_SXM"]
        s = spec.to_prompt_string()
        assert "H100_SXM" in s
        assert "80 GB" in s

    def test_to_prompt_string_includes_disk(self):
        spec = get_hardware_specs()["H100_SXM"]
        s = spec.to_prompt_string()
        assert "Disk" in s
        # Edge device with no disk should not mention Disk
        snap = get_hardware_specs()["Snapdragon_X_Elite"]
        assert "Disk" not in snap.to_prompt_string()

    def test_to_tensor_shape(self):
        spec = get_hardware_specs()["H100_SXM"]
        t = spec.to_tensor()
        assert t.shape == (12,)
        assert t.dtype == np.float32

    def test_catalog_has_16_entries(self):
        specs = get_hardware_specs()
        assert len(specs) == 16

    def test_catalog_has_expected_keys(self):
        specs = get_hardware_specs()
        expected = {
            "A100_80GB", "H100_SXM", "H200", "B200", "B300",
            "L40S", "RTX_5090",
            "MI300X", "MI325X", "MI350X",
            "TPU_v5e", "TPU_v6e", "TPU_v7",
            "Gaudi_3",
            "M4_Max", "Snapdragon_X_Elite",
        }
        assert expected.issubset(set(specs.keys()))

    def test_disk_fields_on_datacenter(self):
        """Datacenter GPUs should have disk > 0."""
        specs = get_hardware_specs()
        datacenter = ["H100_SXM", "A100_80GB", "H200", "B200", "B300",
                       "L40S", "RTX_5090", "MI300X", "MI325X", "MI350X", "Gaudi_3"]
        for name in datacenter:
            assert specs[name].disk_capacity > 0, f"{name} should have disk"
            assert specs[name].disk_bandwidth > 0, f"{name} should have disk_bandwidth"

    def test_edge_devices_no_disk(self):
        """Unified-memory edge devices should have disk = 0."""
        specs = get_hardware_specs()
        edge = ["M4_Max", "Snapdragon_X_Elite", "TPU_v5e", "TPU_v6e", "TPU_v7"]
        for name in edge:
            assert specs[name].disk_capacity == 0, f"{name} should have no disk"

    def test_legacy_fields_build_default_tier_hierarchy(self):
        h100 = get_hardware_specs()["H100_SXM"]
        tier_names = {t.name for t in h100.tiers}
        assert {"hbm", "cpu", "disk"}.issubset(tier_names)
        assert h100.transfer_bandwidth_to_hbm("cpu") == h100.cpu_gpu_bandwidth
        assert h100.transfer_bandwidth_to_hbm("disk") == h100.disk_bandwidth

    def test_custom_tier_hierarchy_supports_variable_depth(self):
        spec = HardwareSpec(
            name="deep-hierarchy",
            hbm_capacity=64 * GB,
            hbm_bandwidth=2 * TB,
            cpu_ram_capacity=0,
            cpu_gpu_bandwidth=0.0,
            fp16_flops=100 * TFLOPS,
            int8_flops=200 * TFLOPS,
            fp32_flops=20 * TFLOPS,
            sram_capacity=16 * (1 << 20),
            interconnect_bandwidth=0.0,
            tiers=(
                MemoryTier("hbm", capacity=64 * GB, bandwidth=2 * TB),
                MemoryTier("dram", capacity=256 * GB, bandwidth=64 * GB, parent="hbm", link_bandwidth_to_parent=64 * GB),
                MemoryTier("pmem", capacity=1 * TB, bandwidth=20 * GB, parent="dram", link_bandwidth_to_parent=20 * GB),
            ),
        )
        assert spec.transfer_bandwidth_to_hbm("dram") == 64 * GB
        assert spec.transfer_bandwidth_to_hbm("pmem") == 20 * GB


class TestModelConfigs:
    def test_llama31_8b_param_count(self):
        m = get_model_configs()["LLaMA-3.1-8B"]
        assert abs(m.num_params - 8e9) / 8e9 < 0.10, (
            f"LLaMA-3.1 8B params: {m.num_params / 1e9:.2f}B"
        )

    def test_llama31_8b_kv_bytes_per_token(self):
        m = get_model_configs()["LLaMA-3.1-8B"]
        # 2 (K+V) * 8 (kv_heads) * 128 (head_dim) * 2 (bf16) * 32 (layers)
        expected = 2 * 8 * 128 * 2 * 32
        assert m.kv_bytes_per_token == expected

    def test_kv_cache_bytes_scales_linearly(self):
        m = get_model_configs()["LLaMA-3.1-8B"]
        assert m.kv_cache_bytes(2048) == 2 * m.kv_cache_bytes(1024)

    def test_head_dim_default(self):
        m = get_model_configs()["LLaMA-3.1-8B"]
        assert m._head_dim == 128  # 4096 / 32

    def test_all_models_have_positive_params(self):
        for name, m in get_model_configs().items():
            assert m.num_params > 0, f"{name} num_params"
            assert m.param_bytes > 0, f"{name} param_bytes"
            assert m.kv_bytes_per_token > 0, f"{name} kv_bytes_per_token"

    def test_catalog_has_14_entries(self):
        models = get_model_configs()
        assert len(models) == 14

    def test_qwen25_72b_param_count(self):
        m = get_model_configs()["Qwen2.5-72B"]
        assert abs(m.num_params - 72e9) / 72e9 < 0.15, (
            f"Qwen2.5-72B params: {m.num_params / 1e9:.2f}B"
        )

    def test_phi4_14b_param_count(self):
        m = get_model_configs()["Phi-4-14B"]
        assert abs(m.num_params - 14e9) / 14e9 < 0.15, (
            f"Phi-4-14B params: {m.num_params / 1e9:.2f}B"
        )

    def test_gemma3_27b_param_count(self):
        m = get_model_configs()["Gemma-3-27B"]
        assert abs(m.num_params - 27e9) / 27e9 < 0.20, (
            f"Gemma-3-27B params: {m.num_params / 1e9:.2f}B"
        )

    def test_llama31_70b_param_count(self):
        m = get_model_configs()["LLaMA-3.1-70B"]
        assert abs(m.num_params - 70e9) / 70e9 < 0.15, (
            f"LLaMA-3.1-70B params: {m.num_params / 1e9:.2f}B"
        )
