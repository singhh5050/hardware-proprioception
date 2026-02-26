"""Tests for the synthetic hardware sampler."""

import numpy as np
import pytest

from hwprop.sampling import sample_synthetic_hardware
from hwprop.specs import GB, TFLOPS


class TestSyntheticSampling:
    def test_all_specs_positive(self):
        """100 synthetic specs should all have positive key values."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            spec = sample_synthetic_hardware(rng)
            assert spec.hbm_capacity > 0
            assert spec.hbm_bandwidth > 0
            assert spec.fp16_flops > 0
            assert spec.int8_flops > 0
            assert spec.fp32_flops > 0

    def test_plausible_bcrit(self):
        """100 synthetic specs should have plausible B_crit values."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            spec = sample_synthetic_hardware(rng)
            b_crit = spec.critical_batch_size_fp16
            # B_crit should be positive and not astronomical
            assert 0 < b_crit < 100_000, f"B_crit = {b_crit:.0f}"

    def test_unified_memory_fraction(self):
        """About 20% of samples should have unified memory."""
        rng = np.random.default_rng(42)
        n = 1000
        unified_count = sum(
            sample_synthetic_hardware(rng).unified_memory for _ in range(n)
        )
        # Should be roughly 20% ± 5%
        frac = unified_count / n
        assert 0.10 < frac < 0.30, f"Unified fraction = {frac:.2f}"

    def test_deterministic_with_seed(self):
        """Same seed should produce same hardware."""
        spec1 = sample_synthetic_hardware(np.random.default_rng(123))
        spec2 = sample_synthetic_hardware(np.random.default_rng(123))
        assert spec1.hbm_capacity == spec2.hbm_capacity
        assert spec1.fp16_flops == spec2.fp16_flops

    def test_hbm_in_range(self):
        """HBM capacity should be in the specified range."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            spec = sample_synthetic_hardware(rng)
            assert 4 * GB <= spec.hbm_capacity <= 300 * GB

    def test_unified_has_no_cpu(self):
        """Unified memory devices should have 0 CPU RAM."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            spec = sample_synthetic_hardware(rng)
            if spec.unified_memory:
                assert spec.cpu_ram_capacity == 0
                assert spec.cpu_gpu_bandwidth == 0.0
