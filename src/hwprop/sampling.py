"""Synthetic hardware sampler.

Random HardwareSpec instances with log-uniform distributions
spanning edge (Snapdragon) to datacenter (B300/MI350X) ranges.
"""

from __future__ import annotations

import numpy as np

from hwprop.specs import HardwareSpec, GB, TB, TFLOPS


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    """Sample from a log-uniform distribution over [low, high]."""
    log_low = np.log(low)
    log_high = np.log(high)
    return float(np.exp(rng.uniform(log_low, log_high)))


def sample_synthetic_hardware(
    rng: np.random.Generator | None = None,
) -> HardwareSpec:
    """Sample a synthetic hardware spec with plausible-but-random parameters.

    Ranges span edge devices to datacenter GPUs:
      - HBM:   4 – 300 GB
      - BW:    100 GB/s – 10 TB/s
      - FLOPs: 5 – 3000 TFLOPS (FP16)
      - SRAM:  4 – 256 MB
      - CPU:   0 – 1 TB RAM, 0 – 256 GB/s PCIe
      - Interconnect: 0 – 3.2 TB/s
      - Disk:  60% chance of NVMe (256 GB – 8 TB, 1 – 7 GB/s)

    20% chance of unified memory (Apple / TPU-like: no CPU RAM).

    # TODO: add correlated sampling — real hardware has correlated
    # FLOPs/bandwidth (higher-end chips have both). Uncorrelated
    # log-uniform can produce unrealistic combos. Fine for v1.
    """
    if rng is None:
        rng = np.random.default_rng()

    unified = rng.random() < 0.2

    hbm_capacity = int(_log_uniform(rng, 4 * GB, 300 * GB))
    hbm_bandwidth = _log_uniform(rng, 100 * GB, 10 * TB)

    fp16_flops = _log_uniform(rng, 5 * TFLOPS, 3000 * TFLOPS)
    int8_flops = fp16_flops * rng.uniform(1.5, 2.5)  # INT8 usually 2x FP16
    fp32_flops = fp16_flops * rng.uniform(0.05, 0.3)  # FP32 much lower

    sram_capacity = int(_log_uniform(rng, 4 * (1 << 20), 256 * (1 << 20)))

    if unified:
        cpu_ram_capacity = 0
        cpu_gpu_bandwidth = 0.0
        interconnect_bandwidth = _log_uniform(rng, 100 * GB, 3200 * GB) if rng.random() < 0.5 else 0.0
    else:
        cpu_ram_capacity = int(_log_uniform(rng, 32 * GB, 1024 * GB))
        cpu_gpu_bandwidth = _log_uniform(rng, 8 * GB, 256 * GB)
        interconnect_bandwidth = (
            _log_uniform(rng, 100 * GB, 3200 * GB) if rng.random() < 0.7 else 0.0
        )

    # Disk / NVMe: 60% chance (servers), 40% none (edge)
    if rng.random() < 0.6:
        disk_capacity = int(_log_uniform(rng, 256 * GB, 8 * TB))
        disk_bandwidth = _log_uniform(rng, 1 * GB, 7 * GB)
    else:
        disk_capacity = 0
        disk_bandwidth = 0.0

    return HardwareSpec(
        name="synthetic",
        hbm_capacity=hbm_capacity,
        hbm_bandwidth=hbm_bandwidth,
        cpu_ram_capacity=cpu_ram_capacity,
        cpu_gpu_bandwidth=cpu_gpu_bandwidth,
        fp16_flops=fp16_flops,
        int8_flops=int8_flops,
        fp32_flops=fp32_flops,
        sram_capacity=sram_capacity,
        interconnect_bandwidth=interconnect_bandwidth,
        unified_memory=unified,
        disk_capacity=disk_capacity,
        disk_bandwidth=disk_bandwidth,
    )
