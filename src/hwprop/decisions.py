"""KV cache management decisions and step cost output.

KVDecision represents the per-layer allocation of KV cache tokens
across storage tiers. StepCost is the detailed cost breakdown
returned by the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# KVDecision — per-layer KV cache placement
# ---------------------------------------------------------------------------
@dataclass
class KVDecision:
    """Per-layer KV cache management decision.

    Each array has shape (num_layers,) and specifies how many tokens
    of KV cache for that layer reside in each tier.
    """

    tokens_in_hbm_full: np.ndarray       # full-precision tokens in HBM
    tokens_in_hbm_quantized: np.ndarray   # INT8/INT4 quantized tokens in HBM
    tokens_in_cpu: np.ndarray             # tokens offloaded to CPU RAM
    tokens_evicted: np.ndarray            # tokens permanently dropped

    @property
    def num_layers(self) -> int:
        return len(self.tokens_in_hbm_full)

    @property
    def total_tokens_per_layer(self) -> np.ndarray:
        """Total tokens across all tiers (should equal original seq_len)."""
        return (
            self.tokens_in_hbm_full
            + self.tokens_in_hbm_quantized
            + self.tokens_in_cpu
            + self.tokens_evicted
        )

    @property
    def active_tokens_per_layer(self) -> np.ndarray:
        """Tokens that participate in attention (everything except evicted)."""
        return (
            self.tokens_in_hbm_full
            + self.tokens_in_hbm_quantized
            + self.tokens_in_cpu
        )

    # --- factory methods -------------------------------------------------------

    @classmethod
    def all_in_hbm(cls, num_layers: int, seq_len: int) -> KVDecision:
        """Baseline: all KV cache in HBM at full precision."""
        return cls(
            tokens_in_hbm_full=np.full(num_layers, seq_len, dtype=np.int64),
            tokens_in_hbm_quantized=np.zeros(num_layers, dtype=np.int64),
            tokens_in_cpu=np.zeros(num_layers, dtype=np.int64),
            tokens_evicted=np.zeros(num_layers, dtype=np.int64),
        )

    @classmethod
    def uniform_evict(
        cls, num_layers: int, seq_len: int, keep_frac: float
    ) -> KVDecision:
        """Evict a uniform fraction of tokens across all layers."""
        keep = int(seq_len * keep_frac)
        evict = seq_len - keep
        return cls(
            tokens_in_hbm_full=np.full(num_layers, keep, dtype=np.int64),
            tokens_in_hbm_quantized=np.zeros(num_layers, dtype=np.int64),
            tokens_in_cpu=np.zeros(num_layers, dtype=np.int64),
            tokens_evicted=np.full(num_layers, evict, dtype=np.int64),
        )

    @classmethod
    def from_global_policy(
        cls,
        keep_frac: float,
        quant_frac: float,
        offload_frac: float,
        num_layers: int,
        seq_len: int,
    ) -> KVDecision:
        """Apply a single (keep, quantize, offload) policy uniformly.

        keep_frac + quant_frac + offload_frac <= 1.0
        Remainder is evicted.
        """
        keep = int(seq_len * keep_frac)
        quant = int(seq_len * quant_frac)
        offload = int(seq_len * offload_frac)
        evict = seq_len - keep - quant - offload
        if evict < 0:
            raise ValueError(
                f"Fractions sum to {keep_frac + quant_frac + offload_frac:.3f} > 1.0"
            )
        return cls(
            tokens_in_hbm_full=np.full(num_layers, keep, dtype=np.int64),
            tokens_in_hbm_quantized=np.full(num_layers, quant, dtype=np.int64),
            tokens_in_cpu=np.full(num_layers, offload, dtype=np.int64),
            tokens_evicted=np.full(num_layers, evict, dtype=np.int64),
        )


# ---------------------------------------------------------------------------
# StepCost — detailed cost breakdown for one decode step
# ---------------------------------------------------------------------------
@dataclass
class StepCost:
    """Cost breakdown for a single decode step."""

    # Time components (seconds)
    param_load_time: float       # loading model weights from HBM
    kv_hbm_load_time: float      # loading KV cache from HBM
    kv_cpu_transfer_time: float  # transferring KV cache from CPU over PCIe
    mlp_compute_time: float      # MLP FLOPs time
    attn_compute_time: float     # attention FLOPs time

    # Memory usage (bytes)
    hbm_used: int                # total HBM occupied (params + KV)
    cpu_ram_used: int            # CPU RAM occupied by offloaded KV

    # Capacity
    hbm_overflow: bool           # True if hbm_used > hbm_capacity
    cpu_overflow: bool           # True if cpu_ram_used > cpu_ram_capacity

    # Quality
    retention_ratio: float       # fraction of original tokens still active

    # Compute
    total_flops: float           # raw FLOPs performed (MLP + attention)

    # --- derived properties ----------------------------------------------------

    @property
    def total_hbm_time(self) -> float:
        """Total time for all HBM reads (params + KV)."""
        return self.param_load_time + self.kv_hbm_load_time

    @property
    def total_compute_time(self) -> float:
        """Total compute time (MLP + attention)."""
        return self.mlp_compute_time + self.attn_compute_time

    @property
    def wall_clock_time(self) -> float:
        """Estimated wall-clock time for one decode step.

        HBM loads and compute overlap (roofline: take the max),
        CPU transfer is additive (blocks on PCIe).
        """
        return max(self.total_hbm_time, self.total_compute_time) + self.kv_cpu_transfer_time

    @property
    def tokens_per_second(self) -> float:
        """Throughput: 1 / wall_clock_time."""
        if self.wall_clock_time <= 0:
            return float("inf")
        return 1.0 / self.wall_clock_time
