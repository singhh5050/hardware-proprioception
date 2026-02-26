"""HardwareSimulator — analytical roofline-based cost model for LLM decode steps.

Implements the core roofline math:
  - During decode, attention is ALWAYS memory-bandwidth bound (arithmetic intensity ≈ 1).
  - MLP is memory-bound when batch_size < B_crit, compute-bound when batch_size > B_crit.
  - Quantized KV entries use half the bytes but the same compute (dequantize on the fly).
  - HBM loads and compute overlap (roofline: max); CPU transfer is additive (PCIe blocking).
"""

from __future__ import annotations

import numpy as np

from hwprop.specs import HardwareSpec, ModelConfig, BYTES_PER_INT8
from hwprop.decisions import KVDecision, StepCost


class HardwareSimulator:
    """Analytical cost model for a (hardware, model) pair."""

    def __init__(self, hardware: HardwareSpec, model: ModelConfig) -> None:
        self.hw = hardware
        self.model = model

    # -------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------
    def compute_decode_step_cost(
        self,
        decisions: KVDecision,
        batch_size: int = 1,
        original_seq_len: int | None = None,
    ) -> StepCost:
        """Compute detailed cost for one decode step given KV cache decisions.

        Args:
            decisions: Per-layer KV cache placement.
            batch_size: Number of sequences being decoded in parallel.
            original_seq_len: Original sequence length before any eviction.
                Used for retention_ratio. If None, inferred from decisions.
        """
        hw = self.hw
        m = self.model
        L = m.num_layers

        if original_seq_len is None:
            original_seq_len = int(decisions.total_tokens_per_layer[0])

        # --- Parameter load time ---
        param_bytes = m.param_bytes
        param_load_time = param_bytes / hw.hbm_bandwidth

        # --- KV cache load from HBM ---
        kv_bytes_per_token_per_layer = m.kv_bytes_per_token_per_layer()
        kv_quant_bytes_per_token_per_layer = (
            2 * m.num_kv_heads * m._head_dim * BYTES_PER_INT8
        )

        hbm_kv_bytes = float(np.sum(
            decisions.tokens_in_hbm_full * kv_bytes_per_token_per_layer
            + decisions.tokens_in_hbm_quantized * kv_quant_bytes_per_token_per_layer
        ))
        # Scale by batch_size: each sequence has its own KV cache
        hbm_kv_bytes *= batch_size
        kv_hbm_load_time = hbm_kv_bytes / hw.hbm_bandwidth

        # --- KV cache transfer from CPU ---
        cpu_kv_bytes = float(np.sum(
            decisions.tokens_in_cpu * kv_bytes_per_token_per_layer
        ))
        cpu_kv_bytes *= batch_size
        if hw.cpu_gpu_bandwidth > 0:
            kv_cpu_transfer_time = cpu_kv_bytes / hw.cpu_gpu_bandwidth
        else:
            kv_cpu_transfer_time = 0.0

        # --- MLP compute time ---
        # FLOPs per token: 2 * mlp_params_per_layer (multiply-accumulate)
        # Total: 2 * batch_size * num_layers * mlp_params_per_layer
        mlp_flops = 2.0 * batch_size * L * m.mlp_params_per_layer

        # Roofline: MLP can be compute-bound at high batch sizes
        mlp_bytes_loaded = L * m.mlp_params_per_layer * m.bytes_per_param
        mlp_memory_time = mlp_bytes_loaded / hw.hbm_bandwidth
        mlp_compute_time_pure = mlp_flops / hw.fp16_flops
        mlp_compute_time = max(mlp_memory_time, mlp_compute_time_pure)

        # --- Attention compute time ---
        # Attention FLOPs per layer: 4 * batch_size * active_seq_len * num_heads * head_dim
        #   (2 for Q@K^T, 2 for scores@V — each is 2*BS*S*N*H multiply-accumulate)
        # But attention is virtually always memory-bound during decode (BS=1..few hundred),
        # so the bottleneck is loading KV, not the FLOPs.
        # We still compute both sides of the roofline for correctness.
        active_tokens = decisions.active_tokens_per_layer  # (L,)
        attn_flops_per_layer = (
            4.0 * batch_size * active_tokens * m.num_heads * m._head_dim
        )
        total_attn_flops = float(np.sum(attn_flops_per_layer))
        attn_compute_time_pure = total_attn_flops / hw.fp16_flops

        # Attention memory time is already captured in kv_hbm_load_time above,
        # plus the query projection is part of param_load_time.
        # For the roofline, attention compute is overlapped with KV loads.
        attn_compute_time = attn_compute_time_pure

        # --- Total FLOPs ---
        total_flops = mlp_flops + total_attn_flops

        # --- Memory usage ---
        hbm_for_params = param_bytes
        hbm_for_kv = int(hbm_kv_bytes)  # already scaled by batch_size
        hbm_used = hbm_for_params + hbm_for_kv

        cpu_ram_used = int(cpu_kv_bytes)  # already scaled by batch_size

        # --- Capacity flags ---
        hbm_overflow = hbm_used > hw.hbm_capacity
        cpu_overflow = (
            cpu_ram_used > hw.cpu_ram_capacity
            if hw.cpu_ram_capacity > 0
            else cpu_ram_used > 0 and hw.unified_memory
        )

        # --- Retention ratio ---
        active = float(np.mean(decisions.active_tokens_per_layer))
        retention_ratio = active / original_seq_len if original_seq_len > 0 else 1.0

        return StepCost(
            param_load_time=param_load_time,
            kv_hbm_load_time=kv_hbm_load_time,
            kv_cpu_transfer_time=kv_cpu_transfer_time,
            mlp_compute_time=mlp_compute_time,
            attn_compute_time=attn_compute_time,
            hbm_used=hbm_used,
            cpu_ram_used=cpu_ram_used,
            hbm_overflow=hbm_overflow,
            cpu_overflow=cpu_overflow,
            retention_ratio=retention_ratio,
            total_flops=total_flops,
        )

    # -------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------

    def compute_baseline_cost(
        self, seq_len: int, batch_size: int = 1
    ) -> StepCost:
        """Cost with all KV cache in HBM at full precision (no eviction)."""
        decisions = KVDecision.all_in_hbm(self.model.num_layers, seq_len)
        return self.compute_decode_step_cost(decisions, batch_size, seq_len)

    def theoretical_min_step_time(
        self, seq_len: int, batch_size: int = 1
    ) -> float:
        """Lower bound: must at minimum load params + KV from HBM.

        This ignores compute entirely — pure memory-bandwidth floor.
        """
        param_bytes = self.model.param_bytes
        kv_bytes = self.model.kv_bytes_per_token * seq_len * batch_size
        return (param_bytes + kv_bytes) / self.hw.hbm_bandwidth

    def theoretical_max_throughput(
        self, seq_len: int, batch_size: int = 1
    ) -> float:
        """Upper bound on tokens/second (inverse of theoretical min step time)."""
        t = self.theoretical_min_step_time(seq_len, batch_size)
        return 1.0 / t if t > 0 else float("inf")

    def min_decode_latency(self) -> float:
        """Absolute floor: just loading model weights, no KV cache.

        param_bytes / HBM_bandwidth
        """
        return self.model.param_bytes / self.hw.hbm_bandwidth

    def prefill_time(self, seq_len: int) -> float:
        """Estimate time-to-first-token (TTFT) for prefill.

        Prefill is compute-bound for long sequences:
          FLOPs ≈ 2 * num_params * seq_len
          Time  = max(FLOPs / FP16_FLOPS, param_bytes / HBM_BW)
        """
        flops = 2.0 * self.model.num_params * seq_len
        compute_time = flops / self.hw.fp16_flops
        memory_time = self.model.param_bytes / self.hw.hbm_bandwidth
        return max(compute_time, memory_time)

    def max_batch_size(self, seq_len: int) -> int:
        """Maximum batch size that fits in HBM (params + KV cache).

        Returns at least 1 even if the model alone fills HBM.
        """
        param_bytes = self.model.param_bytes
        available = self.hw.hbm_capacity - param_bytes
        if available <= 0:
            return 1
        kv_per_seq = self.model.kv_bytes_per_token * seq_len
        if kv_per_seq <= 0:
            return 1
        return max(1, int(available // kv_per_seq))

    def general_step_time(self, seq_len: int, batch_size: int = 1) -> float:
        """Quick estimate of decode step time without explicit KV decisions.

        NOTE: This is an approximation that treats all params uniformly
        rather than decomposing MLP vs attention. For precise results,
        use compute_decode_step_cost() with explicit KVDecision.

        Formula: max(memory_time, compute_time) where
          memory_time = (param_bytes + kv_bytes * batch_size) / HBM_BW
          compute_time = 2 * num_params * batch_size / FP16_FLOPS
        """
        param_bytes = self.model.param_bytes
        kv_bytes = self.model.kv_bytes_per_token * seq_len * batch_size

        memory_time = (param_bytes + kv_bytes) / self.hw.hbm_bandwidth
        compute_time = 2.0 * self.model.num_params * batch_size / self.hw.fp16_flops

        return max(memory_time, compute_time)
