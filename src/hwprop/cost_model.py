"""CostModel — pure, stateless roofline math for LLM decode steps.

Implements the core roofline model:
  - Attention is ALWAYS memory-bandwidth bound during decode (arithmetic intensity ~ 1).
  - MLP is memory-bound when batch_size < B_crit, compute-bound above.
  - Quantized KV entries use half the bytes but same compute (dequantize on the fly).
  - HBM loads and compute overlap (roofline: max); CPU and disk transfers are additive (blocking).

KV bandwidth degradation model:
  At short context, the KV cache fits in L2 and access is fast. At long context, KV entries
  for different heads/layers are scattered across HBM — the memory controller chases fragments
  instead of bulk-streaming — and effective bandwidth drops well below spec. This is modeled as:

      effective_kv_bw = hbm_bw / (1 + kv_bandwidth_alpha × (kv_bytes / sram_capacity)^kv_bandwidth_beta)

  The base roofline uses spec HBM bandwidth for model-weight loads (always sequential) and
  effective_kv_bw for KV cache loads. When kv_bandwidth_alpha=0 (default), reduces to flat BW.

  kv_bandwidth_alpha and kv_bandwidth_beta are hardware+attention-impl specific and are
  supplied by OverheadProfile (calibrated) rather than read from HardwareSpec (spec-sheet only).
"""

from __future__ import annotations

from dataclasses import dataclass

from hwprop.specs import HardwareSpec, ModelConfig, BYTES_PER_INT8


# ---------------------------------------------------------------------------
# StepCost — cost breakdown for one decode step
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StepCost:
    """Cost breakdown for a single decode step."""

    time_s: float          # wall-clock seconds for this step
    hbm_bytes: int         # total HBM occupied (params + KV)
    cpu_bytes: int         # CPU RAM occupied (offloaded KV)
    disk_bytes: int        # disk occupied (cold KV)
    flops: float           # total FLOPs performed
    hbm_overflow: bool     # exceeds HBM capacity?


# ---------------------------------------------------------------------------
# KVCacheState — scalar KV cache occupancy across tiers
# ---------------------------------------------------------------------------
@dataclass
class KVCacheState:
    """KV cache occupancy (uniform across all layers).

    Token counts are scalars — the CostModel multiplies by num_layers internally.
    This matches what a global (keep, quant, offload, disk) policy produces.
    """

    seq_len: int                    # total tokens generated so far
    tokens_in_hbm: int              # full-precision tokens in HBM
    tokens_in_hbm_quantized: int    # INT8 tokens in HBM
    tokens_in_cpu: int              # tokens offloaded to CPU
    tokens_on_disk: int             # tokens offloaded to disk/NVMe
    tokens_evicted: int             # permanently dropped tokens

    @property
    def active_tokens(self) -> int:
        """Tokens that participate in attention (everything except evicted)."""
        return (
            self.tokens_in_hbm
            + self.tokens_in_hbm_quantized
            + self.tokens_in_cpu
            + self.tokens_on_disk
        )


# ---------------------------------------------------------------------------
# CostModel — pure roofline math: (hardware, model, kv_state) → StepCost
# ---------------------------------------------------------------------------
class CostModel:
    """Stateless analytical cost model for a (hardware, model) pair."""

    def __init__(
        self,
        hardware: HardwareSpec,
        model: ModelConfig,
        kv_bandwidth_alpha: float = 0.0,
        kv_bandwidth_beta: float = 1.0,
    ) -> None:
        self.hw = hardware
        self.model = model
        self.kv_bandwidth_alpha = kv_bandwidth_alpha
        self.kv_bandwidth_beta = kv_bandwidth_beta

    def step_cost(self, kv_state: KVCacheState, batch_size: int = 1) -> StepCost:
        """Cost of generating one token given current KV occupancy.

        Global roofline (Scaling Book formula):
          memory_time  = (param_bytes + hbm_kv_bytes) / hbm_bw
          compute_time = (mlp_flops + attn_flops) / fp16_flops
          wall_clock   = max(memory_time, compute_time) + cpu_transfer_time + disk_transfer_time
        """
        hw = self.hw
        m = self.model
        L = m.num_layers

        # --- HBM bytes loaded ---
        param_bytes = m.param_bytes

        kv_fp16_per_layer = m.kv_bytes_per_token_per_layer()
        kv_int8_per_layer = 2 * m.num_kv_heads * m._head_dim * BYTES_PER_INT8

        hbm_kv_bytes = (
            kv_state.tokens_in_hbm * kv_fp16_per_layer
            + kv_state.tokens_in_hbm_quantized * kv_int8_per_layer
        ) * L * batch_size

        param_time = param_bytes / hw.hbm_bandwidth
        if self.kv_bandwidth_alpha > 0 and hw.sram_capacity > 0:
            ratio = hbm_kv_bytes / hw.sram_capacity
            effective_kv_bw = hw.hbm_bandwidth / (1 + self.kv_bandwidth_alpha * ratio ** self.kv_bandwidth_beta)
        else:
            effective_kv_bw = hw.hbm_bandwidth
        kv_time = hbm_kv_bytes / effective_kv_bw if effective_kv_bw > 0 else 0.0
        memory_time = param_time + kv_time

        # --- CPU transfer (additive — blocks on PCIe) ---
        cpu_kv_bytes = kv_state.tokens_in_cpu * kv_fp16_per_layer * L * batch_size
        cpu_to_hbm_bw = hw.transfer_bandwidth_to_hbm("cpu")
        if cpu_to_hbm_bw > 0:
            cpu_transfer_time = cpu_kv_bytes / cpu_to_hbm_bw
        else:
            cpu_transfer_time = 0.0

        # --- Disk transfer (additive — blocks on NVMe read) ---
        disk_kv_bytes = kv_state.tokens_on_disk * kv_fp16_per_layer * L * batch_size
        disk_to_hbm_bw = hw.transfer_bandwidth_to_hbm("disk")
        if disk_to_hbm_bw > 0:
            disk_transfer_time = disk_kv_bytes / disk_to_hbm_bw
        else:
            disk_transfer_time = 0.0

        # --- FLOPs ---
        mlp_flops = 2.0 * batch_size * L * m.mlp_params_per_layer
        active = kv_state.active_tokens
        attn_flops = 4.0 * batch_size * active * L * m.num_heads * m._head_dim
        total_flops = mlp_flops + attn_flops

        compute_time = total_flops / hw.fp16_flops

        # --- Wall-clock: global roofline ---
        wall_clock = max(memory_time, compute_time) + cpu_transfer_time + disk_transfer_time

        # --- Memory usage ---
        hbm_used = param_bytes + int(hbm_kv_bytes)
        cpu_used = int(cpu_kv_bytes)
        disk_used = int(disk_kv_bytes)
        hbm_overflow = hbm_used > hw.hbm_capacity

        return StepCost(
            time_s=wall_clock,
            hbm_bytes=hbm_used,
            cpu_bytes=cpu_used,
            disk_bytes=disk_used,
            flops=total_flops,
            hbm_overflow=hbm_overflow,
        )

    def prefill_cost(self, prompt_len: int, batch_size: int = 1) -> StepCost:
        """Prefill cost (TTFT) — compute-bound for long sequences.

        FLOPs ~ 2 * num_params * seq_len * batch_size
        Time  = max(FLOPs / FP16_FLOPS, param_bytes / HBM_BW)
        """
        hw = self.hw
        m = self.model

        flops = 2.0 * m.num_params * prompt_len * batch_size
        compute_time = flops / hw.fp16_flops
        memory_time = m.param_bytes / hw.hbm_bandwidth
        wall_clock = max(compute_time, memory_time)

        # During prefill, all KV lands in HBM
        kv_bytes = m.kv_bytes_per_token * prompt_len * batch_size
        hbm_used = m.param_bytes + kv_bytes
        hbm_overflow = hbm_used > hw.hbm_capacity

        return StepCost(
            time_s=wall_clock,
            hbm_bytes=hbm_used,
            cpu_bytes=0,
            disk_bytes=0,
            flops=flops,
            hbm_overflow=hbm_overflow,
        )

    @property
    def min_step_time(self) -> float:
        """Absolute floor: param_bytes / hbm_bw (no KV, no compute)."""
        return self.model.param_bytes / self.hw.hbm_bandwidth
