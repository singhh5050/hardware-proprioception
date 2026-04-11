"""OverheadProfile — calibrated overhead correction on top of the roofline model.

The pure roofline (CostModel) predicts a theoretical lower bound. Real GPU inference
is 3–20x slower due to:

  1. Fixed launch overhead  — kernel dispatch, CUDA sync, framework bookkeeping
                               (constant per step, hardware-class-specific)
  2. Flash Attention tile scan — FA processes K/V in fixed-size blocks (tiles).
                               For autoregressive decode the outer tile loop is
                               serial; cost ∝ (N / fa_block_size)^exponent per head per layer.
                               Empirically exponent ≈ 1.5 for H100 FA2 (superlinear due to
                               cache pressure and pipeline stalls at large N).
  3. Memory allocator pressure — torch allocator overhead at large KV histories;
                               cost ∝ log2(N) (minor, implementation-dependent)

The corrected per-step time is:

    t_sim = roofline_efficiency * t_roofline
            + launch_overhead_s
            + attn_scan_coeff * (N / fa_block_size)^attn_scan_exponent * num_kv_heads * num_layers
            + alloc_coeff * log2(max(N, 1))

where:
    N                   = active tokens in attention this step
    fa_block_size       = FA tile size in tokens (typically 64 for FA2 decode)
    attn_scan_coeff     = seconds per head-layer tile^exponent
    attn_scan_exponent  = scan scaling exponent (1.0 = linear, 1.5 = empirical H100 FA2)
    num_kv_heads        = model.num_kv_heads  (GQA KV heads)
    num_layers          = model.num_layers

Key design decisions:
  - attn_scan_coeff is in units of seconds per HEAD-LAYER tile^exponent, not per
    aggregate tile. This makes it hardware-specific but model-agnostic — the model's
    num_kv_heads × num_layers is multiplied in at call time by the simulator.
  - attn_scan_exponent > 1.0 captures the empirical superlinearity of FA2 decode at
    large context lengths (cache pressure, pipeline stalls). Calibrated at 1.5 for H100
    FA2 using an 8-point context sweep (1K–128K) on LLaMA-3.2-3B; errors drop from
    ±34% (linear) to ±4% (superlinear) across the full range.

Pre-calibrated profiles:
    OVERHEAD_H100_FLASH2  — H100 SXM with flash_attention_2  (calibrated, 8 pts)
    OVERHEAD_A100_SDPA    — A100-40GB with SDPA               (1 context pt, estimated)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from hwprop.specs import HardwareSpec


@dataclass(frozen=True)
class OverheadProfile:
    """Calibrated overhead parameters for a (hardware, attention_impl) pair.

    All time values in seconds.
    """

    name: str

    # Multiplier on the roofline time. Range: (0, 1].
    roofline_efficiency: float

    # Fixed per-step overhead: kernel launch, CUDA sync, Python dispatch.
    launch_overhead_s: float

    # FA tile scan cost: seconds per HEAD-LAYER tile^exponent.
    # Total scan overhead = attn_scan_coeff * (N / fa_block_size)^attn_scan_exponent * num_kv_heads * num_layers
    attn_scan_coeff: float

    # Flash Attention tile size in tokens (FA2 decode default: 64).
    fa_block_size: int = 64

    # Scan scaling exponent. 1.0 = linear in tile count; 1.5 = empirical H100 FA2.
    attn_scan_exponent: float = 1.0

    # Memory allocator pressure: seconds per log2-token.
    alloc_coeff: float = 0.0

    def corrected_time(
        self,
        roofline_time_s: float,
        active_tokens: int,
        kv_head_layers: int = 1,
    ) -> float:
        """Apply overhead correction to a roofline step time.

        Args:
            roofline_time_s: Raw output of CostModel.step_cost().time_s.
            active_tokens:   Number of tokens participating in attention this step.
            kv_head_layers:  model.num_kv_heads * model.num_layers.
                             Pass 1 (default) for a hardware-only estimate.

        Returns:
            Corrected wall-clock time estimate in seconds.
        """
        tiles = (active_tokens / max(self.fa_block_size, 1)) ** self.attn_scan_exponent
        t_roofline = self.roofline_efficiency * roofline_time_s
        t_scan  = self.attn_scan_coeff * tiles * kv_head_layers
        t_alloc = self.alloc_coeff * math.log2(max(active_tokens, 1))
        return t_roofline + self.launch_overhead_s + t_scan + t_alloc

    def overhead_breakdown(
        self,
        roofline_time_s: float,
        active_tokens: int,
        kv_head_layers: int = 1,
    ) -> dict[str, float]:
        """Fractional contribution of each overhead component."""
        tiles = (active_tokens / max(self.fa_block_size, 1)) ** self.attn_scan_exponent
        t_roofline = self.roofline_efficiency * roofline_time_s
        t_scan  = self.attn_scan_coeff * tiles * kv_head_layers
        t_alloc = self.alloc_coeff * math.log2(max(active_tokens, 1))
        total   = t_roofline + self.launch_overhead_s + t_scan + t_alloc
        if total == 0.0:
            return {"roofline": 0.0, "launch": 0.0, "attn_scan": 0.0, "alloc": 0.0}
        return {
            "roofline":  t_roofline / total,
            "launch":    self.launch_overhead_s / total,
            "attn_scan": t_scan / total,
            "alloc":     t_alloc / total,
        }

    # ------------------------------------------------------------------
    # Calibration from benchmark data
    # ------------------------------------------------------------------

    @classmethod
    def calibrate(
        cls,
        rows: Sequence[dict],
        name: str = "calibrated",
        fa_block_size: int = 64,
        kv_head_layers: int = 1,
        attn_scan_exponent: float = 1.0,
    ) -> "OverheadProfile":
        """Fit overhead parameters from benchmark rows via non-negative least squares.

        Each row must have keys:
            measured_per_token_ms   — real wall-clock per token (ms)
            simulated_per_token_ms  — roofline prediction per token (ms)
            context_length          — tokens (used as active_tokens proxy)

        fa_block_size:       FA tile size in tokens (default 64).
        kv_head_layers:      model.num_kv_heads * model.num_layers for the benchmark model.
                             The fitted attn_scan_coeff will be in per-head-layer-tile^exponent units.
                             Default 1 → coefficient absorbs model structure (old behaviour).
        attn_scan_exponent:  Exponent for the tile-count term. 1.0 = linear (default);
                             1.5 matches empirical H100 FA2 behaviour.

        Fits: t_measured = α*t_rf + β_launch + γ*(N/block)^exponent*kv_head_layers + δ*log2(N)
        """
        if not rows:
            raise ValueError("calibrate() requires at least one benchmark row")

        measured, t_rf, n_tokens = [], [], []
        for r in rows:
            measured.append(float(r["measured_per_token_ms"]) / 1000.0)
            t_rf.append(float(r["simulated_per_token_ms"]) / 1000.0)
            n_tokens.append(int(r.get("context_length", r.get("seq_len", 1))))

        A = np.zeros((len(measured), 4), dtype=np.float64)
        for i, (trf, n) in enumerate(zip(t_rf, n_tokens)):
            A[i, 0] = trf
            A[i, 1] = 1.0
            A[i, 2] = (n / max(fa_block_size, 1)) ** attn_scan_exponent * kv_head_layers
            A[i, 3] = math.log2(max(n, 1))

        b = np.array(measured, dtype=np.float64)

        try:
            from scipy.optimize import nnls
            coeffs, _ = nnls(A, b)
        except ImportError:
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            coeffs = np.clip(coeffs, 0.0, None)

        return cls(
            name=name,
            roofline_efficiency=float(max(min(coeffs[0], 1.0), 0.05)),
            launch_overhead_s=float(coeffs[1]),
            attn_scan_coeff=float(coeffs[2]),
            fa_block_size=fa_block_size,
            attn_scan_exponent=attn_scan_exponent,
            alloc_coeff=float(coeffs[3]),
        )

    # ------------------------------------------------------------------
    # Analytic derivation for theoretical hardware
    # ------------------------------------------------------------------

    @classmethod
    def for_hardware(
        cls,
        hw: HardwareSpec,
        attn_impl: str = "flash_attention_2",
        roofline_efficiency: float = 0.7,
        fa_block_size: int = 64,
        attn_scan_exponent: float = 1.5,
    ) -> "OverheadProfile":
        """Derive a default overhead profile from hardware specs.

        Physical assumptions:
        - launch_overhead_s ∝ 1/fp32_flops  (host-side dispatch speed)
        - attn_scan_coeff   ∝ 1/fp16_flops  (tensor-core throughput per tile)
          Both scaled from H100 SXM reference calibration.

        The returned attn_scan_coeff is in per-head-layer-tile^exponent units.
        Multiply by num_kv_heads × num_layers when computing step cost.
        """
        _REF_FP32_FLOPS  = 67e12        # H100 SXM
        _REF_FP16_FLOPS  = 990e12       # H100 SXM
        _REF_LAUNCH      = 0.016674     # s   (recalibrated)
        _REF_SCAN_COEFF  = 4.926e-9     # s/head-layer-tile^1.5 (recalibrated, exponent=1.5)
        _REF_ALLOC_COEFF = 0.0

        launch     = _REF_LAUNCH     * (_REF_FP32_FLOPS / max(hw.fp32_flops, 1e9))
        scan_coeff = _REF_SCAN_COEFF * (_REF_FP16_FLOPS / max(hw.fp16_flops, 1e9))

        return cls(
            name=f"{hw.name}_{attn_impl}_derived",
            roofline_efficiency=roofline_efficiency,
            launch_overhead_s=launch,
            attn_scan_coeff=scan_coeff,
            fa_block_size=fa_block_size,
            attn_scan_exponent=attn_scan_exponent,
            alloc_coeff=_REF_ALLOC_COEFF,
        )


# ---------------------------------------------------------------------------
# Pre-calibrated profiles for known hardware
# ---------------------------------------------------------------------------

# H100 SXM · flash_attention_2 · calibrated on LLaMA-3.2-3B (8 KV heads, 28 layers)
# 8-point context sweep (1K–128K), NNLS fit with attn_scan_exponent=1.5.
# Errors: ±4% across full range (vs ±34% with linear exponent=1.0).
OVERHEAD_H100_FLASH2 = OverheadProfile(
    name="H100_SXM_flash_attention_2",
    roofline_efficiency=0.05,
    launch_overhead_s=0.016674,
    attn_scan_coeff=4.926e-9,    # s per head-layer-tile^1.5
    fa_block_size=64,
    attn_scan_exponent=1.5,
    alloc_coeff=0.0,
)

# A100-40GB · SDPA · estimated from Qwen2.5-7B (4 KV heads, 28 layers) at 7643 tokens
# attn_scan_coeff = 6.37e-5 s/aggregate-tile ÷ (4 × 28) = 5.69e-7 s/head-layer-tile
# Note: SDPA is less efficient per tile than FA2; single context point so scan/launch
# are degenerate — treat as approximate.
OVERHEAD_A100_SDPA = OverheadProfile(
    name="A100_40GB_sdpa",
    roofline_efficiency=0.05,
    launch_overhead_s=0.02165,
    attn_scan_coeff=5.69e-7,     # 0.569 µs per head-layer tile (SDPA > FA2 per tile)
    fa_block_size=64,
    alloc_coeff=4.8e-5,
)
