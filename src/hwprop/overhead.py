"""OverheadProfile — calibrated overhead correction on top of the roofline model.

The pure roofline (CostModel) predicts a theoretical lower bound. Real GPU inference
is 3–20x slower due to:

  1. Fixed launch overhead  — kernel dispatch, CUDA sync, framework bookkeeping
                               (constant per step, hardware-class-specific)
  2. Flash Attention tile scan — FA processes K/V in fixed-size blocks (tiles).
                               For autoregressive decode the outer tile loop is
                               serial; cost ∝ N / fa_block_size tile iterations.
  3. Memory allocator pressure — torch allocator overhead at large KV histories;
                               cost ∝ log2(N) (minor, implementation-dependent)

The corrected per-step time is:

    t_sim = roofline_efficiency * t_roofline
            + launch_overhead_s
            + attn_scan_coeff * (N / fa_block_size)
            + alloc_coeff * log2(max(N, 1))

where:
    N             = active tokens in attention this step
    fa_block_size = FA tile size in tokens (typically 64 for FA2 decode)
    attn_scan_coeff = seconds per tile iteration (~47 µs for H100 FA2)

Previous version used N / sram_capacity (tokens/bytes ≈ 0) — a dimensional bug
that made the scan term essentially zero and caused 20x error at 128K context.
The correct unit is tile iterations: N / fa_block_size, where fa_block_size ≈ 64.

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

    # FA tile scan: seconds per tile iteration.
    # Effective overhead = attn_scan_coeff * (N / fa_block_size).
    # ~47 µs/tile for H100 FA2; ~64 µs/tile for A100 SDPA.
    attn_scan_coeff: float

    # Flash Attention block size in tokens.
    # FA2 decode uses 64 tokens/block (determined by head_dim and SRAM tile policy).
    fa_block_size: int = 64

    # Memory allocator pressure coefficient.
    # Effective overhead = alloc_coeff * log2(max(N, 1)).
    alloc_coeff: float = 0.0

    def corrected_time(self, roofline_time_s: float, active_tokens: int) -> float:
        """Apply overhead correction to a roofline step time.

        Args:
            roofline_time_s: Raw output of CostModel.step_cost().time_s.
            active_tokens:   Number of tokens participating in attention this step.

        Returns:
            Corrected wall-clock time estimate in seconds.
        """
        t_roofline = self.roofline_efficiency * roofline_time_s
        t_scan  = self.attn_scan_coeff * active_tokens / max(self.fa_block_size, 1)
        t_alloc = self.alloc_coeff * math.log2(max(active_tokens, 1))
        return t_roofline + self.launch_overhead_s + t_scan + t_alloc

    def overhead_breakdown(self, roofline_time_s: float, active_tokens: int) -> dict[str, float]:
        """Fractional contribution of each overhead component."""
        t_roofline = self.roofline_efficiency * roofline_time_s
        t_scan  = self.attn_scan_coeff * active_tokens / max(self.fa_block_size, 1)
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
    ) -> "OverheadProfile":
        """Fit overhead parameters from benchmark rows via non-negative least squares.

        Each row must have keys:
            measured_per_token_ms   — real wall-clock per token (ms)
            simulated_per_token_ms  — roofline prediction per token (ms)
            context_length          — tokens (used as active_tokens proxy)

        fa_block_size: Flash Attention tile size in tokens (default 64).

        Fits:  t_measured = α*t_roofline + β_launch + γ*(N/fa_block_size) + δ*log2(N)
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
            A[i, 0] = trf                                  # roofline_efficiency
            A[i, 1] = 1.0                                  # launch_overhead_s
            A[i, 2] = n / max(fa_block_size, 1)            # attn_scan_coeff (tiles)
            A[i, 3] = math.log2(max(n, 1))                 # alloc_coeff

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
    ) -> "OverheadProfile":
        """Derive a default overhead profile from hardware specs.

        Physical assumptions:
        - launch_overhead_s scales inversely with fp32_flops (host-side speed).
          Reference: H100 SXM ~14.5 ms, fp32_flops = 67 TFLOPS.
        - attn_scan_coeff (µs/tile) scales inversely with fp16_flops (faster
          tensor cores → faster per-tile softmax reduction).
          Reference: H100 SXM ~47 µs/tile, fp16_flops = 990 TFLOPS.
        - fa_block_size: fixed at caller-supplied value (default 64).
        - alloc_coeff: conservative fixed default.
        """
        _REF_FP32_FLOPS  = 67e12        # H100 SXM
        _REF_FP16_FLOPS  = 990e12       # H100 SXM
        _REF_LAUNCH      = 0.01447      # s  (calibrated)
        _REF_SCAN_COEFF  = 4.683e-5     # s/tile (calibrated from H100 FA2)
        _REF_ALLOC_COEFF = 4.8e-5       # s/log2-token

        launch     = _REF_LAUNCH    * (_REF_FP32_FLOPS / max(hw.fp32_flops, 1e9))
        scan_coeff = _REF_SCAN_COEFF * (_REF_FP16_FLOPS / max(hw.fp16_flops, 1e9))

        return cls(
            name=f"{hw.name}_{attn_impl}_derived",
            roofline_efficiency=roofline_efficiency,
            launch_overhead_s=launch,
            attn_scan_coeff=scan_coeff,
            fa_block_size=fa_block_size,
            alloc_coeff=_REF_ALLOC_COEFF,
        )


# ---------------------------------------------------------------------------
# Pre-calibrated profiles for known hardware
# ---------------------------------------------------------------------------

# H100 SXM · flash_attention_2 · LLaMA-3.2-3B · context sweep 1K–128K
# Fitted via OverheadProfile.calibrate() on results/benchmark/context_sweep_H100_SXM.csv
# Model: t = 0.05*t_rf + 14.47ms + 46.83µs*(N/64) + 48µs*log2(N)
OVERHEAD_H100_FLASH2 = OverheadProfile(
    name="H100_SXM_flash_attention_2",
    roofline_efficiency=0.05,
    launch_overhead_s=0.01447,   # 14.5 ms fixed per step
    attn_scan_coeff=4.683e-5,    # 46.8 µs per FA tile iteration
    fa_block_size=64,
    alloc_coeff=4.8e-5,
)

# A100-40GB · SDPA · Qwen2.5-7B · 7643-token context (single point)
# SDPA tiles at ~64 tokens/block but with higher per-tile cost than FA2.
# scan_coeff estimated from residual at N=7643 after fitting launch overhead.
# launch_overhead higher than H100 because SDPA > FA2 and A100 host-side slower.
OVERHEAD_A100_SDPA = OverheadProfile(
    name="A100_40GB_sdpa",
    roofline_efficiency=0.05,
    launch_overhead_s=0.02165,   # 21.7 ms fixed per step
    attn_scan_coeff=6.37e-5,     # 63.7 µs per tile (SDPA less efficient than FA2)
    fa_block_size=64,
    alloc_coeff=4.8e-5,
)
