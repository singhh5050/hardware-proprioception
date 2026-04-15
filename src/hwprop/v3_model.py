"""V3 latency model — universal bandwidth degradation with head-count correction.

A simpler alternative to the per-GPU calibrated OverheadProfile system.
Three universal constants (alpha, beta, gamma) plus one per-GPU constant (t_launch).

Model:
    t = t_launch + param_bytes / BW_hbm + total_KV_bytes / BW_eff

    BW_eff = BW_hbm / (1 + alpha × n_kv_heads^gamma × (total_KV_bytes / SRAM)^beta)

Constants:
    alpha = 0.0181   — bandwidth degradation severity
    beta  = 0.619    — degradation onset sharpness
    gamma = 1.776    — head-count scaling exponent

    Fitted on H100 SXM with LLaMA-3.2-3B (8 KV heads) + Qwen2.5-7B (4 KV heads).
    Validated on A100-80GB + Qwen2.5-7B at 1.8% MAE (out-of-sample GPU).

Per-GPU:
    t_launch — kernel launch + CUDA dispatch overhead. Measured from one short-context
    benchmark (1K tokens), or estimated from spec sheet.

    Known values:
        H100 SXM:   15.5 ms
        A100-80GB:  24.3 ms

Comparison vs LLMSimulator (calibrated per-GPU overhead profiles):
    H100 + LLaMA (in-sample):        V3=1.0%   LLMSim=1.6%
    H100 + Qwen (cross-model):       V3=1.2%   LLMSim=28.7%
    A100-80 + Qwen (cross-hw+model): V3=1.8%   LLMSim=29.5%
    H200 + LLaMA:                    V3=7.4%   LLMSim=12.2%
    RTX 5090 + LLaMA:                V3=2.8%   LLMSim=7.1%
    A100-40 + LLaMA (SDPA):          V3=19.1%  LLMSim=0.6%  ← V3 loses on SDPA

Limitations:
    - Fitted on FA2 attention only. SDPA has different overhead characteristics (19% MAE).
    - Only validated with 2 models (4 and 8 KV heads). Untested on 10, 16, or 32 KV heads.
    - t_launch is model-dependent (~2ms difference between 3B and 7B due to param load
      differences absorbed into launch). Fitting from 1K context partially captures this.
    - No batch_size support yet (all fits at batch_size=1).
"""

from __future__ import annotations

from hwprop.specs import HardwareSpec, ModelConfig

# Universal constants (fitted on H100 SXM, LLaMA-3.2-3B + Qwen2.5-7B)
ALPHA = 0.0181
BETA = 0.619
GAMMA = 1.776

# Known per-GPU launch overheads (seconds), calibrated at REF_LAYERS=28.
# Both calibration models (LLaMA-3.2-3B, Qwen2.5-7B) have 28 layers.
# For other models scale: t_launch = base * (model.num_layers / REF_LAYERS)
REF_LAYERS: int = 28

KNOWN_LAUNCH_OVERHEADS: dict[str, float] = {
    "H100_SXM": 0.01547,   # 15.5 ms — from H100 + LLaMA-3.2-3B fit
    "A100_80GB": 0.02430,  # 24.3 ms — from A100-80GB + Qwen2.5-7B 1K-context
}


def effective_bandwidth(
    hw: HardwareSpec,
    model: ModelConfig,
    context_len: int,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> float:
    """Compute effective HBM bandwidth at a given context length.

    Returns bytes/s. Always <= hw.hbm_bandwidth.
    """
    kv_per_token_per_layer = model.kv_bytes_per_token_per_layer()
    total_kv = context_len * kv_per_token_per_layer * model.num_layers
    sram = hw.sram_capacity

    ratio = total_kv / sram if sram > 0 else 0
    alpha_adj = alpha * (model.num_kv_heads ** gamma)

    return hw.hbm_bandwidth / (1 + alpha_adj * ratio ** beta)


def predict_step_ms(
    hw: HardwareSpec,
    model: ModelConfig,
    context_len: int,
    t_launch: float | None = None,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> float:
    """Predict decode step latency in milliseconds.

    Args:
        hw:          Hardware spec.
        model:       Model config.
        context_len: Current context length (tokens in KV cache).
        t_launch:    Per-step launch overhead in seconds. If None, looks up
                     from KNOWN_LAUNCH_OVERHEADS or estimates as 20ms.
        alpha, beta, gamma: Model constants (default: fitted values).

    Returns:
        Predicted wall-clock time per decode step in ms.
    """
    if t_launch is None:
        t_launch = KNOWN_LAUNCH_OVERHEADS.get(hw.name, 0.020)

    bw_eff = effective_bandwidth(hw, model, context_len, alpha, beta, gamma)

    kv_per_token_per_layer = model.kv_bytes_per_token_per_layer()
    total_kv = context_len * kv_per_token_per_layer * model.num_layers

    t_param = model.param_bytes / hw.hbm_bandwidth
    t_kv = total_kv / bw_eff if bw_eff > 0 else 0

    return (t_launch + t_param + t_kv) * 1000


def fit_launch_overhead(
    hw: HardwareSpec,
    model: ModelConfig,
    measured_ms_at_short_context: float,
    short_context_len: int = 1024,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> float:
    """Fit t_launch from a single short-context measurement.

    At short context, KV time is small, so:
        measured ≈ t_launch + t_param + t_kv_small
        t_launch ≈ measured - t_param - t_kv_small

    Args:
        hw:          Hardware spec.
        model:       Model config.
        measured_ms_at_short_context: Measured ms/step at short_context_len.
        short_context_len: Context length of the measurement (default 1024).

    Returns:
        Estimated t_launch in seconds.
    """
    bw_eff = effective_bandwidth(hw, model, short_context_len, alpha, beta, gamma)
    kv_per_token_per_layer = model.kv_bytes_per_token_per_layer()
    total_kv = short_context_len * kv_per_token_per_layer * model.num_layers

    t_param = model.param_bytes / hw.hbm_bandwidth
    t_kv = total_kv / bw_eff if bw_eff > 0 else 0

    return measured_ms_at_short_context / 1000 - t_param - t_kv
