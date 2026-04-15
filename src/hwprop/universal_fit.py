"""Universal physics-informed latency equation (2 constants).

t = t_launch(gpu) + param_bytes/BW + KV_bytes / BW_eff
BW_eff = BW / (1 + ALPHA * (KV_bytes / L2_capacity) ^ BETA)

Fitted on 318 context-sweep data points (9 models x 5 GPUs).

Performance:
    Full-fit MAE:   20.9%   (Spearman rho=0.746)
    LOO-GPU MAE:    31.8%   (predict new GPU from spec sheet + 1 measurement)
    LOO-Model MAE:  20.0%   (predict new model from config.json)

NOT RL-ready (rho=0.746, need >0.95). Use lookup_table.py for RL.
This module is useful for rough extrapolation to unseen hardware/models.

Worst predictions:
    LLaMA-3.2-1B:  57% MAE  (tiny model, launch-overhead dominated)
    Gemma-3-1B:    36% MAE  (tiny model, unusual arch)
    A40 LOO-GPU:   59% MAE  (6MB L2 extreme not capturable from other GPUs)
"""

ALPHA = 0.00163093
BETA = 1.0408

# Per-GPU launch overhead (seconds), fit from shortest-context measurements
LAUNCH_OVERHEADS = {
    "H200":      0.0198,
    "H100_SXM":  0.0162,
    "A100_80GB":  0.0274,
    "L40S":      0.0233,
    "A40":       0.0203,
}


def predict_step_ms(
    hbm_bandwidth: float,
    l2_capacity: int,
    param_bytes: int,
    kv_bytes_per_token: int,
    context_length: int,
    t_launch: float = 0.020,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> float:
    """Predict decode step latency in ms.

    Args:
        hbm_bandwidth:     HBM bandwidth in bytes/s (from specs.py)
        l2_capacity:       L2/SRAM capacity in bytes (from specs.py)
        param_bytes:       Model parameter bytes (bf16)
        kv_bytes_per_token: KV cache bytes per token (all layers)
        context_length:    Current KV cache size in tokens
        t_launch:          Per-step launch overhead in seconds
        alpha, beta:       Universal degradation constants
    """
    total_kv = kv_bytes_per_token * context_length
    ratio = total_kv / l2_capacity if l2_capacity > 0 else 0
    bw_eff = hbm_bandwidth / (1 + alpha * ratio ** beta)

    t_param = param_bytes / hbm_bandwidth
    t_kv = total_kv / bw_eff if bw_eff > 0 else 0

    return (t_launch + t_param + t_kv) * 1000


def fit_launch_from_measurement(
    hbm_bandwidth: float,
    l2_capacity: int,
    param_bytes: int,
    kv_bytes_per_token: int,
    context_length: int,
    measured_ms: float,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> float:
    """Fit t_launch from a single short-context measurement.

    Run one benchmark at short context (e.g., 512 tokens),
    then call this to get t_launch for that GPU.
    """
    total_kv = kv_bytes_per_token * context_length
    ratio = total_kv / l2_capacity if l2_capacity > 0 else 0
    bw_eff = hbm_bandwidth / (1 + alpha * ratio ** beta)

    t_param = param_bytes / hbm_bandwidth
    t_kv = total_kv / bw_eff if bw_eff > 0 else 0

    return max(0.001, measured_ms / 1000 - t_param - t_kv)
