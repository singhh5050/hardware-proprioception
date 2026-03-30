"""Latency replay helpers for post-hoc simulation of KV-cache strategies."""

from __future__ import annotations

from hwprop.specs import HardwareSpec, ModelConfig


def strategy_to_kv_update(
    strategy_name: str,
    budget_tokens: int | None,
    active_tokens: int,
    quantized: bool = False,
) -> dict:
    """Pure function: given a strategy and current token count, return post-eviction state.

    Returns dict with keys: tokens_kept, tokens_evicted, is_quantized.
    """
    if strategy_name == "full_cache":
        return {"tokens_kept": active_tokens, "tokens_evicted": 0, "is_quantized": False}

    if strategy_name in ("full_cache_int4", "full_cache_int8") or quantized:
        return {"tokens_kept": active_tokens, "tokens_evicted": 0, "is_quantized": True}

    # Eviction strategies (window, h2o, snapkv, expected_attn)
    if budget_tokens is not None and active_tokens > budget_tokens:
        return {
            "tokens_kept": budget_tokens,
            "tokens_evicted": active_tokens - budget_tokens,
            "is_quantized": False,
        }
    return {"tokens_kept": active_tokens, "tokens_evicted": 0, "is_quantized": False}


def compute_strategy_latency(
    strategy_name: str,
    budget_tokens: int | None,
    hardware: HardwareSpec,
    model_config: ModelConfig,
    prompt_len: int,
    decode_steps: int,
    decision_interval: int = 64,
    offload_frac: float = 0.0,
    disk_frac: float = 0.0,
    quantized: bool = False,
    batch_size: int = 1,
) -> dict:
    """Simulate decode latency for a strategy on specific hardware.

    Steps through decode, applying eviction at decision boundaries and
    distributing surviving tokens across memory tiers per offload split.
    Uses CostModel directly for fine-grained control.
    """
    from hwprop.cost_model import CostModel, KVCacheState

    cost_model = CostModel(hardware, model_config)

    # Prefill cost
    prefill_cost = cost_model.prefill_cost(prompt_len, batch_size=batch_size)

    # Initialize KV state: all prompt tokens in HBM
    kv = KVCacheState(
        seq_len=prompt_len,
        tokens_in_hbm=prompt_len,
        tokens_in_hbm_quantized=0,
        tokens_in_cpu=0,
        tokens_on_disk=0,
        tokens_evicted=0,
    )

    step_times = []
    for step in range(decode_steps):
        # At decision boundaries, apply eviction and redistribution
        if step % decision_interval == 0:
            active = kv.active_tokens
            update = strategy_to_kv_update(strategy_name, budget_tokens, active, quantized)

            kept = update["tokens_kept"]
            is_quant = update["is_quantized"]

            if is_quant:
                # All tokens quantized in HBM
                kv.tokens_in_hbm = 0
                kv.tokens_in_hbm_quantized = kept
                kv.tokens_in_cpu = 0
                kv.tokens_on_disk = 0
            else:
                # Distribute kept tokens across tiers
                hbm_tokens = int(kept * (1.0 - offload_frac - disk_frac))
                cpu_tokens = int(kept * offload_frac)
                disk_tokens = kept - hbm_tokens - cpu_tokens
                # Clamp disk to 0 if hardware has no disk
                if hardware.disk_capacity == 0:
                    hbm_tokens += disk_tokens
                    disk_tokens = 0
                kv.tokens_in_hbm = hbm_tokens
                kv.tokens_in_hbm_quantized = 0
                kv.tokens_in_cpu = cpu_tokens
                kv.tokens_on_disk = disk_tokens

            kv.tokens_evicted += update["tokens_evicted"]

        # Add new token to HBM
        kv.seq_len += 1
        kv.tokens_in_hbm += 1

        # Compute step cost
        cost = cost_model.step_cost(kv, batch_size=batch_size)
        step_times.append(cost.time_s)

    total_decode_s = sum(step_times)
    mean_latency_ms = (total_decode_s / decode_steps * 1000) if decode_steps > 0 else 0

    return {
        "strategy": strategy_name,
        "hardware": hardware.name if hasattr(hardware, "name") else "unknown",
        "offload_frac": offload_frac,
        "disk_frac": disk_frac,
        "mean_latency_ms": mean_latency_ms,
        "total_time_s": prefill_cost.time_s + total_decode_s,
        "prefill_time_s": prefill_cost.time_s,
    }


def compute_latency_sweep(
    strategies_with_budgets: list[dict],
    hardware_configs: dict[str, HardwareSpec],
    model_config: ModelConfig,
    prompt_len: int = 256,
    decode_steps: int = 512,
    decision_interval: int = 64,
    offload_splits: list[tuple[float, float, float]] | None = None,
) -> list[dict]:
    """Cartesian product: strategy x hardware x offload_split.

    Each entry in strategies_with_budgets should have keys:
        strategy (str), budget_tokens (int|None), quantized (bool, optional)

    offload_splits: list of (hbm_frac, cpu_frac, disk_frac) tuples.
    """
    if offload_splits is None:
        offload_splits = [
            (1.0, 0.0, 0.0),
            (0.7, 0.3, 0.0),
            (0.5, 0.5, 0.0),
            (0.3, 0.3, 0.4),
            (0.5, 0.0, 0.5),
        ]

    results: list[dict] = []

    for strat in strategies_with_budgets:
        strat_name = strat["strategy"]
        budget = strat.get("budget_tokens")
        quantized = strat.get("quantized", False)

        for hw_name, hw in hardware_configs.items():
            for hbm_f, cpu_f, disk_f in offload_splits:
                # Skip disk splits on hardware with no disk
                if disk_f > 0 and hw.disk_capacity == 0:
                    continue

                result = compute_strategy_latency(
                    strategy_name=strat_name,
                    budget_tokens=budget,
                    hardware=hw,
                    model_config=model_config,
                    prompt_len=prompt_len,
                    decode_steps=decode_steps,
                    decision_interval=decision_interval,
                    offload_frac=cpu_f,
                    disk_frac=disk_f,
                    quantized=quantized,
                )
                results.append(result)

    return results
