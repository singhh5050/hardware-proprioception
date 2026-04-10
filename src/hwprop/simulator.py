"""LLMSimulator — configurable LLM inference simulator with overhead correction.

Wraps CostModel (roofline) with OverheadProfile (calibrated corrections) and
EvictionEngine (KV cache strategy simulation) to produce accurate per-token
latency predictions for any hardware (real or theoretical) and any eviction policy.

Quick start:

    from hwprop.simulator import simulate_latency
    from hwprop.specs import get_hardware_specs, get_model_configs

    result = simulate_latency("H100_SXM", "LLaMA-3.1-8B", strategy="window_512")
    print(f"{result.mean_per_token_ms:.2f} ms/token")

    # Theoretical hardware
    from hwprop.specs import HardwareSpec, TB, TFLOPS, GB
    future = HardwareSpec(
        name="future_10x_bw",
        hbm_capacity=80 * GB,
        hbm_bandwidth=33.5 * TB,
        cpu_ram_capacity=512 * GB,
        cpu_gpu_bandwidth=128 * GB,
        fp16_flops=990 * TFLOPS,
        int8_flops=1979 * TFLOPS,
        fp32_flops=67 * TFLOPS,
        sram_capacity=50 * (1 << 20),
        interconnect_bandwidth=900 * GB,
        disk_capacity=2 * TB,
        disk_bandwidth=5 * GB,
    )
    result = simulate_latency(future, "LLaMA-3.1-8B")
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hwprop.cost_model import CostModel, KVCacheState, StepCost
from hwprop.overhead import OverheadProfile, OVERHEAD_H100_FLASH2, OVERHEAD_A100_SDPA
from hwprop.specs import HardwareSpec, ModelConfig, get_hardware_specs, get_model_configs
from hwprop.strategy import KVCacheStrategy, EvictionEngine, STRATEGY_REGISTRY, get_strategy


# ---------------------------------------------------------------------------
# SimStepCost — step cost with overhead breakdown
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimStepCost:
    """Cost of one decode step including overhead corrections."""

    time_s: float               # corrected wall-clock time
    roofline_time_s: float      # raw CostModel prediction (lower bound)

    # Overhead components (all in seconds)
    launch_overhead_s: float
    attn_scan_overhead_s: float
    alloc_overhead_s: float

    # Memory occupancy
    hbm_bytes: int
    cpu_bytes: int
    disk_bytes: int

    flops: float
    hbm_overflow: bool
    active_tokens: int

    @property
    def overhead_breakdown(self) -> dict[str, float]:
        """Fractional contribution of each component to total time."""
        if self.time_s == 0.0:
            return {"roofline": 0.0, "launch": 0.0, "attn_scan": 0.0, "alloc": 0.0}
        roofline_corrected = self.time_s - self.launch_overhead_s - self.attn_scan_overhead_s - self.alloc_overhead_s
        return {
            "roofline": roofline_corrected / self.time_s,
            "launch":   self.launch_overhead_s / self.time_s,
            "attn_scan": self.attn_scan_overhead_s / self.time_s,
            "alloc":    self.alloc_overhead_s / self.time_s,
        }


# ---------------------------------------------------------------------------
# SimResult — aggregated result for a full sequence
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """Result of simulating a full prompt + decode sequence."""

    hardware_name: str
    model_name: str
    strategy_name: str
    overhead_name: str

    prompt_len: int
    decode_steps: int

    prefill_time_s: float
    total_decode_time_s: float

    step_costs: list[SimStepCost] = field(repr=False)
    kv_states: list[KVCacheState] = field(repr=False)

    @property
    def mean_per_token_ms(self) -> float:
        if not self.step_costs:
            return 0.0
        return (self.total_decode_time_s / len(self.step_costs)) * 1000.0

    @property
    def total_time_s(self) -> float:
        return self.prefill_time_s + self.total_decode_time_s

    @property
    def overhead_breakdown(self) -> dict[str, float]:
        """Mean fractional overhead breakdown across all decode steps."""
        if not self.step_costs:
            return {"roofline": 0.0, "launch": 0.0, "attn_scan": 0.0, "alloc": 0.0}
        keys = ["roofline", "launch", "attn_scan", "alloc"]
        totals = {k: 0.0 for k in keys}
        for sc in self.step_costs:
            for k, v in sc.overhead_breakdown.items():
                totals[k] += v
        n = len(self.step_costs)
        return {k: v / n for k, v in totals.items()}

    @property
    def peak_hbm_bytes(self) -> int:
        if not self.step_costs:
            return 0
        return max(sc.hbm_bytes for sc in self.step_costs)


# ---------------------------------------------------------------------------
# LLMSimulator
# ---------------------------------------------------------------------------

class LLMSimulator:
    """Configurable LLM inference simulator.

    Wraps CostModel (pure roofline) with an OverheadProfile (calibrated
    overhead correction) and an EvictionEngine (KV cache strategy simulation).

    Designed for:
      - Real hardware: use pre-calibrated OVERHEAD_H100_FLASH2, OVERHEAD_A100_SDPA
      - Theoretical hardware: pass overhead=OverheadProfile.for_hardware(hw)
      - Strategy sweep: change strategy without reinstantiating

    Args:
        hardware:  HardwareSpec (real or theoretical).
        model:     ModelConfig.
        overhead:  OverheadProfile for overhead correction. If None, a profile
                   is derived analytically from the hardware spec.
        strategy:  KVCacheStrategy for eviction simulation. If None, full cache
                   (no eviction) is used.
        batch_size: Inference batch size (default 1).
    """

    def __init__(
        self,
        hardware: HardwareSpec,
        model: ModelConfig,
        overhead: OverheadProfile | None = None,
        strategy: KVCacheStrategy | None = None,
        batch_size: int = 1,
    ) -> None:
        self.hardware = hardware
        self.model = model
        self.overhead = overhead if overhead is not None else OverheadProfile.for_hardware(hardware)
        self.strategy = strategy if strategy is not None else KVCacheStrategy.full_cache()
        self.batch_size = batch_size

        self._cost_model = CostModel(hardware, model)
        self._engine = EvictionEngine()
        self._kv_head_layers = model.num_kv_heads * model.num_layers

    def step_cost(self, kv_state: KVCacheState) -> SimStepCost:
        """Compute corrected cost for one decode step at the given KV state.

        This is the core method. It:
          1. Calls CostModel.step_cost() to get the roofline time.
          2. Applies OverheadProfile.corrected_time() with model-aware kv_head_layers.
          3. Decomposes the overhead into its constituent parts.
        """
        raw: StepCost = self._cost_model.step_cost(kv_state, self.batch_size)
        active = kv_state.active_tokens

        import math
        t_scan  = self.overhead.attn_scan_coeff * (active / max(self.overhead.fa_block_size, 1)) * self._kv_head_layers
        t_alloc = self.overhead.alloc_coeff * math.log2(max(active, 1))
        t_total = self.overhead.corrected_time(raw.time_s, active, self._kv_head_layers)

        return SimStepCost(
            time_s=t_total,
            roofline_time_s=raw.time_s,
            launch_overhead_s=self.overhead.launch_overhead_s,
            attn_scan_overhead_s=t_scan,
            alloc_overhead_s=t_alloc,
            hbm_bytes=raw.hbm_bytes,
            cpu_bytes=raw.cpu_bytes,
            disk_bytes=raw.disk_bytes,
            flops=raw.flops,
            hbm_overflow=raw.hbm_overflow,
            active_tokens=active,
        )

    def prefill_cost(self, prompt_len: int) -> SimStepCost:
        """Prefill cost (TTFT). Overhead correction applied with prompt_len as active tokens."""
        raw: StepCost = self._cost_model.prefill_cost(prompt_len, self.batch_size)
        active = prompt_len

        import math
        t_scan  = self.overhead.attn_scan_coeff * (active / max(self.overhead.fa_block_size, 1)) * self._kv_head_layers
        t_alloc = self.overhead.alloc_coeff * math.log2(max(active, 1))
        t_total = self.overhead.corrected_time(raw.time_s, active, self._kv_head_layers)

        return SimStepCost(
            time_s=t_total,
            roofline_time_s=raw.time_s,
            launch_overhead_s=self.overhead.launch_overhead_s,
            attn_scan_overhead_s=t_scan,
            alloc_overhead_s=t_alloc,
            hbm_bytes=raw.hbm_bytes,
            cpu_bytes=raw.cpu_bytes,
            disk_bytes=raw.disk_bytes,
            flops=raw.flops,
            hbm_overflow=raw.hbm_overflow,
            active_tokens=active,
        )

    def simulate_sequence(
        self,
        prompt_len: int,
        decode_steps: int,
    ) -> SimResult:
        """Simulate a full prompt + decode sequence.

        Applies the eviction strategy at each decision_interval boundary,
        then adds the new token, then computes step cost — matching the
        semantics of eval_pipeline.compute_strategy_latency().

        Returns a SimResult with per-step costs and KV states.
        """
        has_disk = self.hardware.disk_capacity > 0

        # Prefill
        prefill = self.prefill_cost(prompt_len)

        # Initialise KV state: all prompt tokens in HBM
        kv = KVCacheState(
            seq_len=prompt_len,
            tokens_in_hbm=prompt_len,
            tokens_in_hbm_quantized=0,
            tokens_in_cpu=0,
            tokens_on_disk=0,
            tokens_evicted=0,
        )

        step_costs: list[SimStepCost] = []
        kv_states: list[KVCacheState] = []
        interval = self.strategy.decision_interval

        for step in range(decode_steps):
            # Apply eviction at decision boundaries
            if step % interval == 0:
                kv = self._engine.apply(kv, self.strategy, hardware_has_disk=has_disk)

            # Add new token to HBM
            kv.tokens_in_hbm += 1
            kv.seq_len += 1

            cost = self.step_cost(kv)
            step_costs.append(cost)
            kv_states.append(
                KVCacheState(
                    seq_len=kv.seq_len,
                    tokens_in_hbm=kv.tokens_in_hbm,
                    tokens_in_hbm_quantized=kv.tokens_in_hbm_quantized,
                    tokens_in_cpu=kv.tokens_in_cpu,
                    tokens_on_disk=kv.tokens_on_disk,
                    tokens_evicted=kv.tokens_evicted,
                )
            )

        total_decode_s = sum(sc.time_s for sc in step_costs)

        return SimResult(
            hardware_name=self.hardware.name,
            model_name=self.model.name,
            strategy_name=self.strategy.name,
            overhead_name=self.overhead.name,
            prompt_len=prompt_len,
            decode_steps=decode_steps,
            prefill_time_s=prefill.time_s,
            total_decode_time_s=total_decode_s,
            step_costs=step_costs,
            kv_states=kv_states,
        )


# ---------------------------------------------------------------------------
# simulate_latency() — convenience entry point
# ---------------------------------------------------------------------------

def simulate_latency(
    hardware: HardwareSpec | str,
    model: ModelConfig | str,
    strategy: KVCacheStrategy | str | None = None,
    prompt_len: int = 256,
    decode_steps: int = 512,
    overhead: OverheadProfile | None = None,
    batch_size: int = 1,
) -> SimResult:
    """Simulate per-token decode latency for any hardware + model + strategy.

    Args:
        hardware:     HardwareSpec or name string (e.g. "H100_SXM", "A100_40GB").
                      Pass a custom HardwareSpec for theoretical hardware exploration.
        model:        ModelConfig or name string (e.g. "LLaMA-3.1-8B", "Qwen2.5-7B").
        strategy:     KVCacheStrategy, strategy name string, or None (full cache).
                      Named strategies: "full_cache", "window_512", "snapkv_512", etc.
        prompt_len:   Length of the input prompt in tokens.
        decode_steps: Number of tokens to generate.
        overhead:     OverheadProfile for overhead correction. If None:
                        - "H100_SXM"   → OVERHEAD_H100_FLASH2
                        - "A100_40GB"  → OVERHEAD_A100_SDPA
                        - other str/HardwareSpec → derived analytically
        batch_size:   Batch size (default 1).

    Returns:
        SimResult with mean_per_token_ms, total_time_s, overhead_breakdown, etc.

    Example:
        >>> result = simulate_latency("H100_SXM", "LLaMA-3.1-8B", strategy="window_512")
        >>> print(f"{result.mean_per_token_ms:.1f} ms/token")
    """
    # Resolve hardware
    hw: HardwareSpec
    if isinstance(hardware, str):
        catalog = get_hardware_specs()
        if hardware not in catalog:
            available = ", ".join(sorted(catalog))
            raise KeyError(f"Unknown hardware {hardware!r}. Available: {available}")
        hw = catalog[hardware]
    else:
        hw = hardware

    # Resolve model
    mdl: ModelConfig
    if isinstance(model, str):
        catalog = get_model_configs()
        if model not in catalog:
            available = ", ".join(sorted(catalog))
            raise KeyError(f"Unknown model {model!r}. Available: {available}")
        mdl = catalog[model]
    else:
        mdl = model

    # Resolve strategy
    strat: KVCacheStrategy
    if strategy is None:
        strat = KVCacheStrategy.full_cache()
    elif isinstance(strategy, str):
        strat = get_strategy(strategy)
    else:
        strat = strategy

    # Resolve overhead profile
    prof: OverheadProfile
    if overhead is not None:
        prof = overhead
    elif isinstance(hardware, str) and hardware == "H100_SXM":
        prof = OVERHEAD_H100_FLASH2
    elif isinstance(hardware, str) and hardware == "A100_40GB":
        prof = OVERHEAD_A100_SDPA
    else:
        prof = OverheadProfile.for_hardware(hw)

    sim = LLMSimulator(hw, mdl, overhead=prof, strategy=strat, batch_size=batch_size)
    return sim.simulate_sequence(prompt_len=prompt_len, decode_steps=decode_steps)
