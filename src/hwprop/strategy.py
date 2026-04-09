"""KVCacheStrategy and EvictionEngine — parametric KV cache eviction as pure math.

KVCacheStrategy is a frozen dataclass that fully describes any KV cache compression
policy as a set of parameters (budget, eviction rule, quantization, tier placement).
It is the analytical counterpart to what `kvpress` press classes do with real tensors.

EvictionEngine.apply() translates a KVCacheStrategy into an updated KVCacheState,
operating on token counts rather than actual attention weights. This makes it
GPU-free and usable inside the LLMSimulator.

Supported eviction policies:
    none            — no eviction, full cache kept in HBM
    window          — keep last `window_size` tokens + `num_sink_tokens` attention sinks
    heavy_hitter    — keep top-k tokens by assumed importance (H2O-style)
    snapkv          — observation-window-based scoring (SnapKV-style)
    expected_attn   — expected attention score-based (ExpectedAttention-style)

For score-based policies (heavy_hitter, snapkv, expected_attn), no actual attention
scores are available in simulation. These are modelled as "retain the top
budget_tokens out of active_tokens" — identical in terms of token count, differing
only in which tokens are kept (irrelevant for latency simulation). The
`retention_efficiency` parameter (0–1) allows modelling sub-optimal eviction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from hwprop.cost_model import KVCacheState


EvictionPolicy = Literal["none", "window", "heavy_hitter", "snapkv", "expected_attn"]


@dataclass(frozen=True)
class KVCacheStrategy:
    """Parametric description of a KV cache compression policy.

    Designed to cover the full space of strategies used in accuracy_eval.py
    and to support novel theoretical strategies.

    Token budget:
        If budget_tokens is set, it acts as a hard cap on retained tokens.
        If budget_frac is set instead, the cap = round(seq_len * budget_frac).
        If both are set, the minimum of the two is used.
        If neither is set (budget_tokens=None, budget_frac=None), no eviction occurs.

    Quantization:
        quant_bits=8 → surviving tokens are stored as INT8 in HBM.
        quant_bits=None → full precision.

    Tier placement (fractions must sum to ≤ 1.0; remainder stays in HBM):
        hbm_frac  — fraction of retained tokens kept in HBM full-precision
        cpu_frac  — fraction offloaded to CPU DDR
        disk_frac — fraction offloaded to NVMe / disk

    Window / sink parameters (for "window" and as sink-token handling in other policies):
        num_sink_tokens  — always-retained attention sink tokens (positions 0..k)
        window_size      — sliding window size; if None, budget_tokens is used as window

    Score-based parameters (heavy_hitter, snapkv, expected_attn):
        snapkv_window    — size of observation window for score accumulation
        retention_efficiency — fraction of budget_tokens that are actually useful
                               (models the mismatch between heuristic and oracle selection)

    Decision interval:
        decision_interval — steps between eviction decisions (matches `kvpress` default of 64)
    """

    name: str
    eviction_policy: EvictionPolicy = "none"

    # Budget
    budget_tokens: int | None = None
    budget_frac: float | None = None

    # Quantization
    quant_bits: int | None = None

    # Tier placement
    hbm_frac: float = 1.0
    cpu_frac: float = 0.0
    disk_frac: float = 0.0

    # Window / sink
    num_sink_tokens: int = 4
    window_size: int | None = None

    # Score-based params
    snapkv_window: int = 32
    retention_efficiency: float = 1.0

    # Decision frequency
    decision_interval: int = 64

    def __post_init__(self) -> None:
        total_tier = self.hbm_frac + self.cpu_frac + self.disk_frac
        if total_tier > 1.0 + 1e-6:
            raise ValueError(
                f"KVCacheStrategy tier fractions sum to {total_tier:.3f} > 1.0"
            )
        if self.retention_efficiency < 0.0 or self.retention_efficiency > 1.0:
            raise ValueError("retention_efficiency must be in [0, 1]")
        if self.quant_bits is not None and self.quant_bits not in (4, 8):
            raise ValueError("quant_bits must be 4, 8, or None")

    def effective_budget(self, seq_len: int) -> int | None:
        """Resolve the effective token budget for the current sequence length."""
        budgets = []
        if self.budget_tokens is not None:
            budgets.append(self.budget_tokens)
        if self.budget_frac is not None:
            budgets.append(round(seq_len * self.budget_frac))
        if not budgets:
            return None
        return min(budgets)

    # ------------------------------------------------------------------
    # Convenience constructors matching the 12 strategies in accuracy_eval
    # ------------------------------------------------------------------

    @classmethod
    def full_cache(cls) -> "KVCacheStrategy":
        return cls(name="full_cache", eviction_policy="none")

    @classmethod
    def full_cache_int8(cls) -> "KVCacheStrategy":
        return cls(name="full_cache_int8", eviction_policy="none", quant_bits=8)

    @classmethod
    def window(cls, size: int) -> "KVCacheStrategy":
        return cls(
            name=f"window_{size}",
            eviction_policy="window",
            budget_tokens=size,
            window_size=size,
            num_sink_tokens=4,
        )

    @classmethod
    def h2o(cls, budget: int) -> "KVCacheStrategy":
        return cls(
            name=f"h2o_{budget}",
            eviction_policy="heavy_hitter",
            budget_tokens=budget,
            num_sink_tokens=4,
        )

    @classmethod
    def snapkv(cls, budget: int, observation_window: int = 32) -> "KVCacheStrategy":
        return cls(
            name=f"snapkv_{budget}",
            eviction_policy="snapkv",
            budget_tokens=budget,
            snapkv_window=observation_window,
        )

    @classmethod
    def expected_attn(cls, budget: int) -> "KVCacheStrategy":
        return cls(
            name=f"expected_attn_{budget}",
            eviction_policy="expected_attn",
            budget_tokens=budget,
        )


# ---------------------------------------------------------------------------
# EvictionEngine — pure-math KV state updates
# ---------------------------------------------------------------------------

class EvictionEngine:
    """Translates a KVCacheStrategy into KVCacheState updates.

    All methods are pure (no side effects on hardware or model state).
    The engine only deals with token *counts* — not actual attention weights.
    """

    @staticmethod
    def apply(
        kv: KVCacheState,
        strategy: KVCacheStrategy,
        hardware_has_disk: bool = True,
    ) -> KVCacheState:
        """Apply one eviction decision, returning a *new* KVCacheState.

        Called at each decision_interval boundary (the caller controls timing).
        The new token for this step is NOT yet added — that happens after eviction
        in the decode loop (matching eval_pipeline.py semantics).

        Args:
            kv: Current KV cache state (before this step's new token).
            strategy: The eviction policy to apply.
            hardware_has_disk: Whether the hardware has NVMe; if False, disk tokens
                               fall back to HBM.

        Returns:
            A new KVCacheState with eviction and tier redistribution applied.
        """
        active = kv.active_tokens

        # Determine how many tokens to retain after eviction
        retained = EvictionEngine._compute_retained(active, kv.seq_len, strategy)

        # Apply quantization or tier distribution to the retained tokens
        if strategy.quant_bits is not None:
            # All retained tokens are quantized in HBM
            new_kv = KVCacheState(
                seq_len=kv.seq_len,
                tokens_in_hbm=0,
                tokens_in_hbm_quantized=retained,
                tokens_in_cpu=0,
                tokens_on_disk=0,
                tokens_evicted=kv.tokens_evicted + (active - retained),
            )
        else:
            new_kv = EvictionEngine._distribute_tiers(
                kv, active, retained, strategy, hardware_has_disk
            )

        return new_kv

    @staticmethod
    def _compute_retained(active: int, seq_len: int, strategy: KVCacheStrategy) -> int:
        """How many tokens survive eviction?"""
        budget = strategy.effective_budget(seq_len)

        if budget is None or strategy.eviction_policy == "none":
            return active

        if strategy.eviction_policy == "window":
            # StreamingLLM / sliding window: keep sink tokens + last (window - sinks) tokens
            sinks = min(strategy.num_sink_tokens, active)
            window = strategy.window_size if strategy.window_size is not None else budget
            recent = max(0, window - sinks)
            retained = min(active, sinks + recent)
            return retained

        # Score-based policies: heavy_hitter, snapkv, expected_attn
        # Analytical model: keep top-budget_tokens, scaled by retention_efficiency
        if active <= budget:
            return active

        # retention_efficiency models how well the heuristic approximates oracle
        effective = round(budget * strategy.retention_efficiency)
        return max(1, min(effective, active))

    @staticmethod
    def _distribute_tiers(
        kv: KVCacheState,
        active: int,
        retained: int,
        strategy: KVCacheStrategy,
        hardware_has_disk: bool,
    ) -> KVCacheState:
        """Distribute retained tokens across memory tiers."""
        evicted = active - retained

        # Normalise tier fractions (they may not sum to 1 — remainder stays in HBM)
        hbm_f = strategy.hbm_frac
        cpu_f = strategy.cpu_frac
        disk_f = strategy.disk_frac if hardware_has_disk else 0.0
        # Unallocated fraction stays in HBM
        unallocated = max(0.0, 1.0 - hbm_f - cpu_f - disk_f)
        hbm_f += unallocated

        cpu_tokens = round(retained * cpu_f)
        disk_tokens = round(retained * disk_f)
        # Clamp disk to 0 if hardware has no disk
        if not hardware_has_disk:
            disk_tokens = 0
        hbm_tokens = retained - cpu_tokens - disk_tokens

        return KVCacheState(
            seq_len=kv.seq_len,
            tokens_in_hbm=max(0, hbm_tokens),
            tokens_in_hbm_quantized=0,
            tokens_in_cpu=max(0, cpu_tokens),
            tokens_on_disk=max(0, disk_tokens),
            tokens_evicted=kv.tokens_evicted + evicted,
        )


# ---------------------------------------------------------------------------
# Registry: map strategy name strings to KVCacheStrategy objects
# (mirrors the 12 strategies defined in accuracy_eval.py)
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, KVCacheStrategy] = {
    "full_cache":          KVCacheStrategy.full_cache(),
    "full_cache_int8":     KVCacheStrategy.full_cache_int8(),
    "window_128":          KVCacheStrategy.window(128),
    "window_256":          KVCacheStrategy.window(256),
    "window_512":          KVCacheStrategy.window(512),
    "window_1024":         KVCacheStrategy.window(1024),
    "h2o_128":             KVCacheStrategy.h2o(128),
    "h2o_256":             KVCacheStrategy.h2o(256),
    "h2o_512":             KVCacheStrategy.h2o(512),
    "h2o_1024":            KVCacheStrategy.h2o(1024),
    "snapkv_512":          KVCacheStrategy.snapkv(512),
    "expected_attn_512":   KVCacheStrategy.expected_attn(512),
}


def get_strategy(name: str) -> KVCacheStrategy:
    """Look up a strategy by name from the registry.

    Raises KeyError with a helpful message if not found.
    """
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY))
        raise KeyError(f"Unknown strategy {name!r}. Available: {available}")
    return STRATEGY_REGISTRY[name]
