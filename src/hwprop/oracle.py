"""CostOracle — stateful RL interface wrapping the roofline cost model.

Usage:
    oracle = CostOracle(hardware, model, budget_s=0.05, decision_interval=64)
    oracle.reset(prompt_len=128)
    for token in model.generate():
        action = policy(oracle.observation()) if oracle.is_decision_step else None
        info = oracle.step(action)
        reward = accuracy - lam * info.budget_overshoot_frac

TODO — Eviction Ordering / Action Space Redesign (pending supervisor discussion)
================================================================================
The current action space is a flat (keep, quant, offload, disk) simplex that
redistributes tokens uniformly. Future versions should decompose the action into:

  1. **Retention target** (scalar): fraction of active tokens to keep.
  2. **Tier allocation** (simplex): how retained tokens split across
     HBM-fp16 / HBM-int8 / CPU / disk.
  3. **Eviction strategy params**: mixture weights over heuristics that
     decide *which* tokens to evict:
       - Sink tokens (attention sinks at position 0 and nearby)
       - Recent-first (sliding window)
       - Heavy-hitter (top-k by cumulative attention score)

  oracle.step() would accept an optional ``attn_scores`` tensor from the
  training loop so the selected eviction strategy can execute.

  Stretch goals:
    - Per-layer decisions (v3): each layer gets its own action vector.
    - Recomputation trade-off: spend FLOPs to recompute evicted KV instead
      of storing it, trading compute budget for memory budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from hwprop.cost_model import CostModel, KVCacheState, StepCost
from hwprop.specs import HardwareSpec, ModelConfig


# ---------------------------------------------------------------------------
# KVAction — global (keep, quant, offload, disk) policy
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KVAction:
    """Global KV cache management action applied uniformly across layers.

    keep_frac + quant_frac + offload_frac + disk_frac <= 1.0
    Remainder is evicted.
    """

    keep_frac: float       # fraction to keep in HBM full precision
    quant_frac: float      # fraction to quantize (INT8, stays in HBM)
    offload_frac: float    # fraction to offload to CPU
    disk_frac: float       # fraction to offload to disk/NVMe

    @classmethod
    def from_tensor(cls, t: np.ndarray) -> KVAction:
        """Build from a 4-float array (e.g. NN policy output).

        Clips to [0, 1] and normalises so fractions sum to <= 1.
        """
        fracs = np.clip(t[:4], 0.0, 1.0)
        total = fracs.sum()
        if total > 1.0:
            fracs = fracs / total
        return cls(
            keep_frac=float(fracs[0]),
            quant_frac=float(fracs[1]),
            offload_frac=float(fracs[2]),
            disk_frac=float(fracs[3]),
        )

    _TEXT_RE = re.compile(
        r"keep\s*=\s*([0-9.]+).*?"
        r"quant\s*=\s*([0-9.]+).*?"
        r"offload\s*=\s*([0-9.]+)"
        r"(?:.*?disk\s*=\s*([0-9.]+))?",
        re.IGNORECASE | re.DOTALL,
    )

    @classmethod
    def from_text(cls, text: str) -> KVAction:
        """Parse from LLM output, e.g. ``[KV: keep=0.8 quant=0.1 offload=0.05 disk=0.0]``.

        ``disk=`` is optional and defaults to 0.0.
        Raises ``ValueError`` if the core pattern (keep/quant/offload) is not found.
        """
        m = cls._TEXT_RE.search(text)
        if m is None:
            raise ValueError(f"Cannot parse KVAction from: {text!r}")
        disk_val = float(m.group(4)) if m.group(4) is not None else 0.0
        raw = np.array([float(m.group(1)), float(m.group(2)),
                        float(m.group(3)), disk_val])
        return cls.from_tensor(raw)


# ---------------------------------------------------------------------------
# CostInfo — return value from oracle.step()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CostInfo:
    """Rich return value from CostOracle.step()."""

    step_cost: StepCost
    budget_ok: bool                 # still within budget?
    budget_remaining_frac: float    # budget remaining / total
    budget_overshoot_frac: float    # 0.0 if within budget, (spent - budget) / budget if over
    hbm_pressure: float             # hbm_used / hbm_capacity (>1 = overflow)
    seq_position_frac: float        # current position / max_seq_len
    retention: float                # active_tokens / seq_len
    is_decision_step: bool          # was this a decision boundary?


# ---------------------------------------------------------------------------
# CostOracle — stateful step/reset with budget tracking
# ---------------------------------------------------------------------------
class CostOracle:
    """Stateful cost oracle for RL training loops.

    Tracks KV cache state, accumulates wall-clock budget, and provides
    observation vectors for the policy network.
    """

    def __init__(
        self,
        hardware: HardwareSpec,
        model: ModelConfig,
        budget_s: float,
        max_seq_len: int = 2048,
        decision_interval: int = 64,
        batch_size: int = 1,
    ) -> None:
        self.hardware = hardware
        self.model = model
        self.budget_s = budget_s
        self.max_seq_len = max_seq_len
        self.decision_interval = decision_interval
        self.batch_size = batch_size

        self._cost_model = CostModel(hardware, model)
        self._kv_state: KVCacheState = KVCacheState(0, 0, 0, 0, 0, 0)
        self._spent_s: float = 0.0
        self._step_count: int = 0

    def reset(self, prompt_len: int = 0) -> CostInfo:
        """Reset state. Optionally charges prefill cost for prompt_len tokens.

        Initializes KV state with all prompt tokens in HBM.
        """
        self._kv_state = KVCacheState(
            seq_len=prompt_len,
            tokens_in_hbm=prompt_len,
            tokens_in_hbm_quantized=0,
            tokens_in_cpu=0,
            tokens_on_disk=0,
            tokens_evicted=0,
        )
        self._step_count = 0
        self._spent_s = 0.0

        if prompt_len > 0:
            prefill = self._cost_model.prefill_cost(prompt_len, self.batch_size)
            self._spent_s = prefill.time_s
            return self._make_cost_info(prefill, is_decision_step=False)

        # Zero-cost info for empty reset
        zero_cost = StepCost(time_s=0.0, hbm_bytes=self.model.param_bytes,
                             cpu_bytes=0, disk_bytes=0, flops=0.0, hbm_overflow=False)
        return self._make_cost_info(zero_cost, is_decision_step=False)

    def step(self, action: KVAction | None = None) -> CostInfo:
        """Advance one decode token.

        If action is provided AND this is a decision step, apply it first.
        Then add new token to HBM, compute step cost, accumulate budget.
        """
        is_decision = self._step_count % self.decision_interval == 0

        # Apply action at decision boundaries
        if action is not None and is_decision:
            self._apply_action(action)

        # Add new token to HBM
        self._kv_state.seq_len += 1
        self._kv_state.tokens_in_hbm += 1

        # Compute cost
        cost = self._cost_model.step_cost(self._kv_state, self.batch_size)
        self._spent_s += cost.time_s
        self._step_count += 1

        return self._make_cost_info(cost, is_decision_step=is_decision)

    @property
    def is_decision_step(self) -> bool:
        """True if the NEXT step() call is a decision boundary."""
        return self._step_count % self.decision_interval == 0

    def observation(self) -> np.ndarray:
        """4-float vector: (budget_remaining_frac, hbm_pressure, seq_position_frac, retention).

        Append to hardware.to_tensor() for NN policy input.
        """
        return np.array([
            self._budget_remaining_frac(),
            self._hbm_pressure(),
            self._seq_position_frac(),
            self._retention(),
        ], dtype=np.float32)

    def observation_str(self) -> str:
        """Human-readable for LLM prompt insertion."""
        return (
            f"Budget: {self._budget_remaining_frac() * 100:.0f}% remaining | "
            f"HBM: {self._hbm_pressure() * 100:.0f}% full | "
            f"Position: {self._kv_state.seq_len}/{self.max_seq_len} | "
            f"Retention: {self._retention():.2f}"
        )

    @property
    def within_budget(self) -> bool:
        return self._spent_s <= self.budget_s

    @property
    def kv_state(self) -> KVCacheState:
        return self._kv_state

    @property
    def spent_s(self) -> float:
        return self._spent_s

    @property
    def step_count(self) -> int:
        return self._step_count

    # --- private helpers -------------------------------------------------------

    def _make_cost_info(self, cost: StepCost, is_decision_step: bool) -> CostInfo:
        """Build a CostInfo from a StepCost and current oracle state."""
        return CostInfo(
            step_cost=cost,
            budget_ok=self._spent_s <= self.budget_s,
            budget_remaining_frac=self._budget_remaining_frac(),
            budget_overshoot_frac=self._budget_overshoot_frac(),
            hbm_pressure=self._hbm_pressure(),
            seq_position_frac=self._seq_position_frac(),
            retention=self._retention(),
            is_decision_step=is_decision_step,
        )

    def _apply_action(self, action: KVAction) -> None:
        """Redistribute active tokens according to action fractions.

        Allocations are clamped sequentially so rounding can never
        cause the sum to exceed ``active``, which would make eviction negative.
        """
        st = self._kv_state
        active = (
            st.tokens_in_hbm
            + st.tokens_in_hbm_quantized
            + st.tokens_in_cpu
            + st.tokens_on_disk
        )

        new_hbm = min(round(active * action.keep_frac), active)
        new_quant = min(round(active * action.quant_frac), active - new_hbm)
        new_cpu = min(round(active * action.offload_frac), active - new_hbm - new_quant)
        new_disk = min(round(active * action.disk_frac), active - new_hbm - new_quant - new_cpu)
        new_evict = active - new_hbm - new_quant - new_cpu - new_disk

        st.tokens_in_hbm = new_hbm
        st.tokens_in_hbm_quantized = new_quant
        st.tokens_in_cpu = new_cpu
        st.tokens_on_disk = new_disk
        st.tokens_evicted += new_evict

    def _budget_remaining_frac(self) -> float:
        remaining = max(0.0, self.budget_s - self._spent_s)
        return remaining / self.budget_s if self.budget_s > 0 else 0.0

    def _budget_overshoot_frac(self) -> float:
        overshoot = max(0.0, self._spent_s - self.budget_s)
        return overshoot / self.budget_s if self.budget_s > 0 else 0.0

    def _hbm_pressure(self) -> float:
        kv_fp16_bytes = (
            self._kv_state.tokens_in_hbm
            * self.model.kv_bytes_per_token_per_layer()
            * self.model.num_layers
            * self.batch_size
        )
        kv_int8_bytes = (
            self._kv_state.tokens_in_hbm_quantized
            * 2 * self.model.num_kv_heads * self.model._head_dim * 1  # INT8
            * self.model.num_layers
            * self.batch_size
        )
        hbm_used = self.model.param_bytes + kv_fp16_bytes + kv_int8_bytes
        return hbm_used / self.hardware.hbm_capacity

    def _seq_position_frac(self) -> float:
        return self._kv_state.seq_len / self.max_seq_len if self.max_seq_len > 0 else 0.0

    def _retention(self) -> float:
        if self._kv_state.seq_len == 0:
            return 1.0
        return self._kv_state.active_tokens / self._kv_state.seq_len
