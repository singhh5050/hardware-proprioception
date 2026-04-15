"""Lookup-table cost model built from real benchmark data.

Provides interpolated latency predictions for any (gpu, model, cache_size)
triple using measured data from the benchmark grid sweep. No analytical model,
no curve fitting — just linear interpolation between measured points.

Usage:
    from hwprop.lookup_table import LookupCostModel

    cost_model = LookupCostModel.from_grid("results/grid")

    # RL environment: agent evicts tokens, get new latency
    latency_ms = cost_model.step_cost("H200", "meta-llama/Llama-3.2-3B", cache_size=16000)

    # Batch-aware cost
    latency_ms = cost_model.step_cost("H200", "meta-llama/Llama-3.2-3B", cache_size=8192, batch_size=4)

    # What's available?
    print(cost_model.gpus)
    print(cost_model.models)
    print(cost_model.summary())
"""

from __future__ import annotations

import json
import glob
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CurveData:
    """Measured latency-vs-context curve for one (gpu, model) pair."""
    context_lengths: np.ndarray  # sorted ascending
    latencies_ms: np.ndarray     # ms per decode step, same order
    stdevs_ms: np.ndarray        # standard deviation, same order

    def interpolate(self, cache_size: int) -> float:
        """Linearly interpolate latency at an arbitrary cache size."""
        return float(np.interp(cache_size, self.context_lengths, self.latencies_ms))

    @property
    def is_monotonic(self) -> bool:
        """Check if latency is non-decreasing with context length."""
        return bool(np.all(np.diff(self.latencies_ms) >= -0.01))


@dataclass
class StrategyCost:
    """Measured strategy latency at specific context lengths."""
    context_lengths: np.ndarray
    strategies: dict[str, np.ndarray]  # strategy_name -> latencies_ms


@dataclass
class BatchCurve:
    """Measured latency-vs-batch-size curve at a fixed context length."""
    batch_sizes: np.ndarray
    latencies_ms_per_step: np.ndarray
    throughputs_tok_per_sec: np.ndarray
    context_length: int

    def interpolate_throughput(self, batch_size: int) -> float:
        """Linearly interpolate throughput at an arbitrary batch size."""
        return float(np.interp(batch_size, self.batch_sizes, self.throughputs_tok_per_sec))


class LookupCostModel:
    """Interpolated cost model from benchmark grid data.

    The core lookup table maps (gpu, model, context_length) -> ms_per_step
    using linear interpolation between measured context sweep points.

    Additionally stores strategy sweep data (for eviction overhead estimation)
    and batch sweep data (for throughput modeling).
    """

    def __init__(self):
        # Core context curves: (gpu, model) -> CurveData
        self._context_curves: dict[tuple[str, str], CurveData] = {}
        # Strategy data: (gpu, model) -> StrategyCost
        self._strategy_data: dict[tuple[str, str], StrategyCost] = {}
        # Batch data: (gpu, model) -> BatchCurve
        self._batch_data: dict[tuple[str, str], BatchCurve] = {}
        # Raw rows for inspection
        self._raw_rows: list[dict] = []

    @classmethod
    def from_grid(cls, grid_dir: str = "results/grid") -> "LookupCostModel":
        """Load from the benchmark grid directory structure.

        Expected layout:
            results/grid/<model_short>/<gpu_key>.jsonl
        """
        model = cls()

        # Load all JSONL files
        for jsonl_path in sorted(glob.glob(os.path.join(grid_dir, "*", "*.jsonl"))):
            if "checkpoint" in jsonl_path:
                continue
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        model._raw_rows.append(json.loads(line))

        if not model._raw_rows:
            raise FileNotFoundError(f"No JSONL data found in {grid_dir}/")

        # Build context curves (full_cache, bs=1)
        context_rows = [r for r in model._raw_rows
                        if r["strategy"] == "full_cache" and r["batch_size"] == 1]

        pairs = set((r["hardware_key"], r["model_name"]) for r in context_rows)
        for gpu, model_name in sorted(pairs):
            pair_rows = sorted(
                [r for r in context_rows
                 if r["hardware_key"] == gpu and r["model_name"] == model_name],
                key=lambda r: r["context_length"]
            )
            # Deduplicate by context_length (keep first occurrence)
            seen = set()
            deduped = []
            for r in pair_rows:
                if r["context_length"] not in seen:
                    seen.add(r["context_length"])
                    deduped.append(r)

            if len(deduped) >= 2:
                model._context_curves[(gpu, model_name)] = CurveData(
                    context_lengths=np.array([r["context_length"] for r in deduped]),
                    latencies_ms=np.array([r["mean_ms_per_step"] for r in deduped]),
                    stdevs_ms=np.array([r["std_ms_per_step"] for r in deduped]),
                )

        # Build strategy data (bs=1, non-full_cache)
        strat_rows = [r for r in model._raw_rows if r["batch_size"] == 1]
        strat_pairs = set((r["hardware_key"], r["model_name"]) for r in strat_rows)
        for gpu, model_name in sorted(strat_pairs):
            pair_rows = [r for r in strat_rows
                         if r["hardware_key"] == gpu and r["model_name"] == model_name]
            strategies_at_ctx: dict[int, dict[str, float]] = {}
            for r in pair_rows:
                ctx = r["context_length"]
                strat = r["strategy"]
                if ctx not in strategies_at_ctx:
                    strategies_at_ctx[ctx] = {}
                if strat not in strategies_at_ctx[ctx]:
                    strategies_at_ctx[ctx][strat] = r["mean_ms_per_step"]

            # Only keep context lengths where we have multiple strategies
            multi_strat_ctxs = {ctx: strats for ctx, strats in strategies_at_ctx.items()
                                if len(strats) > 1}
            if multi_strat_ctxs:
                all_strat_names = set()
                for strats in multi_strat_ctxs.values():
                    all_strat_names.update(strats.keys())

                ctx_list = sorted(multi_strat_ctxs.keys())
                strat_arrays = {}
                for sname in sorted(all_strat_names):
                    strat_arrays[sname] = np.array([
                        multi_strat_ctxs[ctx].get(sname, np.nan)
                        for ctx in ctx_list
                    ])

                model._strategy_data[(gpu, model_name)] = StrategyCost(
                    context_lengths=np.array(ctx_list),
                    strategies=strat_arrays,
                )

        # Build batch data (full_cache, context=8192)
        batch_rows = [r for r in model._raw_rows
                      if r["strategy"] == "full_cache" and r["batch_size"] > 1]
        batch_pairs = set((r["hardware_key"], r["model_name"]) for r in batch_rows)
        for gpu, model_name in sorted(batch_pairs):
            pair_rows = sorted(
                [r for r in batch_rows
                 if r["hardware_key"] == gpu and r["model_name"] == model_name],
                key=lambda r: r["batch_size"]
            )
            if pair_rows:
                ctx = pair_rows[0]["context_length"]
                # Add bs=1 data point from context curves
                bs1_rows = [r for r in model._raw_rows
                            if r["hardware_key"] == gpu and r["model_name"] == model_name
                            and r["strategy"] == "full_cache" and r["batch_size"] == 1
                            and r["context_length"] == ctx]
                all_batch_rows = sorted(bs1_rows + pair_rows, key=lambda r: r["batch_size"])
                # Deduplicate by batch_size
                seen = set()
                deduped = []
                for r in all_batch_rows:
                    if r["batch_size"] not in seen:
                        seen.add(r["batch_size"])
                        deduped.append(r)

                model._batch_data[(gpu, model_name)] = BatchCurve(
                    batch_sizes=np.array([r["batch_size"] for r in deduped]),
                    latencies_ms_per_step=np.array([r["mean_ms_per_step"] for r in deduped]),
                    throughputs_tok_per_sec=np.array([r["throughput_tok_per_sec"] for r in deduped]),
                    context_length=ctx,
                )

        return model

    # -----------------------------------------------------------------
    # Core API
    # -----------------------------------------------------------------

    def step_cost(self, gpu: str, model_name: str, cache_size: int,
                  batch_size: int = 1) -> float:
        """Predict decode step latency in ms.

        Args:
            gpu:        Hardware key (e.g., "H200", "H100_SXM", "A100_80GB")
            model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.2-3B")
            cache_size: Current KV cache size in tokens
            batch_size: Batch size (default 1). If >1 and batch data exists,
                        adjusts based on measured batch scaling.

        Returns:
            Estimated ms per decode step.
        """
        key = (gpu, model_name)
        if key not in self._context_curves:
            # Try matching by model short name
            for (g, m), curve in self._context_curves.items():
                if g == gpu and model_name in m:
                    key = (g, m)
                    break
            else:
                available = [(g, m.split("/")[-1]) for g, m in self._context_curves.keys()
                             if g == gpu]
                raise KeyError(
                    f"No data for ({gpu}, {model_name}). "
                    f"Available on {gpu}: {[m for _, m in available]}"
                )

        base_ms = self._context_curves[key].interpolate(cache_size)

        if batch_size > 1:
            batch_key = key
            if batch_key in self._batch_data:
                bd = self._batch_data[batch_key]
                # Get bs=1 baseline and interpolated batch latency
                bs1_ms = float(np.interp(1, bd.batch_sizes, bd.latencies_ms_per_step))
                bs_n_ms = float(np.interp(batch_size, bd.batch_sizes, bd.latencies_ms_per_step))
                # Apply the batch scaling ratio to the context-interpolated base
                if bs1_ms > 0:
                    base_ms = base_ms * (bs_n_ms / bs1_ms)

        return base_ms

    def strategy_cost(self, gpu: str, model_name: str, context_length: int,
                      strategy: str) -> float | None:
        """Get measured strategy latency, or None if not available.

        Returns exact measurement if context_length matches, otherwise None.
        Use step_cost() for the interpolated full_cache baseline and compare.
        """
        key = self._resolve_key(gpu, model_name)
        if key not in self._strategy_data:
            return None

        sd = self._strategy_data[key]
        if strategy not in sd.strategies:
            return None

        idx = np.where(sd.context_lengths == context_length)[0]
        if len(idx) == 0:
            return None

        val = sd.strategies[strategy][idx[0]]
        return float(val) if not np.isnan(val) else None

    def eviction_overhead(self, gpu: str, model_name: str,
                          context_length: int, strategy: str) -> float | None:
        """Compute eviction overhead = strategy_latency - full_cache_latency.

        Returns the overhead in ms, or None if data not available.
        Positive = eviction is slower (overhead > savings at this context).
        Negative = eviction is faster (savings > overhead).
        """
        strat_ms = self.strategy_cost(gpu, model_name, context_length, strategy)
        fc_ms = self.strategy_cost(gpu, model_name, context_length, "full_cache")
        if strat_ms is None or fc_ms is None:
            return None
        return strat_ms - fc_ms

    # -----------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------

    @property
    def gpus(self) -> list[str]:
        """List of available GPU keys."""
        return sorted(set(g for g, _ in self._context_curves.keys()))

    @property
    def models(self) -> list[str]:
        """List of available model names."""
        return sorted(set(m for _, m in self._context_curves.keys()))

    def context_curve(self, gpu: str, model_name: str) -> CurveData | None:
        """Get the raw context curve for a (gpu, model) pair."""
        key = self._resolve_key(gpu, model_name)
        return self._context_curves.get(key)

    def batch_curve(self, gpu: str, model_name: str) -> BatchCurve | None:
        """Get the raw batch curve for a (gpu, model) pair."""
        key = self._resolve_key(gpu, model_name)
        return self._batch_data.get(key)

    def summary(self) -> str:
        """Human-readable summary of the lookup table."""
        lines = [
            f"LookupCostModel: {len(self._raw_rows)} rows",
            f"  GPUs: {self.gpus}",
            f"  Models: {[m.split('/')[-1] for m in self.models]}",
            f"  Context curves: {len(self._context_curves)}",
            f"  Strategy data: {len(self._strategy_data)}",
            f"  Batch data: {len(self._batch_data)}",
        ]

        # Monotonicity check
        non_mono = []
        for (gpu, model), curve in self._context_curves.items():
            if not curve.is_monotonic:
                non_mono.append(f"{gpu}/{model.split('/')[-1]}")
        if non_mono:
            lines.append(f"  Non-monotonic curves: {len(non_mono)} ({', '.join(non_mono[:5])}...)")
        else:
            lines.append(f"  All context curves monotonic: YES")

        return "\n".join(lines)

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    def _resolve_key(self, gpu: str, model_name: str) -> tuple[str, str]:
        """Resolve a (gpu, model) key, trying short name matching."""
        key = (gpu, model_name)
        if key in self._context_curves:
            return key
        for (g, m) in self._context_curves.keys():
            if g == gpu and (model_name in m or m.split("/")[-1] == model_name):
                return (g, m)
        return key
