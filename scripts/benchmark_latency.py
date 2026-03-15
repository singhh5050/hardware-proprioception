#!/usr/bin/env python3
"""Benchmark wall-clock latency per eviction strategy and compare to simulation.

Measures GPU time using timeit.default_timer() + torch.cuda.synchronize()
(per CS336 assignment2 timing pattern), then compares against the roofline
cost model from compute_strategy_latency().

Attention implementation selection:
  - Non-H2O strategies: flash_attention_2 if available, else sdpa, else eager
  - H2O strategies: always eager (ObservedAttentionPress requires materialised
    attention weights, which only exist with eager)
  - If a run mixes both groups, the model is loaded twice sequentially so each
    group uses its optimal implementation without doubling VRAM at the same time.

Usage:
    # Quick run: 3 strategies, 2 warmup, 3 repeats
    python benchmark_latency.py --strategies full_cache,window_512,snapkv_512

    # Full run with all strategies (incl. H2O)
    python benchmark_latency.py --include-h2o --warmup 3 --repeats 5

    # Override hardware for simulation (default: auto-detected from GPU name)
    python benchmark_latency.py --hardware H100_SXM
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import timeit

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))


# ---------------------------------------------------------------------------
# Hardware auto-detection
# ---------------------------------------------------------------------------
_GPU_NAME_TO_HW_KEY = [
    ("h200",  "H200"),
    ("h100",  "H100_SXM"),
    ("b200",  "B200"),
    ("a100-sxm4-80", "A100_80GB"),
    ("a100-sxm4-40", "A100_40GB"),
    ("a100",  "A100_40GB"),  # default to 40GB; use --hardware A100_80GB if needed
    ("l40",   "L40S"),
    ("mi350", "MI350X"),
    ("mi325", "MI325X"),
    ("mi300", "MI300X"),
    ("gaudi", "Gaudi_3"),
]


def detect_hardware_key() -> str:
    """Map torch GPU device name to our hardware catalog key."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "H100_SXM"
        name = torch.cuda.get_device_name(0).lower()
        for fragment, key in _GPU_NAME_TO_HW_KEY:
            if fragment in name:
                return key
        print(f"  [warn] Unknown GPU '{name}', defaulting to H100_SXM for simulation")
    except Exception:
        pass
    return "H100_SXM"


# ---------------------------------------------------------------------------
# Attention implementation detection
# ---------------------------------------------------------------------------
def detect_best_attn_impl(model_name: str, dtype, device_map: str) -> str:
    """Return the best available attn_implementation for this environment.

    Priority: flash_attention_2 > sdpa > eager

    flash_attention_2 requires:
      - flash-attn package installed
      - CUDA GPU with compute capability >= 8.0 (Ampere+)
      - transformers >= 4.36
    sdpa is always available via PyTorch >= 2.0 (uses memory-efficient attention).
    """
    import torch

    # Check flash-attn package
    try:
        import flash_attn  # noqa: F401
        # Check compute capability — flash-attn needs Ampere (8.0) or newer
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            if major >= 8:
                # Do a quick probe load to confirm transformers accepts it
                try:
                    from transformers import AutoConfig
                    cfg = AutoConfig.from_pretrained(model_name)
                    if hasattr(cfg, "model_type"):
                        # Most recent transformer models support FA2
                        return "flash_attention_2"
                except Exception:
                    pass
        print("  [attn] flash-attn installed but GPU compute capability < 8.0 — falling back to sdpa")
    except ImportError:
        print("  [attn] flash-attn not installed — falling back to sdpa")

    # sdpa is the PyTorch built-in fused attention (always available >= torch 2.0)
    return "sdpa"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_name: str, attn_impl: str, dtype, device_map: str):
    """Load model with the specified attention implementation."""
    from transformers import AutoModelForCausalLM
    import torch

    print(f"  Loading with attn_implementation='{attn_impl}'")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_impl,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Timed generation
# ---------------------------------------------------------------------------
def run_timed_generation(
    model,
    input_ids,
    strategy_config,
    decode_steps: int,
    warmup: int,
    repeats: int,
) -> tuple[float, float, float]:
    """Run generation with proper GPU timing.

    Pattern from CS336 assignment2:
    - W warmup runs (discarded — handles JIT, caching, cuDNN autotuning)
    - N timed runs: synchronize → start timer → generate → synchronize → stop

    Returns (mean_total_s, stdev_total_s, peak_memory_mb).
    """
    import torch

    def _generate():
        if strategy_config.press_factory is not None:
            press = strategy_config.press_factory()
            with press(model):
                return model.generate(
                    input_ids,
                    max_new_tokens=decode_steps,
                    do_sample=False,
                )
        elif strategy_config.quantized:
            return model.generate(
                input_ids,
                max_new_tokens=decode_steps,
                do_sample=False,
                cache_implementation="quantized",
                cache_config={"nbits": 8, "backend": "hqq", "residual_length": 128},
            )
        else:
            return model.generate(
                input_ids,
                max_new_tokens=decode_steps,
                do_sample=False,
            )

    # Warm-up runs — not timed
    for i in range(warmup):
        with torch.no_grad():
            _generate()
        torch.cuda.synchronize()
        print(f"    warmup {i + 1}/{warmup} done")

    # Reset peak memory before timed runs
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Timed runs
    times = []
    for i in range(repeats):
        torch.cuda.synchronize()          # flush pending GPU work before starting
        t0 = timeit.default_timer()
        with torch.no_grad():
            _generate()
        torch.cuda.synchronize()          # wait for GPU to finish before stopping
        elapsed = timeit.default_timer() - t0
        times.append(elapsed)
        print(f"    run {i + 1}/{repeats}: {elapsed * 1000:.1f} ms")

    peak_mb = 0.0
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    mean_s = statistics.mean(times)
    stdev_s = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean_s, stdev_s, peak_mb


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def run_simulation(
    strategy_name: str,
    budget_tokens: int | None,
    quantized: bool,
    hardware_key: str,
    model_key: str,
    prompt_len: int,
    decode_steps: int,
) -> dict:
    from hwprop.eval_pipeline import compute_strategy_latency
    from hwprop.specs import get_hardware_specs, get_model_configs

    hw = get_hardware_specs()[hardware_key]
    model_cfg = get_model_configs()[model_key]

    return compute_strategy_latency(
        strategy_name=strategy_name,
        budget_tokens=budget_tokens,
        hardware=hw,
        model_config=model_cfg,
        prompt_len=prompt_len,
        decode_steps=decode_steps,
        quantized=quantized,
    )


# ---------------------------------------------------------------------------
# Benchmark one group of strategies with a shared model
# ---------------------------------------------------------------------------
def benchmark_group(
    strategies: dict,
    model,
    attn_impl: str,
    input_ids,
    actual_prompt_len: int,
    hw_key: str,
    model_config_key: str,
    decode_steps: int,
    warmup: int,
    repeats: int,
) -> list[dict]:
    rows = []
    for strat_name, strat_cfg in strategies.items():
        print(f"\n{'=' * 60}")
        print(f"Strategy: {strat_name}  [{attn_impl}]  ({strat_cfg.description})")

        try:
            mean_s, stdev_s, peak_mb = run_timed_generation(
                model=model,
                input_ids=input_ids,
                strategy_config=strat_cfg,
                decode_steps=decode_steps,
                warmup=warmup,
                repeats=repeats,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        measured_per_token_ms = (mean_s / decode_steps) * 1000

        sim = run_simulation(
            strategy_name=strat_name,
            budget_tokens=strat_cfg.budget_tokens,
            quantized=strat_cfg.quantized,
            hardware_key=hw_key,
            model_key=model_config_key,
            prompt_len=actual_prompt_len,
            decode_steps=decode_steps,
        )
        simulated_total_s = sim["total_time_s"]
        simulated_per_token_ms = sim["mean_latency_ms"]
        ratio = (
            measured_per_token_ms / simulated_per_token_ms
            if simulated_per_token_ms > 0
            else float("nan")
        )

        print(f"  Measured:   {mean_s * 1000:.1f} ms total  |  {measured_per_token_ms:.2f} ms/token  (±{stdev_s * 1000:.1f} ms)")
        print(f"  Simulated:  {simulated_total_s * 1000:.1f} ms total  |  {simulated_per_token_ms:.2f} ms/token")
        print(f"  Ratio (measured/simulated): {ratio:.2f}x")
        print(f"  Peak memory: {peak_mb:.0f} MB")

        rows.append({
            "strategy": strat_name,
            "attn_impl": attn_impl,
            "prompt_len": actual_prompt_len,
            "decode_steps": decode_steps,
            "warmup_runs": warmup,
            "timed_runs": repeats,
            "measured_total_ms": round(mean_s * 1000, 2),
            "measured_stdev_ms": round(stdev_s * 1000, 2),
            "measured_per_token_ms": round(measured_per_token_ms, 3),
            "simulated_total_ms": round(simulated_total_s * 1000, 2),
            "simulated_per_token_ms": round(simulated_per_token_ms, 3),
            "ratio_measured_over_simulated": round(ratio, 3),
            "peak_memory_mb": round(peak_mb, 1),
            "hardware_key": hw_key,
            "model_config_key": model_config_key,
        })
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark wall-clock vs simulated latency per eviction strategy"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Math-7B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2.5-Math-7B-Instruct)",
    )
    parser.add_argument(
        "--model-config-key",
        default="Qwen2.5-7B",
        help="hwprop model config key for simulation (default: Qwen2.5-7B)",
    )
    parser.add_argument(
        "--hardware",
        default=None,
        help="Hardware catalog key for simulation (default: auto-detect from GPU)",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=256,
        help="Prompt length in tokens (default: 256)",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=128,
        help="Tokens to generate per run (default: 128)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup runs before timing (default: 2)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Timed runs to average (default: 3)",
    )
    parser.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated strategy names (default: all non-H2O)",
    )
    parser.add_argument(
        "--include-h2o",
        action="store_true",
        help="Include H2O strategies (benchmarked separately under eager attention)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_outputs",
        help="Output directory (default: benchmark_outputs)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device map for model loading (default: auto)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    from transformers import AutoTokenizer
    from hwprop.accuracy_eval import get_strategies

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required for benchmarking.")
        return 1

    hw_key = args.hardware or detect_hardware_key()
    print(f"GPU:                    {torch.cuda.get_device_name(0)}")
    print(f"Simulation hardware:    {hw_key}")
    print(f"Simulation model key:   {args.model_config_key}")

    # Select strategies
    all_strategies = get_strategies()
    if args.strategies:
        names = [s.strip() for s in args.strategies.split(",")]
    else:
        names = [
            n for n in all_strategies
            if not n.startswith("h2o_") or args.include_h2o
        ]

    strategies = {n: all_strategies[n] for n in names if n in all_strategies}
    unknown = [n for n in names if n not in all_strategies]
    if unknown:
        print(f"WARNING: Unknown strategies (skipped): {unknown}")
    if not strategies:
        print("ERROR: No valid strategies.")
        return 1

    # Split into two groups by attention requirement
    flash_group = {n: s for n, s in strategies.items() if not n.startswith("h2o_")}
    eager_group  = {n: s for n, s in strategies.items() if n.startswith("h2o_")}

    print(f"\nStrategies (flash/sdpa): {list(flash_group.keys()) or '(none)'}")
    print(f"Strategies (eager/H2O):  {list(eager_group.keys()) or '(none)'}")
    print(f"Prompt len: {args.prompt_len} tokens, Decode steps: {args.decode_steps}")
    print(f"Warmup: {args.warmup}, Repeats: {args.repeats}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Build fixed-length prompt
    base_text = (
        "Solve the following problem step by step. "
        "A train travels at 60 mph. How far does it go in 3 hours? "
    )
    base_ids = tokenizer(base_text, return_tensors="pt").input_ids
    reps = max(1, (args.prompt_len // base_ids.shape[1]) + 1)
    all_ids = tokenizer(base_text * reps, return_tensors="pt").input_ids
    # Keep on CPU; move to device after model is loaded
    prompt_ids_cpu = all_ids[:, : args.prompt_len]
    actual_prompt_len = prompt_ids_cpu.shape[1]
    print(f"Prompt: {actual_prompt_len} tokens (target: {args.prompt_len})")

    benchmark_rows: list[dict] = []

    # --- Group 1: flash_attention_2 / sdpa ---
    if flash_group:
        print(f"\n{'#' * 60}")
        print("# Loading model for flash/sdpa group")
        best_impl = detect_best_attn_impl(args.model, torch.bfloat16, args.device)
        model = load_model(args.model, best_impl, torch.bfloat16, args.device)
        input_ids = prompt_ids_cpu.to(model.device)

        rows = benchmark_group(
            strategies=flash_group,
            model=model,
            attn_impl=best_impl,
            input_ids=input_ids,
            actual_prompt_len=actual_prompt_len,
            hw_key=hw_key,
            model_config_key=args.model_config_key,
            decode_steps=args.decode_steps,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        benchmark_rows.extend(rows)

        # Free VRAM before loading the eager model (if H2O group follows)
        if eager_group:
            del model
            torch.cuda.empty_cache()

    # --- Group 2: eager (H2O) ---
    if eager_group:
        print(f"\n{'#' * 60}")
        print("# Loading model for eager/H2O group")
        print("  NOTE: ObservedAttentionPress requires materialised attention weights.")
        print("        eager attention is mandatory here regardless of flash-attn support.")
        model = load_model(args.model, "eager", torch.bfloat16, args.device)
        input_ids = prompt_ids_cpu.to(model.device)

        rows = benchmark_group(
            strategies=eager_group,
            model=model,
            attn_impl="eager",
            input_ids=input_ids,
            actual_prompt_len=actual_prompt_len,
            hw_key=hw_key,
            model_config_key=args.model_config_key,
            decode_steps=args.decode_steps,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        benchmark_rows.extend(rows)

    if not benchmark_rows:
        print("\nNo results collected.")
        return 1

    # Print summary table
    print(f"\n{'=' * 90}")
    print(f"{'Strategy':<22} {'Attn':>16} {'Meas ms/tok':>12} {'Sim ms/tok':>12} {'Ratio':>8} {'Peak MB':>9}")
    print("-" * 90)
    for row in benchmark_rows:
        print(
            f"{row['strategy']:<22} "
            f"{row['attn_impl']:>16} "
            f"{row['measured_per_token_ms']:>12.3f} "
            f"{row['simulated_per_token_ms']:>12.3f} "
            f"{row['ratio_measured_over_simulated']:>8.2f}x "
            f"{row['peak_memory_mb']:>9.0f}"
        )
    print("=" * 90)

    # Save CSV
    csv_path = os.path.join(args.output_dir, "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(benchmark_rows[0].keys()))
        writer.writeheader()
        writer.writerows(benchmark_rows)
    print(f"\nSaved CSV:  {csv_path}")

    # Save JSONL
    jsonl_path = os.path.join(args.output_dir, "benchmark_results.jsonl")
    with open(jsonl_path, "w") as f:
        for row in benchmark_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved JSONL: {jsonl_path}")

    # Plot: measured vs simulated scatter with y=x reference line
    # Color-code by attn_impl so flash vs eager points are visually distinct
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        impl_colors = {
            "flash_attention_2": "#1565C0",
            "sdpa":              "#2E7D32",
            "eager":             "#B71C1C",
        }

        fig, ax = plt.subplots(figsize=(7, 6))

        all_vals = (
            [r["measured_per_token_ms"] for r in benchmark_rows]
            + [r["simulated_per_token_ms"] for r in benchmark_rows]
        )
        lo, hi = min(all_vals) * 0.9, max(all_vals) * 1.1
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.4, label="y = x (perfect)")

        seen_impls: set[str] = set()
        for row in benchmark_rows:
            impl = row["attn_impl"]
            color = impl_colors.get(impl, "#607D8B")
            label = impl if impl not in seen_impls else None
            seen_impls.add(impl)
            ax.scatter(
                row["simulated_per_token_ms"],
                row["measured_per_token_ms"],
                s=80,
                color=color,
                zorder=3,
                label=label,
            )
            ax.annotate(
                row["strategy"],
                (row["simulated_per_token_ms"], row["measured_per_token_ms"]),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=7,
            )

        ax.set_xlabel("Simulated latency (ms/token)")
        ax.set_ylabel("Measured latency (ms/token)")
        ax.set_title(
            f"Measured vs Simulated per-token latency\n({hw_key}, {args.model_config_key})"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        plot_path = os.path.join(args.output_dir, "benchmark_vs_simulation.png")
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print(f"Saved plot:  {plot_path}")
    except Exception as e:
        print(f"WARNING: Plot failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
