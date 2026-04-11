#!/usr/bin/env python3
"""Benchmark wall-clock decode latency across context lengths.

Loads a long-context model (default: LLaMA-3.1-8B, 128K context) and measures
per-token decode latency at increasing prompt lengths. Compares measured times
against the roofline cost model to validate simulation accuracy.

No kvpress, no eviction strategies — pure vanilla generation.

Usage:
    # Full sweep on auto-detected GPU
    python benchmark_context_sweep.py

    # Quick test
    python benchmark_context_sweep.py --context-lengths 1024,4096 --decode-steps 16 --repeats 1

    # Override hardware key for simulation
    python benchmark_context_sweep.py --hardware H100_SXM
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
    ("gh200", "GH200"),
    ("h200",  "H200"),
    ("h100",  "H100_SXM"),
    ("b200",  "B200"),
    ("a100-sxm4-80", "A100_80GB"),
    ("a100-sxm4-40", "A100_40GB"),
    ("a100",  "A100_40GB"),
    ("l40",   "L40S"),
    ("4090",  "RTX_4090"),
    ("5090",  "RTX_5090"),
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


def detect_best_attn_impl(model_name: str) -> str:
    """Return the best available attn_implementation: flash_attention_2 > sdpa."""
    import torch

    try:
        import flash_attn  # noqa: F401
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            if major >= 8:
                return "flash_attention_2"
        print("  [attn] flash-attn installed but GPU compute capability < 8.0 — falling back to sdpa")
    except ImportError:
        print("  [attn] flash-attn not installed — falling back to sdpa")

    return "sdpa"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_prompt(tokenizer, target_len: int):
    """Build a prompt of exactly target_len tokens by tiling base text."""
    base_text = (
        "The quick brown fox jumps over the lazy dog. "
        "A journey of a thousand miles begins with a single step. "
        "To be or not to be, that is the question. "
    )
    base_ids = tokenizer(base_text, return_tensors="pt").input_ids[0]
    n_base = len(base_ids)

    if target_len <= n_base:
        return base_ids[:target_len].unsqueeze(0)

    # Tile to reach target length
    reps = (target_len // n_base) + 1
    tiled = base_ids.repeat(reps)[:target_len]
    return tiled.unsqueeze(0)


# ---------------------------------------------------------------------------
# VRAM estimation
# ---------------------------------------------------------------------------
def estimate_vram_gb(model_config, context_len: int) -> float:
    """Estimate peak VRAM in GB: model weights + KV cache at context_len."""
    param_bytes = model_config.param_bytes
    kv_bytes = model_config.kv_bytes_per_token * context_len
    return (param_bytes + kv_bytes) / (1024**3)


def get_gpu_memory_gb() -> float:
    """Get total GPU memory in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_mem / (1024**3)
    except Exception:
        pass
    return 80.0  # default assumption


# ---------------------------------------------------------------------------
# Timed generation
# ---------------------------------------------------------------------------
def run_timed_generation(
    model,
    input_ids,
    decode_steps: int,
    warmup: int,
    repeats: int,
) -> tuple[float, float, float, int]:
    """Measure wall-clock decode latency.

    Returns (mean_total_s, stdev_total_s, peak_memory_mb, actual_new_tokens).
    """
    import torch

    def _generate():
        return model.generate(
            input_ids,
            max_new_tokens=decode_steps,
            do_sample=False,
        )

    # Warmup
    for i in range(warmup):
        with torch.no_grad():
            _generate()
        torch.cuda.synchronize()
        print(f"    warmup {i + 1}/{warmup} done")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Timed runs
    times = []
    actual_new_tokens = 0
    for i in range(repeats):
        torch.cuda.synchronize()
        t0 = timeit.default_timer()
        with torch.no_grad():
            output = _generate()
        torch.cuda.synchronize()
        elapsed = timeit.default_timer() - t0
        times.append(elapsed)
        actual_new_tokens = output.shape[1] - input_ids.shape[1]
        print(f"    run {i + 1}/{repeats}: {elapsed * 1000:.1f} ms ({actual_new_tokens} new tokens)")

    peak_mb = 0.0
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

    mean_s = statistics.mean(times)
    stdev_s = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean_s, stdev_s, peak_mb, actual_new_tokens


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def run_simulation(
    hardware_key: str,
    model_key: str,
    prompt_len: int,
    decode_steps: int,
) -> dict:
    """Run roofline simulation for full_cache at given context length."""
    from hwprop.eval_pipeline import compute_strategy_latency
    from hwprop.specs import get_hardware_specs, get_model_configs

    hw = get_hardware_specs()[hardware_key]
    model_cfg = get_model_configs()[model_key]

    return compute_strategy_latency(
        strategy_name="full_cache",
        budget_tokens=None,
        hardware=hw,
        model_config=model_cfg,
        prompt_len=prompt_len,
        decode_steps=decode_steps,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(rows: list[dict], hw_key: str, output_dir: str):
    """Generate latency vs context length plot: measured vs simulated."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping plot")
        return

    ctx_lens = [r["context_length"] for r in rows]
    measured = [r["measured_per_token_ms"] for r in rows]
    simulated = [r["simulated_per_token_ms"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: absolute latency
    ax1.plot(ctx_lens, measured, "o-", color="#1565C0", linewidth=2, markersize=6, label="Measured")
    ax1.plot(ctx_lens, simulated, "s--", color="#B71C1C", linewidth=2, markersize=6, label="Simulated (roofline)")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Context length (tokens)")
    ax1.set_ylabel("Latency (ms/token)")
    ax1.set_title(f"Decode latency vs context length ({hw_key})")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Format x-axis with K/M labels
    def fmt_ctx(x, _):
        if x >= 1_000_000:
            return f"{x / 1_000_000:.0f}M"
        if x >= 1024:
            return f"{x / 1024:.0f}K"
        return str(int(x))

    from matplotlib.ticker import FuncFormatter
    ax1.xaxis.set_major_formatter(FuncFormatter(fmt_ctx))

    # Right: measured/simulated ratio
    ratios = [r["ratio_measured_over_simulated"] for r in rows]
    ax2.plot(ctx_lens, ratios, "o-", color="#2E7D32", linewidth=2, markersize=6)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Context length (tokens)")
    ax2.set_ylabel("Ratio (measured / simulated)")
    ax2.set_title("Overhead ratio vs context length")
    ax2.axhline(y=1.0, color="k", linestyle="--", alpha=0.3, label="Perfect prediction")
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(FuncFormatter(fmt_ctx))
    ax2.legend()

    fig.tight_layout()
    plot_path = os.path.join(output_dir, f"latency_vs_context_{hw_key}.png")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark decode latency across context lengths"
    )
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.1-8B",
        help="HuggingFace model name (default: meta-llama/Llama-3.1-8B)",
    )
    parser.add_argument(
        "--model-config", default="LLaMA-3.1-8B",
        help="hwprop model config key for simulation (default: LLaMA-3.1-8B)",
    )
    parser.add_argument(
        "--hardware", default=None,
        help="Hardware catalog key for simulation (default: auto-detect from GPU)",
    )
    parser.add_argument(
        "--context-lengths",
        default="1024,2048,4096,8192,16384,32768,65536,131072",
        help="Comma-separated context lengths to sweep",
    )
    parser.add_argument(
        "--decode-steps", type=int, default=128,
        help="Tokens to generate per run (default: 128)",
    )
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Warmup runs before timing (default: 1)",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Timed runs to average (default: 3)",
    )
    parser.add_argument(
        "--output-dir", default="results/benchmark",
        help="Output directory (default: results/benchmark)",
    )
    parser.add_argument(
        "--skip-oom", action="store_true",
        help="Skip context lengths that cause OOM instead of aborting",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hwprop.specs import get_model_configs

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required for benchmarking.")
        return 1

    hw_key = args.hardware or detect_hardware_key()
    gpu_mem_gb = get_gpu_memory_gb()
    model_cfg = get_model_configs()[args.model_config]
    context_lengths = [int(x) for x in args.context_lengths.split(",")]

    print(f"GPU:                  {torch.cuda.get_device_name(0)}")
    print(f"GPU memory:           {gpu_mem_gb:.1f} GB")
    print(f"Simulation hardware:  {hw_key}")
    print(f"Model:                {args.model}")
    print(f"Model config:         {args.model_config}")
    print(f"KV cache/token:       {model_cfg.kv_bytes_per_token / 1024:.1f} KB")
    print(f"Model size:           {model_cfg.param_bytes / (1024**3):.1f} GB")
    print(f"Context lengths:      {context_lengths}")
    print(f"Decode steps:         {args.decode_steps}")

    # Load model
    attn_impl = detect_best_attn_impl(args.model)
    print(f"\nLoading model with attn_implementation='{attn_impl}'...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Model loaded.\n")

    rows: list[dict] = []

    for ctx_len in context_lengths:
        print(f"\n{'=' * 60}")
        print(f"Context length: {ctx_len:,} tokens")

        # VRAM check
        est_gb = estimate_vram_gb(model_cfg, ctx_len + args.decode_steps)
        print(f"  Estimated peak VRAM: {est_gb:.1f} GB (GPU has {gpu_mem_gb:.1f} GB)")
        if est_gb > gpu_mem_gb * 0.95:
            msg = f"  SKIP: estimated VRAM ({est_gb:.1f} GB) exceeds GPU capacity ({gpu_mem_gb:.1f} GB)"
            print(msg)
            continue

        # Build prompt
        input_ids = build_prompt(tokenizer, ctx_len).to(model.device)
        actual_len = input_ids.shape[1]
        print(f"  Prompt: {actual_len:,} tokens")

        # Adjust warmup for long contexts
        effective_warmup = min(args.warmup, 1) if ctx_len > 8192 else args.warmup

        try:
            mean_s, stdev_s, peak_mb, actual_new = run_timed_generation(
                model=model,
                input_ids=input_ids,
                decode_steps=args.decode_steps,
                warmup=effective_warmup,
                repeats=args.repeats,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at context length {ctx_len:,}")
            torch.cuda.empty_cache()
            if args.skip_oom:
                continue
            else:
                print("  Use --skip-oom to continue past OOM errors")
                return 1

        # Use actual generated tokens for per-token calculation
        tokens_for_calc = actual_new if actual_new > 0 else args.decode_steps
        measured_per_token_ms = (mean_s / tokens_for_calc) * 1000

        # Simulation
        sim = run_simulation(
            hardware_key=hw_key,
            model_key=args.model_config,
            prompt_len=actual_len,
            decode_steps=tokens_for_calc,
        )
        simulated_per_token_ms = sim["mean_latency_ms"]
        ratio = (
            measured_per_token_ms / simulated_per_token_ms
            if simulated_per_token_ms > 0 else float("nan")
        )

        print(f"  Measured:   {measured_per_token_ms:.3f} ms/token  (±{stdev_s * 1000:.1f} ms total)")
        print(f"  Simulated:  {simulated_per_token_ms:.3f} ms/token")
        print(f"  Ratio:      {ratio:.2f}x")
        print(f"  Peak VRAM:  {peak_mb:.0f} MB")

        rows.append({
            "context_length": actual_len,
            "decode_steps": tokens_for_calc,
            "measured_total_ms": round(mean_s * 1000, 2),
            "measured_stdev_ms": round(stdev_s * 1000, 2),
            "measured_per_token_ms": round(measured_per_token_ms, 3),
            "simulated_per_token_ms": round(simulated_per_token_ms, 3),
            "ratio_measured_over_simulated": round(ratio, 3),
            "peak_memory_mb": round(peak_mb, 1),
            "hardware_key": hw_key,
            "model_config_key": args.model_config,
            "model_name": args.model,
            "attn_impl": attn_impl,
        })

    if not rows:
        print("\nNo results collected.")
        return 1

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"{'Context':>10} {'Meas ms/tok':>12} {'Sim ms/tok':>12} {'Ratio':>8} {'Peak MB':>9}")
    print("-" * 80)
    for row in rows:
        print(
            f"{row['context_length']:>10,} "
            f"{row['measured_per_token_ms']:>12.3f} "
            f"{row['simulated_per_token_ms']:>12.3f} "
            f"{row['ratio_measured_over_simulated']:>8.2f}x "
            f"{row['peak_memory_mb']:>9.0f}"
        )
    print("=" * 80)

    # Save CSV
    csv_path = os.path.join(args.output_dir, f"context_sweep_{hw_key}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV:  {csv_path}")

    # Save JSONL
    jsonl_path = os.path.join(args.output_dir, f"context_sweep_{hw_key}.jsonl")
    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved JSONL: {jsonl_path}")

    # Plot
    plot_results(rows, hw_key, args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
