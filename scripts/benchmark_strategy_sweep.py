#!/usr/bin/env python3
"""Benchmark decode latency across context lengths AND KV cache strategies.

Measures wall-clock per-token latency for each (context_length, strategy) pair
and compares against the roofline simulator. Results are saved as CSV/JSONL.

Usage:
    python benchmark_strategy_sweep.py
    python benchmark_strategy_sweep.py --hardware GH200 --decode-steps 64
    python benchmark_strategy_sweep.py --context-lengths 4096,16384 --strategies full_cache,snapkv_512
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
# Hardware auto-detection (same map as benchmark_context_sweep.py)
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
    ("5090",  "RTX_5090"),
    ("mi350", "MI350X"),
    ("mi325", "MI325X"),
    ("mi300", "MI300X"),
]


def detect_hardware_key() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return "H100_SXM"
        name = torch.cuda.get_device_name(0).lower()
        for fragment, key in _GPU_NAME_TO_HW_KEY:
            if fragment in name:
                return key
        print(f"  [warn] Unknown GPU '{name}', defaulting to H100_SXM")
    except Exception:
        pass
    return "H100_SXM"


def get_gpu_memory_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    except Exception:
        pass
    return 80.0


# ---------------------------------------------------------------------------
# kvpress strategy factories
# ---------------------------------------------------------------------------

def _parse_strategy(name: str) -> tuple[str, int | None]:
    """Return (family, budget) from a strategy name like 'window_512'."""
    for prefix in ("window_", "h2o_", "snapkv_", "expected_attn_"):
        if name.startswith(prefix):
            budget = int(name[len(prefix):])
            return name, budget
    return name, None


def build_press_for_strategy(name: str, prompt_len: int, decision_interval: int):
    """Build a correctly-configured kvpress press for a strategy at a given prompt length.

    compression_ratio = 1 - budget/prompt_len so that budget tokens are retained.
    DecodingPress uses target_size (absolute token count) for window/h2o.
    SnapKV and ExpectedAttention compress at prefill via compression_ratio.
    """
    if name in ("full_cache", "full_cache_int8"):
        return None

    _, budget = _parse_strategy(name)

    if name.startswith("window_"):
        from kvpress import DecodingPress, StreamingLLMPress
        return DecodingPress(
            StreamingLLMPress(n_sink=4),
            compression_interval=decision_interval,
            target_size=budget,
        )

    if name.startswith("h2o_"):
        from kvpress import DecodingPress, ObservedAttentionPress
        return DecodingPress(
            ObservedAttentionPress(),
            compression_interval=decision_interval,
            target_size=budget,
        )

    if name.startswith("snapkv_"):
        from kvpress import SnapKVPress
        ratio = max(0.0, 1.0 - budget / prompt_len) if prompt_len > 0 else 0.0
        return SnapKVPress(compression_ratio=ratio, window_size=32)

    if name.startswith("expected_attn_"):
        from kvpress import ExpectedAttentionPress
        ratio = max(0.0, 1.0 - budget / prompt_len) if prompt_len > 0 else 0.0
        return ExpectedAttentionPress(compression_ratio=ratio)

    raise ValueError(f"Unknown strategy: {name!r}")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(tokenizer, target_len: int):
    base_text = (
        "The quick brown fox jumps over the lazy dog. "
        "A journey of a thousand miles begins with a single step. "
        "To be or not to be, that is the question. "
    )
    base_ids = tokenizer(base_text, return_tensors="pt").input_ids[0]
    n_base = len(base_ids)
    reps = (target_len // n_base) + 1
    tiled = base_ids.repeat(reps)[:target_len]
    return tiled.unsqueeze(0)


# ---------------------------------------------------------------------------
# Timed generation
# ---------------------------------------------------------------------------

def run_timed_generation(model, input_ids, decode_steps: int, warmup: int,
                          repeats: int, press=None) -> tuple[float, float, float, int]:
    import torch

    def _generate():
        gen_kwargs = dict(max_new_tokens=decode_steps, min_new_tokens=decode_steps, do_sample=False)
        if press is not None:
            with press(model):
                return model.generate(input_ids, **gen_kwargs)
        else:
            return model.generate(input_ids, **gen_kwargs)

    for i in range(warmup):
        with torch.no_grad():
            _generate()
        torch.cuda.synchronize()
        print(f"    warmup {i+1}/{warmup} done")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    times = []
    actual_new = 0
    for i in range(repeats):
        torch.cuda.synchronize()
        t0 = timeit.default_timer()
        with torch.no_grad():
            output = _generate()
        torch.cuda.synchronize()
        elapsed = timeit.default_timer() - t0
        times.append(elapsed)
        actual_new = output.shape[1] - input_ids.shape[1]
        print(f"    run {i+1}/{repeats}: {elapsed*1000:.1f} ms ({actual_new} new tokens)")

    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    mean_s = statistics.mean(times)
    stdev_s = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean_s, stdev_s, peak_mb, actual_new


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(hw_key: str, model_key: str, strategy_name: str,
                   prompt_len: int, decode_steps: int) -> float:
    from hwprop.simulator import simulate_latency
    _, budget = _parse_strategy(strategy_name)
    from hwprop.strategy import get_strategy
    try:
        strat = get_strategy(strategy_name)
    except Exception:
        strat = None
    result = simulate_latency(hw_key, model_key, strategy=strategy_name,
                               prompt_len=prompt_len, decode_steps=decode_steps)
    return result.mean_per_token_ms


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark latency across contexts and strategies")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--model-config", default="LLaMA-3.2-3B")
    parser.add_argument("--hardware", default=None)
    parser.add_argument("--context-lengths", default="4096,16384,32768,65536,131072")
    parser.add_argument("--strategies",
                        default="full_cache,window_512,h2o_512,snapkv_512,expected_attn_512")
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--decision-interval", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-dir", default="results/benchmark")
    parser.add_argument("--skip-oom", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required.")
        return 1

    hw_key = args.hardware or detect_hardware_key()
    gpu_mem_gb = get_gpu_memory_gb()
    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",")]

    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"Hardware key:    {hw_key}")
    print(f"Model:           {args.model}")
    print(f"Context lengths: {context_lengths}")
    print(f"Strategies:      {strategies}")
    print(f"Decode steps:    {args.decode_steps}")
    print(f"Repeats:         {args.repeats}")

    try:
        from flash_attn import flash_attn_func  # noqa
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    print(f"Attn impl:       {attn_impl}\n")

    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        attn_implementation=attn_impl,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Model loaded.\n")

    rows: list[dict] = []
    fieldnames = [
        "context_length", "strategy", "decode_steps", "decision_interval",
        "measured_per_token_ms", "measured_stdev_ms", "simulated_per_token_ms",
        "error_pct", "ratio_measured_over_simulated", "peak_memory_mb",
        "hardware_key", "model_config_key", "attn_impl",
    ]

    for ctx_len in context_lengths:
        input_ids = build_prompt(tokenizer, ctx_len).to(model.device)
        actual_len = input_ids.shape[1]

        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"Context: {ctx_len:,}  Strategy: {strategy}")

            try:
                press = build_press_for_strategy(strategy, actual_len, args.decision_interval)
            except Exception as e:
                print(f"  SKIP: failed to build press — {e}")
                continue

            effective_warmup = 1 if ctx_len > 8192 else args.warmup

            try:
                mean_s, stdev_s, peak_mb, actual_new = run_timed_generation(
                    model=model, input_ids=input_ids,
                    decode_steps=args.decode_steps,
                    warmup=effective_warmup, repeats=args.repeats,
                    press=press,
                )
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM")
                torch.cuda.empty_cache()
                if args.skip_oom:
                    continue
                return 1

            tokens_used = actual_new if actual_new > 0 else args.decode_steps
            meas_ms = mean_s / tokens_used * 1000
            stdev_ms = stdev_s * 1000

            try:
                sim_ms = run_simulation(hw_key, args.model_config, strategy,
                                        actual_len, tokens_used)
            except Exception as e:
                print(f"  [warn] simulation failed: {e}")
                sim_ms = float("nan")

            ratio = meas_ms / sim_ms if sim_ms > 0 else float("nan")
            err_pct = (sim_ms - meas_ms) / meas_ms * 100 if meas_ms > 0 else float("nan")

            print(f"  Measured:  {meas_ms:.3f} ms/token")
            print(f"  Simulated: {sim_ms:.3f} ms/token")
            print(f"  Error:     {err_pct:+.1f}%")

            rows.append({
                "context_length": actual_len,
                "strategy": strategy,
                "decode_steps": tokens_used,
                "decision_interval": args.decision_interval,
                "measured_per_token_ms": round(meas_ms, 3),
                "measured_stdev_ms": round(stdev_ms, 3),
                "simulated_per_token_ms": round(sim_ms, 3),
                "error_pct": round(err_pct, 1),
                "ratio_measured_over_simulated": round(ratio, 3),
                "peak_memory_mb": round(peak_mb, 1),
                "hardware_key": hw_key,
                "model_config_key": args.model_config,
                "attn_impl": attn_impl,
            })

    if not rows:
        print("\nNo results collected.")
        return 1

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Context':>10} {'Strategy':>20} {'Meas':>10} {'Sim':>10} {'Err':>8}")
    print("-" * 65)
    for r in rows:
        print(f"{r['context_length']:>10,} {r['strategy']:>20} "
              f"{r['measured_per_token_ms']:>10.3f} {r['simulated_per_token_ms']:>10.3f} "
              f"{r['error_pct']:>+7.1f}%")
    print("=" * 80)

    csv_path = os.path.join(args.output_dir, f"strategy_sweep_{hw_key}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    jsonl_path = os.path.join(args.output_dir, f"strategy_sweep_{hw_key}.jsonl")
    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {jsonl_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
