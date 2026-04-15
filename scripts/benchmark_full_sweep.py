#!/usr/bin/env python3
"""Unified benchmark: context sweep + strategy sweep + batch sweep.

Runs all three sweeps for a single model on the current GPU, producing one
CSV with all results. Designed to be called once per model per GPU.

RESILIENCE:
    - Results saved incrementally to JSONL after EVERY config (survives crashes)
    - Conservative VRAM headroom to prevent kernel-killing OOMs
    - Warmup memory check: if warmup uses >90% VRAM, timed runs are skipped
    - Progress logging with elapsed time and ETA

TIMING SEMANTICS:
    Every run uses the SAME code path: manual prefill + manual decode loop.
    For strategy runs, a kvpress context manager wraps both phases.
    Sync at phase boundaries only. No per-step sync.

Usage:
    python scripts/benchmark_full_sweep.py --model meta-llama/Llama-3.2-3B
    python scripts/benchmark_full_sweep.py --model meta-llama/Llama-3.2-3B --sweeps context
    python scripts/benchmark_full_sweep.py --model THUDM/glm-4-9b --sweeps preflight
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time as time_mod
import timeit
from contextlib import nullcontext

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

# Script start time for progress tracking
_SCRIPT_START = time_mod.time()


def _elapsed() -> str:
    """Human-readable elapsed time since script start."""
    s = int(time_mod.time() - _SCRIPT_START)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


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
    ("a100",  "A100_80GB"),
    ("l40s",  "L40S"),
    ("l40",   "L40S"),
    ("a40",   "A40"),
    ("5090",  "RTX_5090"),
    ("4090",  "RTX_4090"),
    ("mi350", "MI350X"),
    ("mi325", "MI325X"),
    ("mi300", "MI300X"),
    ("gaudi", "Gaudi_3"),
]


def detect_gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown"


def detect_hardware_key() -> str:
    name = detect_gpu_name().lower()
    for fragment, key in _GPU_NAME_TO_HW_KEY:
        if fragment in name:
            return key
    print(f"  [warn] Unknown GPU '{name}', defaulting to H100_SXM")
    return "H100_SXM"


def detect_best_attn_impl() -> str:
    import torch
    try:
        import flash_attn  # noqa: F401
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            if major >= 8:
                return "flash_attention_2"
        print("  [attn] flash-attn: GPU compute < 8.0, falling back to sdpa")
    except ImportError:
        print("  [attn] flash-attn not installed, falling back to sdpa")
    return "sdpa"


def get_gpu_memory_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_mem / (1024**3)
    except Exception:
        pass
    return 80.0


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_prompt(tokenizer, target_len: int, batch_size: int = 1):
    """Build prompt of shape [batch_size, target_len]."""
    base_text = (
        "The quick brown fox jumps over the lazy dog. "
        "A journey of a thousand miles begins with a single step. "
        "To be or not to be, that is the question. "
    )
    base_ids = tokenizer(base_text, return_tensors="pt").input_ids[0]
    n_base = len(base_ids)

    if target_len <= n_base:
        single = base_ids[:target_len].unsqueeze(0)
    else:
        reps = (target_len // n_base) + 1
        tiled = base_ids.repeat(reps)[:target_len]
        single = tiled.unsqueeze(0)

    if batch_size > 1:
        return single.expand(batch_size, -1).contiguous()
    return single


def estimate_vram_gb(param_bytes: int, kv_bytes_per_token: int,
                     context_len: int, batch_size: int = 1) -> float:
    kv = kv_bytes_per_token * context_len * batch_size
    return (param_bytes + kv) / (1024**3)


# ---------------------------------------------------------------------------
# Strategy setup (kvpress)
# ---------------------------------------------------------------------------
def build_press(strategy: str, decision_interval: int = 64):
    """Build a kvpress press object. Returns None for full_cache."""
    if strategy == "full_cache":
        return None

    if strategy.startswith("window_"):
        budget = int(strategy.split("_")[1])
        from kvpress import DecodingPress, StreamingLLMPress
        return DecodingPress(
            StreamingLLMPress(n_sink=4),
            compression_interval=decision_interval,
            target_size=budget,
        )

    if strategy.startswith("snapkv_"):
        budget = int(strategy.split("_")[1])
        from kvpress import DecodingPress, SnapKVPress
        return DecodingPress(
            SnapKVPress(window_size=32),
            compression_interval=decision_interval,
            target_size=budget,
        )

    if strategy.startswith("expected_attn_"):
        budget = int(strategy.split("_")[2])
        from kvpress import DecodingPress, ExpectedAttentionPress
        return DecodingPress(
            ExpectedAttentionPress(),
            compression_interval=decision_interval,
            target_size=budget,
        )

    raise ValueError(f"Unknown strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Core timing function — SINGLE CODE PATH for all configurations
# ---------------------------------------------------------------------------
def time_generation(model, input_ids, decode_steps: int, warmup: int,
                    repeats: int, press=None, gpu_mem_gb: float = 80.0) -> dict:
    """Time prefill + decode with consistent semantics for all configurations.

    Uses manual prefill + decode loop for ALL paths.
    When press is provided, it wraps both phases via kvpress forward hooks.

    Returns dict with timing results, or None if warmup shows VRAM is too tight.
    """
    import torch

    batch_size = input_ids.shape[0]

    def _run_once(num_decode, ctx_mgr):
        with ctx_mgr:
            # Phase 1: Prefill
            torch.cuda.synchronize()
            t_prefill = timeit.default_timer()
            with torch.no_grad():
                outputs = model(input_ids, use_cache=True)
                past_kv = outputs.past_key_values
                next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
            torch.cuda.synchronize()
            prefill_s = timeit.default_timer() - t_prefill

            # Phase 2: Decode
            torch.cuda.synchronize()
            t_decode = timeit.default_timer()
            for _ in range(num_decode):
                with torch.no_grad():
                    outputs = model(next_token, past_key_values=past_kv,
                                    use_cache=True)
                    past_kv = outputs.past_key_values
                    next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
            torch.cuda.synchronize()
            decode_s = timeit.default_timer() - t_decode

        return prefill_s, decode_s, num_decode

    def _make_ctx():
        if press is not None:
            return press(model)
        return nullcontext()

    # --- Warmup (full-length) + VRAM safety check ---
    torch.cuda.reset_peak_memory_stats()
    for i in range(warmup):
        _run_once(decode_steps, _make_ctx())
        print(f"    warmup {i + 1}/{warmup}")

    warmup_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    warmup_peak_frac = warmup_peak_mb / (gpu_mem_gb * 1024)
    print(f"    warmup peak VRAM: {warmup_peak_mb:.0f} MB "
          f"({warmup_peak_frac * 100:.0f}% of {gpu_mem_gb:.0f} GB)")

    if warmup_peak_frac > 0.92:
        print(f"    DANGER: warmup used {warmup_peak_frac*100:.0f}% VRAM — "
              f"skipping timed runs to avoid kernel OOM")
        return None

    torch.cuda.reset_peak_memory_stats()

    # --- Timed runs ---
    prefill_times = []
    decode_times = []
    actual_steps = 0

    for i in range(repeats):
        prefill_s, decode_s, actual_steps = _run_once(decode_steps, _make_ctx())
        prefill_times.append(prefill_s)
        decode_times.append(decode_s)
        ms_per_step = decode_s / actual_steps * 1000
        total_toks = actual_steps * batch_size
        print(f"    run {i + 1}/{repeats}: prefill {prefill_s * 1000:.0f} ms, "
              f"decode {decode_s * 1000:.0f} ms "
              f"({ms_per_step:.2f} ms/step, {total_toks} tokens @ bs={batch_size})")

    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    mean_prefill = statistics.mean(prefill_times)
    mean_decode = statistics.mean(decode_times)
    stdev_decode = statistics.stdev(decode_times) if len(decode_times) > 1 else 0.0

    return {
        "prefill_ms": mean_prefill * 1000,
        "decode_total_ms": mean_decode * 1000,
        "mean_ms_per_step": mean_decode / actual_steps * 1000,
        "std_ms_per_step": stdev_decode / actual_steps * 1000,
        "num_decode_steps": actual_steps,
        "batch_size": batch_size,
        "peak_memory_mb": peak_mb,
    }


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
def run_preflight(model, tokenizer, input_ids, attn_impl: str, decode_steps: int = 8):
    import torch

    print("\n  Preflight: generating 8 tokens...")
    torch.cuda.synchronize()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_kv = outputs.past_key_values
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        for _ in range(decode_steps):
            outputs = model(next_token, past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
    torch.cuda.synchronize()
    print(f"  Preflight OK: {attn_impl}, generated {decode_steps} tokens")

    config = model.config
    sliding = getattr(config, "sliding_window", None)
    if sliding is not None:
        print(f"  WARNING: model has sliding_window={sliding}")
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            n_sliding = sum(1 for layer in model.model.layers
                          if hasattr(layer.self_attn, "is_sliding")
                          and layer.self_attn.is_sliding)
            if n_sliding > 0:
                print(f"  WARNING: {n_sliding}/{len(model.model.layers)} layers "
                      f"use sliding window attention")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Full benchmark sweep for one model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark_full_sweep.py --model meta-llama/Llama-3.2-3B
  python scripts/benchmark_full_sweep.py --model THUDM/glm-4-9b --sweeps preflight
  python scripts/benchmark_full_sweep.py --model meta-llama/Llama-3.2-3B --sweeps context,batch
        """,
    )
    p.add_argument("--model", required=True, help="HuggingFace model name")
    p.add_argument("--hardware", default=None, help="Override hardware key (metadata only)")
    p.add_argument("--output-dir", default="results/grid", help="Output directory")
    p.add_argument("--sweeps", default="context,strategy,batch",
                   help="Comma-separated: context, strategy, batch, preflight")
    p.add_argument("--decode-steps", type=int, default=512)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--skip-oom", action="store_true",
                   help="Skip OOM/error configs instead of aborting")

    p.add_argument("--context-lengths",
                   default="512,1024,2048,4096,8192,16384,32768,65536,131072")
    p.add_argument("--strategies",
                   default="full_cache,window_512,snapkv_512,expected_attn_512")
    p.add_argument("--strategy-contexts", default="4096,8192,16384,65536")
    p.add_argument("--decision-interval", type=int, default=64)
    p.add_argument("--batch-sizes", default="1,4,16,64,128,256")
    p.add_argument("--batch-context", type=int, default=8192)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    sweeps = [s.strip() for s in args.sweeps.split(",")]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required.")
        return 1

    # --- Detect hardware ---
    gpu_raw = detect_gpu_name()
    hw_key = args.hardware or detect_hardware_key()
    gpu_mem_gb = get_gpu_memory_gb()
    attn_impl = detect_best_attn_impl()

    # Conservative VRAM headroom: max(8GB, 15% of total)
    vram_headroom_gb = max(8.0, gpu_mem_gb * 0.15)

    # --- Load model config ---
    print(f"\nLoading config for {args.model}...")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    # Handle different config attribute names across model families
    # ChatGLM uses num_layers instead of num_hidden_layers
    num_layers = getattr(config, "num_hidden_layers",
                 getattr(config, "num_layers", None))
    if num_layers is None:
        print(f"ERROR: Cannot find layer count in config for {args.model}")
        print(f"  Config attributes: {[k for k in config.to_dict().keys()]}")
        return 1
    num_heads = config.num_attention_heads
    # ChatGLM uses multi_query_group_num instead of num_key_value_heads
    num_kv_heads = getattr(config, "num_key_value_heads",
                   getattr(config, "multi_query_group_num", num_heads))
    hidden_size = config.hidden_size
    head_dim = getattr(config, "head_dim", hidden_size // num_heads)
    max_ctx = getattr(config, "max_position_embeddings",
              getattr(config, "seq_length", 131072))
    kv_bytes_per_token = num_kv_heads * head_dim * 2 * 2 * num_layers

    model_short = args.model.split("/")[-1]

    print(f"GPU:             {gpu_raw}")
    print(f"GPU memory:      {gpu_mem_gb:.1f} GB (headroom: {vram_headroom_gb:.1f} GB)")
    print(f"Hardware key:    {hw_key}")
    print(f"Attention impl:  {attn_impl}")
    print(f"Model:           {args.model}")
    print(f"  layers={num_layers}, heads={num_heads}, kv_heads={num_kv_heads}, "
          f"head_dim={head_dim}, hidden={hidden_size}")
    print(f"  kv_bytes/tok:  {kv_bytes_per_token / 1024:.0f} KB")
    print(f"  max_context:   {max_ctx}")
    print(f"Sweeps:          {sweeps}")
    print(f"Decode steps:    {args.decode_steps}")

    # --- Load model ---
    print(f"\nLoading model with attn_implementation='{attn_impl}'...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  FA2 failed ({e}), falling back to sdpa...")
        attn_impl = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
    model.eval()

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"  param_bytes:   {param_bytes / 1e9:.2f} GB")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded.\n")

    # --- Preflight ---
    if "preflight" in sweeps:
        print("=" * 70)
        print("PREFLIGHT CHECK")
        print("=" * 70)
        pf_ids = build_prompt(tokenizer, 512).to(model.device)
        run_preflight(model, tokenizer, pf_ids, attn_impl)
        if sweeps == ["preflight"]:
            print("\nPreflight only — exiting.")
            return 0

    # --- Build config list (so we can count total for progress) ---
    configs = []  # list of (ctx, bs, strategy, sweep_name)

    if "context" in sweeps:
        for ctx in [int(x) for x in args.context_lengths.split(",")]:
            configs.append((ctx, 1, "full_cache", "context"))

    if "strategy" in sweeps:
        strategies = [s.strip() for s in args.strategies.split(",")]
        strategy_contexts = [int(x) for x in args.strategy_contexts.split(",")]
        for ctx in strategy_contexts:
            for strat in strategies:
                configs.append((ctx, 1, strat, "strategy"))

    if "batch" in sweeps:
        for bs in [int(x) for x in args.batch_sizes.split(",")]:
            configs.append((args.batch_context, bs, "full_cache", "batch"))

    total_configs = len(configs)
    print(f"Total configs to run: {total_configs}")

    # --- Output setup: open JSONL for incremental writing ---
    out_dir = os.path.join(args.output_dir, model_short)
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, f"{hw_key}.jsonl")

    # Write mode — fresh file per model invocation to avoid duplicates
    # (each model is a separate script call, so no append-resume needed)
    jsonl_file = open(jsonl_path, "w")
    print(f"Streaming results to: {jsonl_path}")

    all_rows = []
    completed = 0
    skipped = 0
    failed = 0
    config_times = []  # for ETA calculation

    def should_skip(context_len: int, batch_size: int) -> str | None:
        if context_len > max_ctx:
            return f"context {context_len} > max {max_ctx}"
        est_gb = estimate_vram_gb(param_bytes, kv_bytes_per_token,
                                  context_len + args.decode_steps, batch_size)
        if est_gb > gpu_mem_gb - vram_headroom_gb:
            return f"VRAM est {est_gb:.1f} GB > limit {gpu_mem_gb - vram_headroom_gb:.1f} GB"
        return None

    def make_row(timing: dict, context_length: int, batch_size: int,
                 strategy: str) -> dict:
        ms_per_step = timing["mean_ms_per_step"]
        ms_per_token = ms_per_step / batch_size
        return {
            "gpu_name": gpu_raw,
            "hardware_key": hw_key,
            "model_name": args.model,
            "context_length": context_length,
            "batch_size": batch_size,
            "strategy": strategy,
            "mean_ms_per_step": round(ms_per_step, 3),
            "std_ms_per_step": round(timing["std_ms_per_step"], 3),
            "mean_ms_per_token": round(ms_per_token, 3),
            "throughput_tok_per_sec": round(batch_size / ms_per_step * 1000, 1)
                                     if ms_per_step > 0 else 0,
            "prefill_ms": round(timing["prefill_ms"], 2),
            "total_decode_ms": round(timing["decode_total_ms"], 2),
            "num_decode_steps": timing["num_decode_steps"],
            "num_warmup_runs": args.warmup,
            "total_generate_ms": round(timing["prefill_ms"] + timing["decode_total_ms"], 2),
            "peak_memory_mb": round(timing["peak_memory_mb"], 1),
            "param_bytes": param_bytes,
            "kv_bytes_per_token": kv_bytes_per_token,
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "max_context": max_ctx,
            "attention_impl_actual": attn_impl,
        }

    # --- Run all configs ---
    current_sweep = None
    for idx, (ctx, bs, strategy, sweep_name) in enumerate(configs):
        # Print sweep header on transition
        if sweep_name != current_sweep:
            current_sweep = sweep_name
            print(f"\n{'=' * 70}")
            print(f"SWEEP: {sweep_name.upper()}")
            print("=" * 70)

        # Progress
        eta_str = ""
        if config_times:
            avg_s = statistics.mean(config_times)
            remaining = (total_configs - idx) * avg_s
            eta_m = int(remaining / 60)
            eta_str = f" | ETA ~{eta_m}min"

        # Skip check
        skip = should_skip(ctx, bs)
        if skip:
            print(f"\n  [{idx+1}/{total_configs}] [{_elapsed()}]{eta_str} "
                  f"SKIP {strategy} ctx={ctx} bs={bs}: {skip}")
            skipped += 1
            continue

        label = f"{strategy} @ ctx={ctx:,} bs={bs}"
        print(f"\n  [{idx+1}/{total_configs}] [{_elapsed()}]{eta_str} {label}")

        # Aggressive memory cleanup before each config
        gc.collect()
        torch.cuda.empty_cache()

        # Extra safety: check actual GPU memory usage
        alloc_gb = torch.cuda.memory_allocated() / (1024**3)
        if alloc_gb > gpu_mem_gb * 0.5:
            print(f"    [mem] {alloc_gb:.1f} GB allocated (>{gpu_mem_gb*0.5:.0f} GB), "
                  f"forcing cleanup...")
            gc.collect()
            torch.cuda.empty_cache()

        config_start = time_mod.time()

        try:
            input_ids = build_prompt(tokenizer, ctx, bs).to(model.device)
            press = build_press(strategy, decision_interval=args.decision_interval)

            timing = time_generation(
                model, input_ids, args.decode_steps,
                args.warmup, args.repeats, press=press,
                gpu_mem_gb=gpu_mem_gb,
            )

            if timing is None:
                # Warmup VRAM check failed — skip this config
                print(f"  -> SKIPPED (VRAM too tight during warmup)")
                skipped += 1
                continue

            row = make_row(timing, ctx, bs, strategy)
            all_rows.append(row)

            # --- INCREMENTAL SAVE: write to JSONL immediately ---
            jsonl_file.write(json.dumps(row) + "\n")
            jsonl_file.flush()
            os.fsync(jsonl_file.fileno())

            tps = row["throughput_tok_per_sec"]
            print(f"  -> {timing['mean_ms_per_step']:.2f} ms/step, "
                  f"{tps:.0f} tok/s, peak {timing['peak_memory_mb']:.0f} MB "
                  f"[SAVED row {len(all_rows)}]")
            completed += 1

        except torch.cuda.OutOfMemoryError:
            # OOM: always skip (never let it propagate — can kill kernel)
            print(f"  -> OOM at {label} — skipping (OOM is always non-fatal)")
            torch.cuda.empty_cache()
            gc.collect()
            failed += 1
        except Exception as e:
            # Non-OOM errors: fatal unless --skip-oom
            print(f"  -> ERROR at {label}: {e}")
            failed += 1
            if not args.skip_oom:
                print(f"  -> FATAL: non-OOM error. Use --skip-oom to continue past errors.")
                jsonl_file.close()
                raise

        config_times.append(time_mod.time() - config_start)

    jsonl_file.close()

    # --- Final CSV (convenience, all data already in JSONL) ---
    if all_rows:
        csv_path = os.path.join(out_dir, f"{hw_key}.csv")
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nSaved CSV: {csv_path} ({len(all_rows)} rows)")

    # --- Summary ---
    print(f"\n{'=' * 90}")
    print(f"DONE in {_elapsed()} | "
          f"completed={completed} skipped={skipped} failed={failed} "
          f"total_rows={len(all_rows)}")
    print(f"{'=' * 90}")

    if all_rows:
        print(f"\n{'Strategy':<20} {'Ctx':>8} {'BS':>4} {'Prefill':>10} {'ms/step':>10} "
              f"{'tok/s':>10} {'Peak MB':>10}")
        print("-" * 90)
        for r in all_rows:
            print(f"{r['strategy']:<20} {r['context_length']:>8} "
                  f"{r['batch_size']:>4} {r['prefill_ms']:>10.1f} "
                  f"{r['mean_ms_per_step']:>10.2f} "
                  f"{r['throughput_tok_per_sec']:>10.1f} "
                  f"{r['peak_memory_mb']:>10.0f}")
        print("=" * 90)

    print(f"\nJSONL (durable): {jsonl_path}")
    if all_rows:
        print(f"CSV (convenience): {os.path.join(out_dir, f'{hw_key}.csv')}")

    if not all_rows:
        print("\nERROR: No results collected!")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
