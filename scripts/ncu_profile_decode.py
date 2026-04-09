#!/usr/bin/env python3
"""Profile a single decode step with NVIDIA Nsight Compute (NCU).

Runs prefill WITHOUT profiling, then profiles exactly ONE decode step
using CUDA profiler APIs. NCU captures per-kernel metrics (bandwidth,
L2 hit rate, compute utilization) for the decode step only.

Usage (must be run under ncu):
    ncu --profile-from-start off --set full --export results/ncu_profile \
        python3 scripts/ncu_profile_decode.py --model meta-llama/Llama-3.2-3B --context-length 4096
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import timeit


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

    reps = (target_len // n_base) + 1
    tiled = base_ids.repeat(reps)[:target_len]
    return tiled.unsqueeze(0)


def detect_best_attn_impl(model_name: str) -> str:
    """Return the best available attn_implementation: flash_attention_2 > sdpa."""
    import torch

    try:
        import flash_attn  # noqa: F401
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            if major >= 8:
                return "flash_attention_2"
        print("[attn] flash-attn installed but GPU < 8.0 — sdpa")
    except ImportError:
        print("[attn] flash-attn not installed — sdpa")

    return "sdpa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile one decode step for NCU")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--context-length", type=int, required=True, help="Prompt length in tokens")
    parser.add_argument("--output-dir", default="results/ncu", help="Where to save metadata JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required.")
        return 1

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Model: {args.model}")
    print(f"Context length: {args.context_length:,}")

    # Load model
    attn_impl = detect_best_attn_impl(args.model)
    print(f"Loading model with attn_implementation='{attn_impl}'...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Model loaded.")

    # Build prompt
    input_ids = build_prompt(tokenizer, args.context_length).to(model.device)
    actual_len = input_ids.shape[1]
    print(f"Prompt: {actual_len:,} tokens")

    # Phase 1: Prefill (NOT profiled — NCU --profile-from-start off)
    print("Running prefill (not profiled)...")
    torch.cuda.synchronize()
    t0 = timeit.default_timer()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_kv = outputs.past_key_values
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
    torch.cuda.synchronize()
    prefill_s = timeit.default_timer() - t0
    print(f"Prefill done: {prefill_s * 1000:.1f} ms")

    # Phase 2: Profile ONE decode step
    print("Profiling one decode step...")
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.synchronize()
    t1 = timeit.default_timer()
    with torch.no_grad():
        outputs = model(next_token, past_key_values=past_kv, use_cache=True)
    torch.cuda.synchronize()
    decode_s = timeit.default_timer() - t1
    torch.cuda.cudart().cudaProfilerStop()
    print(f"Decode step done: {decode_s * 1000:.2f} ms")

    # Save metadata
    meta = {
        "model": args.model,
        "context_length": actual_len,
        "attn_impl": attn_impl,
        "gpu": gpu_name,
        "prefill_ms": round(prefill_s * 1000, 2),
        "decode_step_ms": round(decode_s * 1000, 3),
    }
    tag = f"{args.model.split('/')[-1]}_ctx{actual_len}"
    meta_path = os.path.join(args.output_dir, f"{tag}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
