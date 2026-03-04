#!/usr/bin/env python3
"""Smoke test: validate kvpress DecodingPress with Qwen2.5-Math-7B-Instruct.

Run BEFORE building the full accuracy pipeline. This script checks:
  1. Baseline generation (no press) produces coherent math answers
  2. StreamingLLM via DecodingPress compresses the cache and still generates text
  3. ExpectedAttentionPress via DecodingPress compresses and generates text

If DecodingPress crashes or produces garbled output, the accuracy pipeline
needs a manual generation fallback.

Usage:
    pip install kvpress transformers accelerate optimum-quanto torch
    python smoke_test_kvpress.py
"""

from __future__ import annotations

import sys
import time
from contextlib import nullcontext

import torch

# ---------------------------------------------------------------------------
# Hardcoded MATH problem for testing
# ---------------------------------------------------------------------------
MATH_PROBLEM = (
    "Find the value of $x$ such that $\\sqrt{3x + 7} = 10$."
)
EXPECTED_ANSWER = "31"  # sqrt(3*31 + 7) = sqrt(100) = 10

MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
MAX_NEW_TOKENS = 512


def build_prompt(tokenizer, problem: str) -> list[dict]:
    """Build chat-format messages for Qwen2.5-Math-Instruct."""
    return [
        {
            "role": "system",
            "content": (
                "Please reason step by step, and put your final answer "
                "within \\boxed{}."
            ),
        },
        {"role": "user", "content": problem},
    ]


def run_generation(
    model,
    tokenizer,
    problem: str,
    press_ctx=None,
    label: str = "baseline",
) -> dict:
    """Run a single generation and return diagnostics."""
    messages = build_prompt(tokenizer, problem)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    prompt_len = input_ids.shape[1]

    ctx = press_ctx if press_ctx is not None else nullcontext()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()

    with ctx:
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0

    generated_ids = output_ids[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tokens_generated = len(generated_ids)

    # Check for boxed answer
    import re
    boxed_match = re.findall(r"\\boxed\{([^}]*)\}", generated_text)
    extracted = boxed_match[-1] if boxed_match else None

    return {
        "label": label,
        "prompt_tokens": prompt_len,
        "tokens_generated": tokens_generated,
        "elapsed_s": elapsed,
        "tokens_per_sec": tokens_generated / elapsed if elapsed > 0 else 0,
        "extracted_answer": extracted,
        "correct": extracted is not None and extracted.strip() == EXPECTED_ANSWER,
        "text_preview": generated_text[:300],
        "full_text": generated_text,
    }


def print_result(r: dict) -> None:
    """Pretty-print a generation result."""
    status = "CORRECT" if r["correct"] else "WRONG"
    print(f"\n{'='*60}")
    print(f"Strategy: {r['label']}")
    print(f"  Prompt tokens:    {r['prompt_tokens']}")
    print(f"  Generated tokens: {r['tokens_generated']}")
    print(f"  Elapsed:          {r['elapsed_s']:.2f}s")
    print(f"  Tokens/sec:       {r['tokens_per_sec']:.1f}")
    print(f"  Extracted answer: {r['extracted_answer']}")
    print(f"  Correct:          {status}")
    print(f"  Text preview:\n    {r['text_preview'][:200]}...")
    print(f"{'='*60}")


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded. Device: {model.device}")

    results = []

    # --- 1. Baseline (no press) ---
    print("\n[1/4] Running baseline (no press)...")
    r_baseline = run_generation(model, tokenizer, MATH_PROBLEM, label="baseline")
    print_result(r_baseline)
    results.append(r_baseline)

    # --- 2. StreamingLLM via DecodingPress ---
    print("\n[2/4] Running StreamingLLM (DecodingPress + StreamingLLMPress)...")
    try:
        from kvpress import DecodingPress, StreamingLLMPress

        streaming_press = DecodingPress(
            StreamingLLMPress(n_sink=4),
            target_size=512,
            compression_interval=64,
        )
        r_streaming = run_generation(
            model,
            tokenizer,
            MATH_PROBLEM,
            press_ctx=streaming_press(model),
            label="streaming_llm_512",
        )
        print_result(r_streaming)
        results.append(r_streaming)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        results.append({"label": "streaming_llm_512", "error": str(e)})

    # --- 3. ExpectedAttentionPress via DecodingPress ---
    print("\n[3/4] Running ExpectedAttention (DecodingPress + ExpectedAttentionPress)...")
    try:
        from kvpress import DecodingPress, ExpectedAttentionPress

        ea_press = DecodingPress(
            ExpectedAttentionPress(),
            target_size=512,
            compression_interval=64,
        )
        r_ea = run_generation(
            model,
            tokenizer,
            MATH_PROBLEM,
            press_ctx=ea_press(model),
            label="expected_attention_512",
        )
        print_result(r_ea)
        results.append(r_ea)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        results.append({"label": "expected_attention_512", "error": str(e)})

    # --- 4. INT8 HQQ quantized cache ---
    print("\n[4/4] Running INT8 HQQ quantized cache...")
    try:
        messages = build_prompt(tokenizer, MATH_PROBLEM)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        prompt_len = input_ids.shape[1]

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()

        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            cache_implementation="quantized",
            cache_config={"nbits": 8, "backend": "HQQ", "residual_length": 128},
        )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - t0

        generated_ids = output_ids[0, prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        tokens_generated = len(generated_ids)

        import re
        boxed_match = re.findall(r"\\boxed\{([^}]*)\}", generated_text)
        extracted = boxed_match[-1] if boxed_match else None

        r_int8 = {
            "label": "int8_hqq",
            "prompt_tokens": prompt_len,
            "tokens_generated": tokens_generated,
            "elapsed_s": elapsed,
            "tokens_per_sec": tokens_generated / elapsed if elapsed > 0 else 0,
            "extracted_answer": extracted,
            "correct": extracted is not None and extracted.strip() == EXPECTED_ANSWER,
            "text_preview": generated_text[:300],
            "full_text": generated_text,
        }
        print_result(r_int8)
        results.append(r_int8)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        results.append({"label": "int8_hqq", "error": str(e)})

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    all_ok = True
    for r in results:
        if "error" in r:
            print(f"  {r['label']}: FAILED — {r['error']}")
            all_ok = False
        else:
            status = "OK" if r["correct"] else "GENERATED (wrong answer)"
            print(f"  {r['label']}: {status} ({r['tokens_generated']} tokens, {r['elapsed_s']:.1f}s)")
            if "error" not in r and r["tokens_generated"] < 5:
                print(f"    WARNING: very few tokens generated — possible issue")
                all_ok = False

    if all_ok:
        print("\nDECISION: kvpress DecodingPress works. Proceed with full pipeline.")
    else:
        print("\nDECISION: Some strategies failed. Check errors above.")
        print("  If DecodingPress crashed, implement manual generation fallback.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
