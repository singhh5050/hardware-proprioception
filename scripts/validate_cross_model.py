#!/usr/bin/env python3
"""Validate whether overhead profiles generalize across models.

Given a benchmark CSV (from benchmark_context_sweep.py) for a NEW model,
computes what the existing calibrated overhead profile would predict for
that model, and reports the error.

This answers the critical question: are overhead coefficients hardware-only,
or do they absorb model-specific effects?

Usage:
    # After running benchmark_context_sweep.py with a new model:
    python scripts/validate_cross_model.py \
        --csv results/benchmark/context_sweep_H100_SXM.csv \
        --model-config Qwen2.5-7B \
        --hardware H100_SXM \
        --attn-impl flash_attention_2

    # Compare against the profile that was calibrated on LLaMA-3.2-3B.
    # If error is ~5%, the coefficients are hardware-only.
    # If error is 40%, something model-dependent is leaking in.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hwprop.specs import get_hardware_specs, get_model_configs
from hwprop.cost_model import CostModel, KVCacheState
from hwprop.overhead import (
    OVERHEAD_H100_FLASH2,
    OVERHEAD_H200_FLASH2,
    OVERHEAD_RTX5090_FLASH2,
    OVERHEAD_A100_SDPA,
    OVERHEAD_A100_SDPA_64,
    OVERHEAD_GH200_SDPA,
    OVERHEAD_GH200_SDPA_64,
    OverheadProfile,
)

# Map (hardware_key, attn_impl) -> calibrated profile
CALIBRATED_PROFILES = {
    ("H100_SXM", "flash_attention_2"): OVERHEAD_H100_FLASH2,
    ("H200", "flash_attention_2"): OVERHEAD_H200_FLASH2,
    ("RTX_5090", "flash_attention_2"): OVERHEAD_RTX5090_FLASH2,
    ("A100_40GB", "sdpa"): OVERHEAD_A100_SDPA,
    ("GH200", "sdpa"): OVERHEAD_GH200_SDPA,
}

# Which model each profile was calibrated on
CALIBRATION_MODEL = {
    ("H100_SXM", "flash_attention_2"): "LLaMA-3.2-3B",
    ("H200", "flash_attention_2"): "LLaMA-3.2-3B",
    ("RTX_5090", "flash_attention_2"): "LLaMA-3.2-3B",
    ("A100_40GB", "sdpa"): "LLaMA-3.2-3B",
    ("GH200", "sdpa"): "LLaMA-3.2-3B",
}


def main():
    parser = argparse.ArgumentParser(description="Cross-model overhead validation")
    parser.add_argument("--csv", required=True, help="Benchmark CSV from context_sweep")
    parser.add_argument("--model-config", required=True, help="hwprop model config key for the NEW model")
    parser.add_argument("--hardware", required=True, help="Hardware key (e.g., H100_SXM)")
    parser.add_argument("--attn-impl", default="flash_attention_2", help="Attention implementation")
    args = parser.parse_args()

    # Load the calibrated profile (fit on original model)
    profile_key = (args.hardware, args.attn_impl)
    if profile_key not in CALIBRATED_PROFILES:
        print(f"No calibrated profile for {profile_key}.")
        print(f"Available: {list(CALIBRATED_PROFILES.keys())}")
        print("Falling back to for_hardware() derivation.")
        hw = get_hardware_specs()[args.hardware]
        profile = OverheadProfile.for_hardware(hw, attn_impl=args.attn_impl)
        cal_model = "N/A (derived)"
    else:
        profile = CALIBRATED_PROFILES[profile_key]
        cal_model = CALIBRATION_MODEL[profile_key]

    # Load the NEW model config
    model_configs = get_model_configs()
    if args.model_config not in model_configs:
        print(f"Unknown model config '{args.model_config}'.")
        print(f"Available: {list(model_configs.keys())}")
        return 1
    new_model = model_configs[args.model_config]

    # Also load the calibration model for comparison
    hw = get_hardware_specs()[args.hardware]
    cost_model = CostModel(
        hw, new_model,
        kv_bandwidth_alpha=profile.kv_bandwidth_alpha,
        kv_bandwidth_beta=profile.kv_bandwidth_beta,
    )

    kv_head_layers = new_model.num_kv_heads * new_model.num_layers

    print(f"=" * 70)
    print(f"Cross-Model Overhead Validation")
    print(f"=" * 70)
    print(f"Hardware:            {args.hardware}")
    print(f"Attention impl:      {args.attn_impl}")
    print(f"Profile:             {profile.name}")
    print(f"Calibrated on:       {cal_model}")
    print(f"Testing on:          {args.model_config}")
    print(f"  num_layers:        {new_model.num_layers}")
    print(f"  num_kv_heads:      {new_model.num_kv_heads}")
    print(f"  head_dim:          {new_model._head_dim}")
    print(f"  kv_head_layers:    {kv_head_layers}")
    if cal_model != "N/A (derived)" and cal_model in model_configs:
        cal_m = model_configs[cal_model]
        cal_khl = cal_m.num_kv_heads * cal_m.num_layers
        print(f"  (calibration model had kv_head_layers={cal_khl}, "
              f"ratio={kv_head_layers/cal_khl:.2f}x)")
    print(f"=" * 70)

    # Read benchmark CSV
    rows = []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        print("ERROR: CSV is empty.")
        return 1

    # For each context length, compute predicted vs measured
    print(f"\n{'Context':>8} {'Measured':>12} {'Predicted':>12} {'Roofline':>12} "
          f"{'Error':>8} {'Overhead':>10}")
    print(f"{'':>8} {'(ms/tok)':>12} {'(ms/tok)':>12} {'(ms/tok)':>12} "
          f"{'(%)':>8} {'breakdown':>10}")
    print("-" * 70)

    errors = []
    for row in rows:
        ctx = int(row["context_length"])
        # Handle both old CSV format (measured_per_token_ms / decode_per_token_ms)
        # and new format (measured_per_step_ms)
        measured_ms = float(
            row.get("measured_per_step_ms",
            row.get("measured_per_token_ms",
            row.get("decode_per_token_ms", 0)))
        )
        if measured_ms == 0:
            continue

        # Roofline prediction for the NEW model
        kv = KVCacheState(
            seq_len=ctx, tokens_in_hbm=ctx,
            tokens_in_hbm_quantized=0, tokens_in_cpu=0,
            tokens_on_disk=0, tokens_evicted=0,
        )
        raw_cost = cost_model.step_cost(kv)
        roofline_ms = raw_cost.time_s * 1000

        # Apply calibrated overhead profile
        predicted_s = profile.corrected_time(
            roofline_time_s=raw_cost.time_s,
            active_tokens=ctx,
            kv_head_layers=kv_head_layers,
        )
        predicted_ms = predicted_s * 1000

        error_pct = (predicted_ms - measured_ms) / measured_ms * 100
        errors.append(abs(error_pct))

        # Overhead breakdown
        breakdown = profile.overhead_breakdown(
            roofline_time_s=raw_cost.time_s,
            active_tokens=ctx,
            kv_head_layers=kv_head_layers,
        )
        dominant = max(breakdown, key=breakdown.get)
        dominant_pct = breakdown[dominant] * 100

        print(f"{ctx:>8,} {measured_ms:>12.2f} {predicted_ms:>12.2f} {roofline_ms:>12.3f} "
              f"{error_pct:>+7.1f}% {dominant:>7}={dominant_pct:.0f}%")

    if errors:
        mae = sum(errors) / len(errors)
        max_err = max(errors)
        print("-" * 70)
        print(f"MAE: {mae:.1f}%   Max error: {max_err:.1f}%")
        print()
        if mae < 10:
            print("RESULT: Coefficients appear to be hardware-only.")
            print("        The overhead profile generalizes across models on this hardware.")
        elif mae < 25:
            print("RESULT: Moderate model-dependent leakage.")
            print("        The overhead profile partially transfers but has meaningful error.")
            print("        Consider whether the functional form needs a model-dependent term.")
        else:
            print("RESULT: Significant model-dependent leakage.")
            print("        The overhead coefficients absorb model-specific effects.")
            print("        The functional form needs revision before use in RL.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
