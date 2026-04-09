#!/usr/bin/env python3
"""Parse NCU CSV exports into per-category summaries.

Reads raw CSV files exported from NCU .ncu-rep files and classifies kernels
into categories (flash attention, weight GEMM, norm/elementwise, other).
Computes per-category time breakdown, achieved bandwidth, and L2 hit rates.

Usage:
    # After running NCU sweep:
    python scripts/parse_ncu_results.py --input-dir results/ncu
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Kernel classification
# ---------------------------------------------------------------------------
_FLASH_ATTN_PATTERNS = re.compile(
    r"flash_fwd|flash_attn|fmha|flash_bwd|sm90_xmma_gemm.*_f16.*qk|"
    r"void flash|cutlassF|attention_kernel",
    re.IGNORECASE,
)

_GEMM_PATTERNS = re.compile(
    r"gemm|cutlass.*gemm|cublas|sm\d+_xmma|ampere.*gemm|hopper.*gemm|"
    r"void cutlass|volta_.*gemm|turing_.*gemm",
    re.IGNORECASE,
)

_NORM_ELEM_PATTERNS = re.compile(
    r"layernorm|rmsnorm|layer_norm|rms_norm|residual|elementwise|"
    r"silu|gelu|swiglu|add_kernel|mul_kernel|cast_kernel|copy_kernel",
    re.IGNORECASE,
)


def classify_kernel(name: str) -> str:
    """Classify a CUDA kernel name into a category."""
    if _FLASH_ATTN_PATTERNS.search(name):
        return "flash_attention"
    if _GEMM_PATTERNS.search(name):
        return "weight_gemm"
    if _NORM_ELEM_PATTERNS.search(name):
        return "norm_elementwise"
    return "other"


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------
def _safe_float(val: str) -> float:
    """Parse a float from NCU CSV, handling commas and empty strings."""
    if not val or val.strip() in ("", "n/a", "N/A"):
        return 0.0
    return float(val.replace(",", ""))


def parse_ncu_csv(csv_path: str) -> list[dict]:
    """Parse an NCU raw CSV into a list of kernel records."""
    kernels = []
    with open(csv_path, "r") as f:
        # NCU CSVs may have header rows starting with "==". Skip them.
        lines = []
        header_found = False
        for line in f:
            if not header_found:
                if line.startswith('"') or line.startswith("ID"):
                    header_found = True
                    lines.append(line)
            else:
                lines.append(line)

        if not lines:
            print(f"WARNING: No data rows found in {csv_path}")
            return []

        reader = csv.DictReader(lines)
        for row in reader:
            # Find kernel name — NCU uses various column names
            name = (
                row.get("Kernel Name", "")
                or row.get("Name", "")
                or row.get("kernel_name", "")
                or ""
            )
            if not name:
                continue

            kernels.append({
                "name": name,
                "category": classify_kernel(name),
                "duration_ns": _safe_float(row.get("gpu__time_duration.sum", "0")),
                "dram_read_bytes": _safe_float(row.get("dram__bytes_read.sum", "0")),
                "dram_write_bytes": _safe_float(row.get("dram__bytes_write.sum", "0")),
                "l2_hit_sectors": _safe_float(
                    row.get("lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum", "0")
                ),
                "l2_total_sectors": _safe_float(
                    row.get("lts__t_sectors_srcunit_tex_op_read.sum", "0")
                ),
                "sm_throughput_pct": _safe_float(
                    row.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", "0")
                ),
                "dram_throughput_pct": _safe_float(
                    row.get("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "0")
                ),
            })

    return kernels


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_by_category(kernels: list[dict]) -> dict:
    """Aggregate kernel metrics by category."""
    cats = defaultdict(lambda: {
        "count": 0,
        "time_ns": 0.0,
        "dram_read_bytes": 0.0,
        "dram_write_bytes": 0.0,
        "l2_hit_sectors": 0.0,
        "l2_total_sectors": 0.0,
    })

    for k in kernels:
        cat = cats[k["category"]]
        cat["count"] += 1
        cat["time_ns"] += k["duration_ns"]
        cat["dram_read_bytes"] += k["dram_read_bytes"]
        cat["dram_write_bytes"] += k["dram_write_bytes"]
        cat["l2_hit_sectors"] += k["l2_hit_sectors"]
        cat["l2_total_sectors"] += k["l2_total_sectors"]

    total_time_ns = sum(c["time_ns"] for c in cats.values())

    result = {}
    for cat_name, cat in cats.items():
        time_s = cat["time_ns"] / 1e9
        dram_total = cat["dram_read_bytes"] + cat["dram_write_bytes"]
        achieved_bw = dram_total / time_s / 1e9 if time_s > 0 else 0  # GB/s

        l2_hit_rate = (
            cat["l2_hit_sectors"] / cat["l2_total_sectors"]
            if cat["l2_total_sectors"] > 0 else 0.0
        )

        result[cat_name] = {
            "count": cat["count"],
            "time_us": round(cat["time_ns"] / 1e3, 1),
            "pct_of_total": round(cat["time_ns"] / total_time_ns * 100, 1) if total_time_ns > 0 else 0,
            "dram_read_GB": round(cat["dram_read_bytes"] / 1e9, 3),
            "dram_write_GB": round(cat["dram_write_bytes"] / 1e9, 3),
            "achieved_bw_GBs": round(achieved_bw, 1),
            "l2_hit_rate": round(l2_hit_rate, 4),
        }

    return {
        "total_kernel_time_us": round(total_time_ns / 1e3, 1),
        "total_kernels": len(kernels),
        "breakdown": result,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse NCU CSV results")
    parser.add_argument("--input-dir", required=True, help="Directory with NCU CSV files")
    parser.add_argument("--output-dir", default=None, help="Output dir (default: same as input)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir

    csv_files = sorted(Path(args.input_dir).glob("*_raw.csv"))
    if not csv_files:
        print(f"No *_raw.csv files found in {args.input_dir}")
        return 1

    all_summaries = []

    for csv_path in csv_files:
        print(f"\n{'=' * 60}")
        print(f"Parsing: {csv_path.name}")

        kernels = parse_ncu_csv(str(csv_path))
        if not kernels:
            print("  No kernels found, skipping.")
            continue

        summary = aggregate_by_category(kernels)

        # Try to load metadata JSON
        meta_name = csv_path.name.replace("_raw.csv", "_meta.json")
        meta_path = csv_path.parent / meta_name
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            summary["model"] = meta.get("model", "unknown")
            summary["context_length"] = meta.get("context_length", 0)
            summary["wall_clock_decode_ms"] = meta.get("decode_step_ms", 0)
        else:
            summary["model"] = "unknown"
            summary["context_length"] = 0

        all_summaries.append(summary)

        # Print summary
        print(f"  Total kernels: {summary['total_kernels']}")
        print(f"  Total kernel time: {summary['total_kernel_time_us']:.0f} us "
              f"({summary['total_kernel_time_us'] / 1000:.2f} ms)")
        if summary.get("wall_clock_decode_ms"):
            gap = summary["wall_clock_decode_ms"] - summary["total_kernel_time_us"] / 1000
            print(f"  Wall-clock decode: {summary['wall_clock_decode_ms']:.2f} ms "
                  f"(gap = {gap:.2f} ms framework overhead)")
        print()
        print(f"  {'Category':<20} {'Time(us)':>10} {'%':>6} {'DRAM Rd(GB)':>12} "
              f"{'BW(GB/s)':>10} {'L2 Hit':>8} {'Count':>6}")
        print(f"  {'-' * 76}")
        for cat_name, cat in sorted(summary["breakdown"].items(),
                                     key=lambda x: -x[1]["time_us"]):
            print(f"  {cat_name:<20} {cat['time_us']:>10.0f} {cat['pct_of_total']:>5.1f}% "
                  f"{cat['dram_read_GB']:>12.3f} {cat['achieved_bw_GBs']:>10.1f} "
                  f"{cat['l2_hit_rate']:>7.2%} {cat['count']:>6}")

    # Save all summaries
    if all_summaries:
        out_path = os.path.join(output_dir, "ncu_summary.json")
        with open(out_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nSaved summary: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
