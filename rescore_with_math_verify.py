#!/usr/bin/env python3
"""Post-hoc rescoring of accuracy eval results using math-verify.

Re-evaluates existing JSONL results with symbolic math equivalence checking
(math_verify.parse + math_verify.verify) to recover false negatives from the
original string-matching scorer.

Usage:
    python rescore_with_math_verify.py accuracy_results.jsonl
    python rescore_with_math_verify.py accuracy_results.jsonl --compare
    python rescore_with_math_verify.py accuracy_results.jsonl -o rescored.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

from hwprop.accuracy_eval import extract_boxed_answer


def rescore_answer(extracted: str | None, ground_truth: str) -> bool:
    """Check answer correctness using math-verify for symbolic equivalence.

    Falls back to False if math-verify raises (e.g., unparseable expressions).
    """
    if extracted is None:
        return False

    try:
        from math_verify import parse, verify
    except ImportError:
        raise RuntimeError(
            "math-verify is required. Install with: "
            "pip install -e '.[accuracy]'"
        )

    try:
        parsed_answer = parse(extracted)
        parsed_truth = parse(ground_truth)
        return verify(parsed_answer, parsed_truth)
    except Exception:
        return False


def rescore_results(
    records: list[dict],
) -> list[dict]:
    """Rescore all records, adding rescore fields."""
    rescored = []
    for rec in records:
        new_rec = dict(rec)

        extracted = rec.get("extracted_answer")
        ground_truth = rec.get("ground_truth_answer", "")

        # Also try re-extracting from generated text in case extraction improved
        if extracted is None:
            generated = rec.get("generated_text", "")
            extracted = extract_boxed_answer(generated)
            if extracted is not None:
                new_rec["extracted_answer"] = extracted

        new_correct = rescore_answer(extracted, ground_truth)
        new_rec["correct_rescored"] = new_correct
        new_rec["rescore_method"] = "math-verify"

        # Keep original score for comparison
        new_rec["correct_original"] = rec.get("correct", False)
        # Update the primary correctness field
        new_rec["correct"] = new_correct

        rescored.append(new_rec)
    return rescored


def compute_accuracy(records: list[dict], key: str = "correct") -> dict[str, float]:
    """Compute per-strategy accuracy."""
    by_strategy: dict[str, list[bool]] = defaultdict(list)
    for rec in records:
        by_strategy[rec["strategy_name"]].append(bool(rec.get(key, False)))
    return {
        name: sum(vals) / len(vals) if vals else 0.0
        for name, vals in by_strategy.items()
    }


def print_comparison(records: list[dict]) -> None:
    """Print side-by-side old vs new accuracy with deltas."""
    old_acc = compute_accuracy(records, key="correct_original")
    new_acc = compute_accuracy(records, key="correct_rescored")

    # Count flips
    flipped_pos = sum(
        1 for r in records
        if r.get("correct_rescored") and not r.get("correct_original")
    )
    flipped_neg = sum(
        1 for r in records
        if not r.get("correct_rescored") and r.get("correct_original")
    )

    strategies = sorted(old_acc.keys())
    print(f"\n{'Strategy':<25} {'Original':>10} {'Rescored':>10} {'Delta':>8}")
    print("-" * 55)
    for name in strategies:
        old = old_acc.get(name, 0.0)
        new = new_acc.get(name, 0.0)
        delta = new - old
        delta_str = f"+{delta:.1%}" if delta > 0 else f"{delta:.1%}"
        if delta == 0:
            delta_str = "—"
        print(f"  {name:<23} {old:>9.1%} {new:>9.1%} {delta_str:>8}")

    print("-" * 55)
    total_old = sum(old_acc.values()) / len(old_acc) if old_acc else 0
    total_new = sum(new_acc.values()) / len(new_acc) if new_acc else 0
    print(f"  {'MEAN':<23} {total_old:>9.1%} {total_new:>9.1%}")
    print(f"\n  Flipped correct (recovered):  {flipped_pos}")
    print(f"  Flipped wrong (lost):         {flipped_neg}")
    print(f"  Net change:                   {flipped_pos - flipped_neg:+d}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rescore accuracy eval results with math-verify"
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument(
        "-o", "--output",
        help="Output JSONL file (default: <input>_rescored.jsonl)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print per-strategy old vs new accuracy comparison",
    )
    args = parser.parse_args()

    # Load records
    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {args.input}")

    # Rescore
    rescored = rescore_results(records)

    # Compare
    if args.compare:
        print_comparison(rescored)

    # Save
    output_path = args.output
    if output_path is None:
        base = args.input.rsplit(".", 1)[0]
        output_path = f"{base}_rescored.jsonl"

    with open(output_path, "w") as f:
        for rec in rescored:
            f.write(json.dumps(rec) + "\n")
    print(f"\nSaved {len(rescored)} rescored results to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
