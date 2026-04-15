#!/bin/bash
# Comprehensive pre-launch check. Run BEFORE the full sweep.
# Loads every model, tests every strategy, verifies JSONL write.
# Takes ~15-20 minutes. If anything fails, do NOT launch the full sweep.
#
# Usage:
#   bash scripts/precheck.sh

set -e

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
echo "=========================================="
echo "COMPREHENSIVE PRE-CHECK"
echo "GPU: $GPU_NAME"
echo "$(date)"
echo "=========================================="

PASS=0
FAIL=0
FAILED_ITEMS=()

run_test() {
    local name="$1"
    shift
    echo ""
    echo "--- TEST: $name ---"
    if "$@"; then
        echo "--- PASS: $name ---"
        PASS=$((PASS + 1))
    else
        echo "--- FAIL: $name ---"
        FAIL=$((FAIL + 1))
        FAILED_ITEMS+=("$name")
    fi
}

rm -rf results/grid/_precheck

MODELS=(
    "google/gemma-3-1b-it"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "Qwen/Qwen2.5-7B-Instruct"
    "tiiuae/Falcon3-7B-Base"
    "THUDM/glm-4-9b"
    "microsoft/phi-4"
)

# ============================================================
# Phase 1: Load every model + generate 32 tokens at 1K context
# This catches: download errors, FA2 incompatibility,
# trust_remote_code issues, tokenizer problems
# ============================================================
echo ""
echo "============================================================"
echo "PHASE 1: Model load + short generation (all 9 models)"
echo "============================================================"

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo "$MODEL" | rev | cut -d/ -f1 | rev)
    run_test "Load + generate: $MODEL_SHORT" \
        python scripts/benchmark_full_sweep.py \
            --model "$MODEL" \
            --sweeps context \
            --context-lengths 1024 \
            --decode-steps 32 \
            --warmup 1 \
            --repeats 1 \
            --skip-oom \
            --output-dir results/grid/_precheck
done

# ============================================================
# Phase 2: Test all 4 strategies with kvpress
# Uses Qwen2.5-1.5B (small, fast) at 4K context
# This catches: kvpress import errors, hook failures,
# DecodingPress incompatibility with manual loop
# ============================================================
echo ""
echo "============================================================"
echo "PHASE 2: Strategy sweep (all 4 strategies)"
echo "============================================================"

run_test "Strategy: full_cache + window_512 + snapkv_512 + expected_attn_512" \
    python scripts/benchmark_full_sweep.py \
        --model "Qwen/Qwen2.5-1.5B-Instruct" \
        --sweeps strategy \
        --strategies "full_cache,window_512,snapkv_512,expected_attn_512" \
        --strategy-contexts 4096 \
        --decode-steps 32 \
        --warmup 1 \
        --repeats 1 \
        --skip-oom \
        --output-dir results/grid/_precheck

# ============================================================
# Phase 3: Batch size test
# Uses Qwen2.5-1.5B at 4K, bs=1 and bs=4
# This catches: batched input_ids issues, KV cache batching
# ============================================================
echo ""
echo "============================================================"
echo "PHASE 3: Batch sweep (bs=1, bs=4)"
echo "============================================================"

run_test "Batch: bs=1,4 at 4K context" \
    python scripts/benchmark_full_sweep.py \
        --model "Qwen/Qwen2.5-1.5B-Instruct" \
        --sweeps batch \
        --batch-sizes "1,4" \
        --batch-context 4096 \
        --decode-steps 32 \
        --warmup 1 \
        --repeats 1 \
        --skip-oom \
        --output-dir results/grid/_precheck

# ============================================================
# Phase 4: VRAM stress — largest model at moderate context
# Phi-4 (14B) at 8K — if this fits, most sweep configs will too
# ============================================================
echo ""
echo "============================================================"
echo "PHASE 4: VRAM stress (Phi-4 at 8K)"
echo "============================================================"

run_test "VRAM stress: Phi-4 at 8K context" \
    python scripts/benchmark_full_sweep.py \
        --model "microsoft/phi-4" \
        --sweeps context \
        --context-lengths 8192 \
        --decode-steps 32 \
        --warmup 1 \
        --repeats 1 \
        --skip-oom \
        --output-dir results/grid/_precheck

# ============================================================
# Phase 5: Verify incremental JSONL saving worked
# ============================================================
echo ""
echo "============================================================"
echo "PHASE 5: Verify incremental JSONL"
echo "============================================================"

JSONL_FILES=$(find results/grid/_precheck -name "*.jsonl" 2>/dev/null)
if [ -z "$JSONL_FILES" ]; then
    echo "--- FAIL: No JSONL files found ---"
    FAIL=$((FAIL + 1))
    FAILED_ITEMS+=("JSONL verification")
else
    TOTAL_LINES=0
    for f in $JSONL_FILES; do
        lines=$(wc -l < "$f")
        TOTAL_LINES=$((TOTAL_LINES + lines))
        echo "  $f: $lines rows"
    done
    if [ "$TOTAL_LINES" -ge 9 ]; then
        echo "  Total rows: $TOTAL_LINES (expected >= 9 from phase 1 alone)"
        echo "--- PASS: JSONL verification ---"
        PASS=$((PASS + 1))
    else
        echo "  Total rows: $TOTAL_LINES (expected >= 9, got fewer)"
        echo "--- FAIL: JSONL verification ---"
        FAIL=$((FAIL + 1))
        FAILED_ITEMS+=("JSONL verification — insufficient rows")
    fi
fi

# Cleanup
rm -rf results/grid/_precheck

# ============================================================
# Report
# ============================================================
echo ""
echo "=========================================="
echo "PRE-CHECK RESULTS"
echo "=========================================="
echo "PASSED: $PASS"
echo "FAILED: $FAIL"

if [ ${#FAILED_ITEMS[@]} -gt 0 ]; then
    echo ""
    echo "Failed tests:"
    for item in "${FAILED_ITEMS[@]}"; do
        echo "  - $item"
    done
fi

echo ""
if [ "$FAIL" -gt 0 ]; then
    echo "!!! DO NOT LAUNCH FULL SWEEP — $FAIL test(s) failed !!!"
    echo "Fix the failures above, then re-run this script."
    exit 1
else
    echo "All $PASS tests passed. Safe to launch:"
    echo ""
    echo "  nohup bash scripts/run_all_models.sh > sweep_\$(date +%Y%m%d_%H%M).log 2>&1 &"
    echo "  tail -f sweep_*.log"
fi
echo "=========================================="
