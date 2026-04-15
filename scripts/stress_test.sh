#!/bin/bash
# Pre-launch stress test. Run this BEFORE nohup-ing the full sweep.
# Verifies: model loading, FA2, kvpress, timing harness, incremental save, VRAM.
# Should take ~5-8 minutes. If anything fails, DO NOT launch the full sweep.
#
# Usage:
#   bash scripts/stress_test.sh

set -e  # Fail fast during stress test — we WANT to catch problems

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
echo "=========================================="
echo "STRESS TEST"
echo "GPU: $GPU_NAME"
echo "$(date)"
echo "=========================================="

PASS=0
FAIL=0

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
    fi
}

# Clean up any leftover stress test data
rm -rf results/grid/_stress_test

# Test 1: Smallest model preflight (Gemma-3-1B or fallback to Qwen2.5-1.5B)
run_test "Preflight: small model (Qwen2.5-1.5B)" \
    python scripts/benchmark_full_sweep.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --sweeps preflight \
        --output-dir results/grid/_stress_test

# Test 2: Largest model preflight (Phi-4, 14B)
run_test "Preflight: large model (Phi-4)" \
    python scripts/benchmark_full_sweep.py \
        --model microsoft/phi-4 \
        --sweeps preflight \
        --output-dir results/grid/_stress_test

# Test 3: Short context benchmark (verifies timing harness + incremental save)
run_test "Context sweep: 2 contexts, 16 decode steps" \
    python scripts/benchmark_full_sweep.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --sweeps context \
        --context-lengths 512,4096 \
        --decode-steps 16 \
        --warmup 1 \
        --repeats 1 \
        --output-dir results/grid/_stress_test

# Test 4: Strategy sweep (verifies kvpress hooks work with manual loop)
run_test "Strategy sweep: window_512 at 4K" \
    python scripts/benchmark_full_sweep.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --sweeps strategy \
        --strategies full_cache,window_512 \
        --strategy-contexts 4096 \
        --decode-steps 16 \
        --warmup 1 \
        --repeats 1 \
        --output-dir results/grid/_stress_test

# Test 5: Batch sweep (verifies batched input works)
run_test "Batch sweep: bs=1,4 at 4K" \
    python scripts/benchmark_full_sweep.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --sweeps batch \
        --batch-sizes 1,4 \
        --batch-context 4096 \
        --decode-steps 16 \
        --warmup 1 \
        --repeats 1 \
        --output-dir results/grid/_stress_test

# Test 6: Verify incremental JSONL was written
echo ""
echo "--- TEST: Incremental save verification ---"
JSONL_FILES=$(find results/grid/_stress_test -name "*.jsonl" 2>/dev/null)
if [ -z "$JSONL_FILES" ]; then
    echo "--- FAIL: No JSONL files found! ---"
    FAIL=$((FAIL + 1))
else
    TOTAL_LINES=0
    for f in $JSONL_FILES; do
        lines=$(wc -l < "$f")
        TOTAL_LINES=$((TOTAL_LINES + lines))
        echo "  $f: $lines rows"
    done
    if [ "$TOTAL_LINES" -gt 0 ]; then
        echo "  Total rows saved: $TOTAL_LINES"
        echo "--- PASS: Incremental save verification ---"
        PASS=$((PASS + 1))
    else
        echo "--- FAIL: JSONL files are empty! ---"
        FAIL=$((FAIL + 1))
    fi
fi

# Test 7: VRAM headroom check with largest feasible config
echo ""
echo "--- TEST: VRAM headroom (Phi-4 at 8K context) ---"
if python scripts/benchmark_full_sweep.py \
    --model microsoft/phi-4 \
    --sweeps context \
    --context-lengths 8192 \
    --decode-steps 16 \
    --warmup 1 \
    --repeats 1 \
    --skip-oom \
    --output-dir results/grid/_stress_test; then
    echo "--- PASS: VRAM headroom ---"
    PASS=$((PASS + 1))
else
    echo "--- FAIL: VRAM headroom (Phi-4 may not fit at higher contexts) ---"
    FAIL=$((FAIL + 1))
fi

# Cleanup
rm -rf results/grid/_stress_test

# Report
echo ""
echo "=========================================="
echo "STRESS TEST RESULTS"
echo "=========================================="
echo "PASSED: $PASS"
echo "FAILED: $FAIL"
echo ""
if [ "$FAIL" -gt 0 ]; then
    echo "!!! DO NOT LAUNCH FULL SWEEP — $FAIL test(s) failed !!!"
    exit 1
else
    echo "All tests passed. Safe to launch:"
    echo "  nohup bash scripts/run_all_models.sh > sweep_\$(date +%Y%m%d_%H%M).log 2>&1 &"
fi
echo "=========================================="
