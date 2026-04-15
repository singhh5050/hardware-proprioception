#!/bin/bash
# Run full benchmark sweep for all 9 models on the current GPU.
# Designed for unattended nohup runs — no interactive prompts.
#
# Setup (run once per pod):
#   cd /workspace
#   git clone https://<TOKEN>@github.com/singhh5050/hardware-proprioception.git
#   cd hardware-proprioception
#   pip install -e ".[dev]"
#   pip install transformers accelerate torch flash-attn kvpress
#   git config user.email "harsh@stanford.edu" && git config user.name "Harsh Singh"
#
# Usage:
#   nohup bash scripts/run_all_models.sh > sweep_$(date +%Y%m%d_%H%M).log 2>&1 &
#   tail -f sweep_*.log
#
# Results saved incrementally to results/grid/<model>/<hw_key>.jsonl
# If the script dies, all completed configs are preserved.

# Do NOT use set -e — we want to continue past individual model failures
# set -e

MODELS=(
    "google/gemma-3-1b-it"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B-Instruct"  # DISABLED: 28GB download fills disk on 48GB GPUs
    "tiiuae/Falcon3-7B-Base"
    # "THUDM/glm-4-9b"  # DISABLED: incompatible with transformers 5.x (ChatGLMConfig.max_length missing)
    "microsoft/phi-4"
)

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
START_TIME=$(date +%s)

echo "=========================================="
echo "Full benchmark sweep"
echo "GPU: $GPU_NAME"
echo "Models: ${#MODELS[@]}"
echo "Started: $(date)"
echo "PID: $$"
echo "=========================================="

# Preflight: verify all models load
echo ""
echo "--- PREFLIGHT CHECKS ---"
PREFLIGHT_FAILED=0
for MODEL in "${MODELS[@]}"; do
    echo -n "  Checking $MODEL... "
    if python scripts/benchmark_full_sweep.py \
        --model "$MODEL" \
        --sweeps preflight \
        --output-dir results/grid 2>&1 | grep -q "Preflight OK"; then
        echo "OK"
    else
        echo "FAILED"
        PREFLIGHT_FAILED=$((PREFLIGHT_FAILED + 1))
    fi
done
echo "--- PREFLIGHT: $PREFLIGHT_FAILED failures ---"
if [ "$PREFLIGHT_FAILED" -gt 0 ]; then
    echo "WARNING: $PREFLIGHT_FAILED model(s) failed preflight."
    echo "Continuing in 10 seconds... (Ctrl-C to abort)"
    sleep 10
fi
echo ""

# Track results
SUCCEEDED=()
FAILED=()
TOTAL_ROWS=0

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_NUM=$((i + 1))
    MODEL_SHORT=$(echo "$MODEL" | rev | cut -d/ -f1 | rev)

    echo ""
    echo "############################################################"
    echo "# MODEL ${MODEL_NUM}/${#MODELS[@]}: $MODEL"
    echo "# Started: $(date)"
    ELAPSED=$(( $(date +%s) - START_TIME ))
    echo "# Elapsed: $((ELAPSED / 60))m$((ELAPSED % 60))s"
    echo "############################################################"

    python scripts/benchmark_full_sweep.py \
        --model "$MODEL" \
        --output-dir results/grid \
        --decode-steps 512 \
        --warmup 1 \
        --repeats 3 \
        --skip-oom
    RC=$?

    if [ "$RC" -eq 0 ]; then
        SUCCEEDED+=("$MODEL")
        echo ">>> SUCCESS: $MODEL"
    else
        FAILED+=("$MODEL")
        echo ">>> FAILED: $MODEL (exit code $RC)"
    fi

    # Count total rows across all JSONL files
    TOTAL_ROWS=$(find results/grid -name "*.jsonl" -exec cat {} + 2>/dev/null | wc -l)
    echo ">>> Total rows saved so far: $TOTAL_ROWS"
    echo ">>> $(date)"
done

# Final report
END_TIME=$(date +%s)
TOTAL_ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "=========================================="
echo "SWEEP COMPLETE"
echo "=========================================="
echo "GPU: $GPU_NAME"
echo "Duration: $((TOTAL_ELAPSED / 3600))h$(( (TOTAL_ELAPSED % 3600) / 60 ))m"
echo "Total rows: $TOTAL_ROWS"
echo ""
echo "Succeeded (${#SUCCEEDED[@]}/${#MODELS[@]}):"
for M in "${SUCCEEDED[@]}"; do
    echo "  ✓ $M"
done
if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed (${#FAILED[@]}/${#MODELS[@]}):"
    for M in "${FAILED[@]}"; do
        echo "  ✗ $M"
    done
fi
echo ""
echo "Results in results/grid/"
echo "Push manually:"
echo "  git add results/grid/ && git commit -m 'result: sweep on $GPU_NAME' && git push"
echo "=========================================="
