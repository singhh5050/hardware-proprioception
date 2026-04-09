#!/bin/bash
# NCU profiling sweep: profile one decode step per (model, context_length).
#
# Usage:
#   bash scripts/run_ncu_sweep.sh              # full sweep (5 runs)
#   bash scripts/run_ncu_sweep.sh --minimal    # minimal sweep (3 runs)
#
# Requires: ncu (NVIDIA Nsight Compute CLI), torch, transformers, flash-attn
set -euo pipefail

OUTDIR="${NCU_OUTPUT_DIR:-results/ncu}"
mkdir -p "$OUTDIR"

# Define profiling matrix: "model:context_length"
if [[ "${1:-}" == "--minimal" ]]; then
    echo "=== Minimal sweep (3 runs) ==="
    RUNS=(
        "meta-llama/Llama-3.2-3B:1024"
        "meta-llama/Llama-3.2-3B:16384"
        "Qwen/Qwen2.5-7B:131072"
    )
else
    echo "=== Full sweep (5 runs) ==="
    RUNS=(
        "meta-llama/Llama-3.2-3B:1024"
        "meta-llama/Llama-3.2-3B:16384"
        "meta-llama/Llama-3.2-3B:131072"
        "Qwen/Qwen2.5-7B:4096"
        "Qwen/Qwen2.5-7B:131072"
    )
fi

echo "Output directory: $OUTDIR"
echo "Runs: ${#RUNS[@]}"
echo ""

for entry in "${RUNS[@]}"; do
    IFS=: read -r model ctx <<< "$entry"
    model_short="${model##*/}"
    tag="${model_short}_ctx${ctx}"

    echo "============================================================"
    echo "Profiling: ${model_short} @ ctx=${ctx}"
    echo "============================================================"

    # Run NCU with profiling disabled until cudaProfilerStart()
    ncu \
        --profile-from-start off \
        --set full \
        --target-processes all \
        --export "${OUTDIR}/${tag}" \
        --force-overwrite \
        python3 scripts/ncu_profile_decode.py \
            --model "$model" \
            --context-length "$ctx" \
            --output-dir "$OUTDIR"

    echo "--- Done: ${tag} ---"
    echo ""
done

# Export all .ncu-rep files to CSV for parsing
echo "============================================================"
echo "Exporting NCU reports to CSV..."
echo "============================================================"
for rep in "${OUTDIR}"/*.ncu-rep; do
    [ -f "$rep" ] || continue
    base="${rep%.ncu-rep}"
    echo "  Exporting: $(basename "$rep")"
    ncu --import "$rep" --csv --page raw > "${base}_raw.csv" 2>/dev/null || \
        echo "    WARNING: CSV export failed for $(basename "$rep")"
done

# Parse all CSVs into summary
echo ""
echo "============================================================"
echo "Parsing results..."
echo "============================================================"
python3 scripts/parse_ncu_results.py --input-dir "$OUTDIR"

echo ""
echo "=== All done! Results in ${OUTDIR}/ ==="
