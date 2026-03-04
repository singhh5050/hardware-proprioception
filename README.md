# Hardware Proprioception

Analytical roofline-based cost oracle for LLM inference on diverse hardware, with real accuracy evaluation of KV cache compression strategies.

## What this project does

LLM inference is memory-bound during decode. KV cache compression (evicting, quantizing, or offloading tokens) reduces memory pressure but can hurt accuracy. This project answers:

**At a given memory budget, does it matter *how* you choose which tokens to keep?**

It has two independent pipelines:

1. **Accuracy pipeline** — Runs real LLM generation on MATH-500 problems with 12 concrete KV cache strategies via [kvpress](https://github.com/simonask/kvpress). Measures whether the model still gets the right answer after cache compression.

2. **Latency pipeline** — Post-hoc roofline simulation that computes per-token decode latency for each strategy across 16 hardware configs. No GPU needed.

The two pipelines are independent: accuracy depends only on *which tokens survive* (eviction policy), while latency depends on *where surviving tokens live* (HBM/CPU/disk tier placement).

## Architecture

```
src/hwprop/
  specs.py          # 16 hardware configs, 14 model configs, memory tier specs
  cost_model.py     # Stateless roofline math: (hardware, model, kv_state) -> StepCost
  oracle.py         # Stateful RL interface: step(), reset(), budget tracking
  eval_pipeline.py  # Synthetic eval + latency replay helpers
  accuracy_eval.py  # Real accuracy eval with kvpress strategies
  sampling.py       # Synthetic hardware sampling for training
```

**Key design decisions:**
- All internal values in **bytes** and **FLOPS/s** (not GB or TFLOPS)
- Roofline: `wall_clock = max(hbm_time, compute_time) + cpu_transfer + disk_transfer`
- 3 memory tiers: HBM, CPU (DDR), Disk (NVMe) — transfers are additive (blocking)
- Dense models only (MoE deferred)

## Installation

```bash
# Core package (cost model + oracle)
pip install -e ".[dev]"

# With evaluation plotting
pip install -e ".[eval]"

# With real accuracy eval (requires GPU)
pip install -e ".[accuracy]"
```

## KV Cache Strategies

The accuracy pipeline evaluates 12 strategies, each using a different approach to manage the KV cache during generation:

| # | Strategy | What it does | Budget |
|---|----------|-------------|--------|
| 1 | `full_cache` | No compression (accuracy ceiling) | unlimited |
| 2 | `full_cache_int4` | Quantize all KV to INT4 | unlimited |
| 3 | `window_128` | StreamingLLM: 4 sinks + last 128 | 132 |
| 4 | `window_256` | StreamingLLM: 4 sinks + last 256 | 260 |
| 5 | `window_512` | StreamingLLM: 4 sinks + last 512 | 516 |
| 6 | `window_1024` | StreamingLLM: 4 sinks + last 1024 | 1028 |
| 7 | `h2o_128` | Keep 128 highest-attention tokens | 128 |
| 8 | `h2o_256` | Keep 256 highest-attention tokens | 256 |
| 9 | `h2o_512` | Keep 512 highest-attention tokens | 512 |
| 10 | `h2o_1024` | Keep 1024 highest-attention tokens | 1024 |
| 11 | `snapkv_512` | SnapKV scoring, keep 512 | 512 |
| 12 | `expected_attn_512` | Expected attention scoring, keep 512 | 512 |

**What these comparisons tell us:**
- **1 vs 2**: Does INT4 quantization alone hurt math reasoning?
- **3-6**: How much context can the model lose before accuracy breaks? (budget sweep)
- **3-6 vs 7-10**: At the same budget, does smart token selection beat recency? (selection policy)
- **5 vs 9 vs 11 vs 12**: Which attention-scoring algorithm wins at fixed budget=512?

All strategies use [kvpress](https://github.com/simonask/kvpress) `DecodingPress` to **remove** tokens from the cache during generation. This is eviction, not movement — tier placement (HBM/CPU/disk) is simulated post-hoc.

## Running the Evaluation

### Step 0: Smoke test (validates kvpress works with your model)

```bash
python smoke_test_kvpress.py
```

Tests baseline, StreamingLLM, and ExpectedAttention on a single hardcoded MATH problem. If this passes, proceed.

### Step 1: Real accuracy eval (requires GPU)

```bash
# Quick test (3 problems, 2 strategies)
python eval_accuracy.py --num-tasks 3 --strategies full_cache,window_512

# Full run (200 problems, all 12 strategies)
python eval_accuracy.py --num-tasks 200

# Skip latency simulation
python eval_accuracy.py --num-tasks 50 --skip-latency

# Split across GPUs (run each on a separate machine)
python eval_accuracy.py --num-tasks 200 --strategies full_cache,full_cache_int4,window_128,window_256,window_512,window_1024
python eval_accuracy.py --num-tasks 200 --strategies h2o_128,h2o_256,h2o_512,h2o_1024,snapkv_512,expected_attn_512
```

**Outputs** (in `eval_outputs_accuracy/`):
- `accuracy_results.jsonl` — per-problem results with full generated text
- `accuracy_by_strategy.png` — bar chart of accuracy per strategy
- `accuracy_vs_budget.png` — accuracy vs cache budget curves
- `latency_results.csv` — simulated latency across hardware configs
- `pareto_plot.png` — accuracy vs latency Pareto frontier

### Step 2: Latency-only simulation (no GPU needed)

```bash
python eval_naive_strategies.py --hardware H100_SXM --model LLaMA-3.1-8B
```

## Hardware Catalog

16 hardware configs spanning datacenter GPUs, TPUs, and edge devices:

**NVIDIA:** A100 80GB, H100 SXM, H200, B200, B300, L40S, RTX 5090
**AMD:** MI300X, MI325X, MI350X
**Google:** TPU v5e, TPU v6e, TPU v7
**Intel:** Gaudi 3
**Apple:** M4 Max
**Qualcomm:** Snapdragon X Elite

## Model Catalog

14 dense model configs: Tiny-1B, LLaMA-3.1-8B, LLaMA-3.1-70B, LLaMA-3.2-3B, LLaMA-3.3-70B, Qwen2.5-14B, Qwen2.5-72B, Qwen3-4B, Qwen3-8B, Qwen3-32B, Gemma-3-4B, Gemma-3-27B, Phi-4-14B, Phi-4-mini-3.8B

## Tests

```bash
pytest tests/ -q
```

125 tests covering cost model, oracle, eval pipeline, accuracy scoring, strategy registry, and latency replay.

## Project Structure

```
hardware-proprioception/
  src/hwprop/           # Core package
  tests/                # Test suite (125 tests)
  eval_accuracy.py      # Real accuracy eval runner
  eval_naive_strategies.py  # Synthetic eval runner
  smoke_test_kvpress.py # kvpress validation
  pyproject.toml        # Build config
```
