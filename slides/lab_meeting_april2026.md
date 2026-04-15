# Hardware Proprioception for LLM Inference
## CoCo Lab Meeting — April 2026

---

## Slide 1: Title

**Hardware Proprioception for LLM Inference**
*Building a calibrated simulator so models can learn to manage their own memory*

Harsh Singh, Michael Li, Kanishk Gandhi, Nicole Ma
Noah Goodman's Computation & Cognition Lab, Stanford

**Speaker notes:**
The term "proprioception" comes from biology — it's the sense of your own body's position and constraints. We're applying the same idea to LLMs: what if the model knew what hardware it's running on and could adapt its inference behavior accordingly? Today I'll walk you through the simulator infrastructure we've built, the 979 real latency measurements we collected, and what we've learned about when and why KV cache eviction helps.

---

## Slide 2: The Problem — Same Model, Different Hardware

**FIGURE: `results/plots/new_heatmap.png`**

9 models × 5 GPUs. Each cell = measured decode latency (ms/step) at 8K context.

**Speaker notes:**
Look at this heatmap. Phi-4 on H200 takes 29ms per decode step. The same model on A40 takes 73ms — 2.5x slower. And Gemma-3-1B, which is a tiny 1B model, takes 29ms on A40 — almost as slow as Phi-4 on H200 despite being 14x smaller. The point is: the "optimal" inference strategy depends entirely on the hardware. Yet today, the model has no idea which chip it's running on. Every deployed model uses the same fixed KV cache management regardless of whether it's on an H200 with 4.8 TB/s bandwidth or an A40 with 696 GB/s.

---

## Slide 3: The Vision — Proprioception for LLMs

**Problem:** Models should run efficiently on different hardware. Current solution: engineers hand-design hardware-specific methods. New chip → start over.

**Us:** What if the model knew its hardware and could decide how it ran?

- Tell the model the hardware specs (~5 numbers: bandwidth, memory, cache size, compute, launch overhead)
- The model learns via RL to make decisions about KV cache management (evict, quantize, offload)
- **Reward = accuracy** given a fixed latency budget from the simulator
- Train on measured hardware, test on held-out hardware

The simulator IS the reward function for RL.

**Speaker notes:**
This is Kanishk's framing of the problem. People already think about hardware when designing architectures — MQA, GQA, RMSNorm replacing LayerNorm, dropping biases — these are all roofline-motivated design choices. But the models themselves don't adapt at inference time. We want to close that loop. The RL agent sees the hardware specs, sees the current KV cache state, and decides what to evict. The simulator tells it how fast that decision would be on the target hardware. Today's talk is mostly about building that simulator and validating it.

---

## Slide 4: Why Decode is Memory-Bound

**GPU Memory Hierarchy:**
```
SRAM / L2 Cache    ← 6-96 MB, on-chip (fast)
       ↓
HBM                ← 48-192 GB, 0.7-4.8 TB/s
       ↓
CPU RAM            ← via PCIe (slow)
       ↓
Disk / NVMe        ← 5 GB/s (very slow)
```

**Key fact:** At batch size 1, every decode step loads ALL model weights + KV cache from HBM. Arithmetic intensity ≈ 1. The GPU's compute units are ~99.7% idle, waiting for data.

**Critical batch size** B_crit = FLOPS_peak / BW_HBM:
- H100: ~269, A100: ~142, A40: ~lots

Below B_crit, decode is fully memory-bandwidth bound. Reducing KV cache bytes directly reduces latency.

**Speaker notes:**
For the math folks in the room — think of this as a constrained optimization problem. You have data spread across memory tiers with different bandwidths. Each decode step needs to read all model weights (~6-29 GB) plus the KV cache (grows with context). At batch=1, the arithmetic intensity is about 1 FLOP per byte loaded. H100 can do 990 TFLOPS but only reads 3.35 TB/s — so you need a batch of ~269 to saturate the compute. At batch=1, the GPU is just a very expensive memory reader.

---

## Slide 5: Roofline Analysis — The Theoretical Lower Bound

**Roofline model:**
$$T = \max\left(\frac{\text{Bytes}}{\text{Bandwidth}},\; \frac{\text{FLOPs}}{\text{FLOPS}_{\text{peak}}}\right)$$

At batch=1, decode is memory-bound:
$$T \approx \frac{\text{param\_bytes} + \text{KV\_bytes}}{\text{HBM\_bandwidth}}$$

**FIGURE: `results/plots/roofline_vs_measured_5gpu.png`**

LLaMA-3.2-3B across all 5 GPUs. Red triangles = roofline prediction. Colored circles = measured reality.

**Speaker notes:**
The roofline gives a theoretical lower bound. Look at the gap — on H200 at short context, roofline predicts ~4ms but we measure ~20ms. That's a 5x gap. On A40 at 128K context, roofline predicts ~7ms, we measure ~50ms. The gap is dominated by kernel launch overhead (~16-27ms per step, constant) and Flash Attention tile scan overhead (superlinear with context length). The roofline is useful for understanding WHY latency scales the way it does, but it's useless for predicting actual values. This motivated our search for better models.

---

## Slide 6: What Roofline Misses — The Overhead Gap

**FIGURE: `results/plots/simulation_vs_benchmark.png`**

Left: H100 context sweep — measured (blue), calibrated simulator (green), roofline (red).
Right: A100 strategy comparison — measured vs calibrated.

The gap comes from:
| Overhead | Source | Magnitude |
|----------|--------|-----------|
| Kernel launch | CUDA dispatch, sync, Python | ~16-27 ms constant per step |
| FA2 tile scan | Flash Attention processes KV in 64-token tiles | Superlinear: $(N/64)^{1.5}$ |
| KV BW degradation | KV scattered across HBM, cache misses | BW drops 50-70% at 128K |
| Memory allocator | PyTorch allocator overhead | ~log₂(N) |

**Speaker notes:**
This figure is from our earlier calibration work on H100 with LLaMA-3.2-3B. The green line (calibrated simulator) tracks the blue line (measured) closely — that's our overhead-corrected model fitting the data. The red line (roofline) sits at the bottom, 5-10x too low. We spent weeks trying to make the overhead correction generalizable — fitting universal constants that work across all GPUs and models. Spoiler: we eventually moved to a different approach, but I'll walk you through everything we tried.

---

## Slide 7: KV Cache Strategies — What We Tested

12 strategies evaluated on 200 MATH-500 tasks (Qwen2.5-Math-7B-Instruct):

- **Baselines:** full_cache, INT8 quantization
- **StreamingLLM (window):** keep N most recent tokens + 4 attention sinks
- **H2O (Heavy-Hitter):** keep tokens with highest cumulative attention weight
- **SnapKV:** observation-window scoring from prefill
- **ExpectedAttention:** statistical prediction of future attention patterns

All strategies (except H2O) work with Flash Attention 2 — no eager attention needed.

**Speaker notes:**
We picked these because they span the space of KV cache management: do nothing, quantize (half the bytes), evict by recency (window), evict by attention importance (H2O, SnapKV), and evict by predicted future attention (ExpectedAttn). Each is a different inductive bias about which tokens matter. H2O requires eager attention which completely changes the latency profile, so we excluded it from our latency benchmarks. The other strategies all work with FA2 via kvpress forward hooks.

---

## Slide 8: Accuracy Results

**FIGURE: `results/plots/accuracy_by_strategy.png`**

**Speaker notes:**
The ceiling is 77% (full cache). INT8 quantization costs only 0.5% — essentially free. Window-1024 matches the ceiling exactly. At 512-token budget, SnapKV and ExpectedAttn outperform window by 3-7%. Below 256 tokens, accuracy collapses — window-128 drops to 28.5%. The surprising result: H2O underperforms simple windowing at equal budgets. This may be a kvpress implementation artifact (eager attention overhead), but it means the "smartest" eviction strategy isn't always the best in practice. The key insight for the simulator: at 1024-token budget you lose nothing; at 512 you need attention-score-based methods; below 256 it's catastrophic regardless of strategy.

---

## Slide 9: Compression vs Accuracy — The Pareto Frontier

**FIGURE: `results/plots/compression_vs_accuracy.png`**

X-axis: tokens kept / tokens generated (compression ratio). Y-axis: MATH-500 accuracy.

**Speaker notes:**
This is the accuracy landscape the simulator needs to navigate. The curve is NOT monotone — at the same compression ratio, SnapKV significantly outperforms window and H2O. The Pareto frontier goes: full_cache → window_1024 → snapkv_512 → steep cliff. A learned policy should find points on or near this frontier. The shape of this curve is the empirical ground truth — without it, you're guessing whether eviction will hurt accuracy.

---

## Slide 10: Simulator Design Principles — Three Properties

We need the simulator to satisfy three properties for RL training:

**1. Monotonicity:** More tokens in cache = higher latency. Always. If the agent evicts tokens, latency must decrease.

**2. Hardware sensitivity:** Eviction benefit is larger on low-bandwidth hardware. A40 should save more from evicting than H200.

**3. Crossover correctness:** At short context, eviction doesn't help (overhead dominates). At long context, it helps a lot. The crossover point depends on the hardware.

These aren't assumptions — they're testable predictions that the simulator must get right.

**Speaker notes:**
These three properties are what make the simulator usable as an RL reward function. If monotonicity fails, the agent gets contradictory rewards. If hardware sensitivity fails, the agent can't learn different policies for different hardware. If the crossover is wrong, the agent will evict when it shouldn't (or not evict when it should). We validated all three against our 979 measured data points. Let me show you.

---

## Slide 11: Property 1 — Monotonicity (Validated)

**FIGURE: `results/plots/monotonicity_demo.png`**

Left: Raw measurements have noise at short context (±3%). Middle and right: same issue.
Blue solid line: isotonic regression corrects the noise while preserving the true signal.

**Key result:** 40/45 curves had minor non-monotonicity from measurement noise. Isotonic regression fixes all violations with <5% max correction. After correction, ALL 45 curves are strictly monotonic.

**Speaker notes:**
The red dashed lines show raw measurements — look at the short-context regime where values bounce around 10-11 ms on H100. This is because at short context, the KV cache is tiny and the measurement is dominated by kernel launch overhead noise. The blue lines show the isotonic regression correction — it flattens the noisy plateau and leaves the long-context upswing untouched. The max correction is 4.7%, which is well within measurement variance. This guarantees the RL agent always gets a consistent signal: evicting tokens reduces or maintains (never increases) latency.

---

## Slide 12: Property 2 — Hardware Sensitivity (Validated)

**FIGURE: `results/plots/hardware_sensitivity.png`**

Eviction savings (full_cache − window_512) at 64K context, LLaMA-3.2-3B:
- H200 (4.8 TB/s): saves 7.0 ms
- H100 (3.35 TB/s): saves 8.1 ms
- A100 (2.0 TB/s): saves 13.8 ms
- L40S (864 GB/s): saves 15.3 ms
- A40 (696 GB/s): saves **30.4 ms**

**Speaker notes:**
Perfectly monotonic with hardware speed. The A40 saves 4.3x more from eviction than the H200. The physics is clear: on slow hardware, KV loading time is a larger fraction of total step time, so removing KV bytes has more impact. This means a hardware-aware RL agent should evict more aggressively on A40 than on H200 — and the simulator correctly reflects this. An agent that learns "always evict to 512 tokens" would leave performance on the table on H200 (where eviction barely helps) and would be correctly aggressive on A40.

---

## Slide 13: Property 3 — Crossover Correctness (Validated)

**FIGURE: `results/plots/eviction_crossover.png`**

Three panels: H200 (fast), A100 (medium), A40 (slow).
Y-axis: window_512 latency − full_cache latency. Positive = eviction is slower. Negative = eviction helps.

**Key finding:** Crossover happens EARLIER on slow hardware:
- A40: eviction starts helping at ~4K-8K tokens
- A100: ~8K-16K tokens
- H200: ~16K-64K tokens

**Speaker notes:**
This was initially surprising — we expected larger L2 cache to mean later crossover. Instead, the crossover is driven primarily by bandwidth. On A40 with 696 GB/s, even moderate KV caches bottleneck bandwidth, so the overhead of eviction scoring is quickly repaid by reduced KV loading time. On H200 with 4.8 TB/s, KV loading is fast even at 16K context, so eviction overhead dominates until very long sequences. The RL agent needs to learn this: on slow hardware, start evicting early; on fast hardware, only evict at very long context.

---

## Slide 14: What We Tried — Approach 1: Pure Roofline

**Equation:** $T = \frac{\text{param\_bytes}}{\text{BW}} + \frac{\text{KV\_bytes}}{\text{BW}}$

No parameters. Fully generalizable from spec sheets.

**Result:** 80.9% MAE, Spearman ρ = 0.588

Predicts 1-6 ms when reality is 10-110 ms. Misses all overhead. 41% of pairwise rankings wrong.

**Speaker notes:**
The roofline is useful for intuition — it tells you WHY larger models are slower and WHY longer context costs more. But it systematically underestimates by 5-12x because it doesn't account for kernel launch overhead, Flash Attention tile scanning, or memory allocator pressure. These overheads are 80-95% of actual latency at short context. We can't use this as an RL reward because it would tell the agent "everything is fast, eviction barely matters."

---

## Slide 15: What We Tried — Approach 2: Universal Equation

**Equation:** $T = t_{\text{launch}} + \frac{\text{param\_bytes}}{\text{BW}} + \frac{\text{KV\_bytes}}{\text{BW}_{\text{eff}}}$

where $\text{BW}_{\text{eff}} = \frac{\text{BW}}{1 + \alpha \cdot \left(\frac{\text{KV\_bytes}}{\text{L2}}\right)^\beta}$

**2 universal constants** (α=0.00163, β=1.04) + per-GPU t_launch.

Physics: effective HBM bandwidth degrades as KV cache working set exceeds L2 cache.

**Cross-validation:**
- Leave-one-GPU-out: 31.8% MAE (predict new GPU from spec sheet + 1 measurement)
- Leave-one-model-out: 20.0% MAE (predict new model from config.json)
- Spearman ρ = 0.746

**Speaker notes:**
This is where we spent the most time. The idea was elegant: 2 universal constants that describe how bandwidth degrades when the KV cache exceeds L2 cache. We fit on 4 GPUs and predicted the 5th, fit on 8 models and predicted the 9th. The results were honest but not good enough. 20% average error, 57% worst case on LLaMA-3.2-1B (a tiny model where launch overhead is everything and the KV term contributes almost nothing). The equation structurally can't capture model-specific kernel launch patterns. We tried adding a head-count exponent (gamma), tried per-head bandwidth modeling — nothing got below 20% on held-out data.

---

## Slide 16: What We Tried — Approach 3: Measured Lookup Table

**What:** Benchmark real latency. 979 measurements across 9 models × 5 GPUs.
- Context sweep (512 to 128K, 9 points)
- Strategy sweep (4 strategies × 4 context lengths)
- Batch sweep (1 to 256)
- 512 decode steps, FA2, 3 repeats, incremental save

Linear interpolation between measured points. Isotonic regression for monotonicity.

**Result:** 0.4% MAE, Spearman ρ = 0.999

275/318 points within 1% error. ALL within 5%.

**Speaker notes:**
We stopped trying to fit equations and just measured everything. $20 on RunPod, one overnight run, 5 GPUs in parallel. The lookup table gives essentially perfect predictions for any (GPU, model, context) triple we've benchmarked. The tradeoff: zero generalization to new hardware or models without running new benchmarks. But for RL training, you only need the hardware and models you're training on — and those are exactly the ones you benchmark. A 30-minute benchmark per new GPU gives you the full context curve. That's the practical workflow.

---

## Slide 17: Head-to-Head Comparison

**FIGURE: `results/plots/three_methods_comparison.png`**

Three scatter plots: predicted vs measured, log-log scale. Dashed line = perfect prediction.

| Method | MAE | Spearman ρ | Generalizes? |
|--------|-----|-----------|-------------|
| Roofline | 80.9% | 0.588 | Full (spec sheets only) |
| Universal eq | 20.9% | 0.746 | Partial (31.8% LOO-GPU) |
| Lookup table | 0.4% | 0.999 | Measured pairs only |

**Speaker notes:**
The visual tells the whole story. Left: roofline points scatter below the diagonal, 10x too low. Middle: universal equation is better but has wide spread, especially for small models. Right: lookup table points sit perfectly on the diagonal. For RL training, we need ρ > 0.95 — the agent must know which actions are better than others. Only the lookup table achieves this. The universal equation is still useful for one thing: rough extrapolation to hardware you haven't benchmarked, for example predicting "will eviction help on a B200?" without spending $6/hr to benchmark it.

---

## Slide 18: The Benchmark Grid

**FIGURE: `results/plots/all_context_curves.png`**

All 9 models × all 5 GPUs, context sweep from 512 to 128K tokens.

**Coverage:**
- 5 GPUs: H200, H100 SXM, A100 80GB, L40S, A40
  - 3 architectures (Hopper, Ampere, Ada)
  - Bandwidth: 696 GB/s → 4.8 TB/s
  - L2 cache: 6 MB → 96 MB
- 9 models: KV heads 1→32, head_dim 64→256, 1B→14B params
- 979 total data points, $20 compute cost

**Speaker notes:**
This is all measured data — no simulation. Each line is one (GPU, model) pair. You can see how different architectures create different curve shapes. On H200 (leftmost panel), most models are flat until 16K-32K then climb steeply. On A40 (rightmost), models diverge earlier and the spread is wider. Gemma-3-1B (brown line, bottom) is the smallest KV footprint — 1 KV head, barely grows with context. Phi-4 (dark line, top) has 10 KV heads and 200KB per token — it climbs fastest. The diversity in these curves is exactly what the RL agent needs to learn from.

---

## Slide 19: How Hardware Changes the Picture

**FIGURE: `results/plots/hardware_comparison.png`**

Left: LLaMA-3.2-3B (8 KV heads, 112 KB/tok) — moderate KV size
Right: Phi-4 (10 KV heads, 200 KB/tok) — large KV size

**Speaker notes:**
Same model, 5 different GPUs. Two interesting observations. First: at short context (512-4K), the curves are flat and the gap between GPUs is entirely from launch overhead differences — H100 at 16ms vs A40 at 27ms. This is not bandwidth-related; it's kernel launch and framework overhead. Second: at long context, the curves diverge dramatically. For Phi-4 at 16K (its max context), A40 is at 73ms while H100 is at 27ms — almost 3x. That gap is the bandwidth-dependent KV loading cost that the RL agent can reduce through eviction.

---

## Slide 20: Eviction Strategies — Real Measurements

**FIGURE: `results/plots/strategy_comparison_5gpu.png`**

4 strategies × 5 GPUs at 64K context with LLaMA-3.2-3B.

**Speaker notes:**
Now we're combining the accuracy story with the latency story. At 64K context on A40, full_cache takes about 50ms and window_512 takes about 20ms — that's a 60% latency reduction. But window_512 at 64K means you're keeping only 512 out of 64K tokens — massive eviction. Is it worth it? From the accuracy eval, window_512 at that compression ratio gets ~72% accuracy vs 77% ceiling. So you're trading 5% accuracy for 60% speedup. Whether that trade is worth it depends on the hardware (A40: yes, H200: less so) and the application (latency-critical chat: yes, offline batch processing: no). That's exactly the decision we want the RL agent to learn.

---

## Slide 21: Batch Throughput — The Other Dimension

**FIGURE: `results/plots/batch_throughput.png`**

Throughput (tok/s) vs batch size at 8K context, 3 GPUs, all 9 models.

**Speaker notes:**
Batch size is the other lever beyond eviction. At batch=1, H200 delivers 30-50 tok/s per model. At batch=16, it reaches 400-1200 tok/s — a 10-30x throughput increase. But notice the curves are sublinear: going from bs=1 to bs=4 roughly 4x's throughput (ideal scaling), but bs=4 to bs=16 only gives 2-3x more. This is because at higher batch sizes, you start approaching the memory-bound to compute-bound transition. The RL agent should consider batch size in its decisions — at batch=1, evicting KV saves bandwidth; at batch=16, the KV is already amortized across sequences and eviction helps less.

---

## Slide 22: Next Steps — The RL Agent

**The lookup table IS the RL environment.**

```python
from hwprop.lookup_table import LookupCostModel
cm = LookupCostModel.from_grid("results/grid")

# Agent evicts tokens → cache_size drops → interpolate new latency
latency = cm.step_cost("H200", "meta-llama/Llama-3.2-3B", cache_size=16000)
# → 20.49 ms (measured, interpolated)
```

**What's next:**
1. **Train the RL agent** on the lookup table. State = (hardware specs, KV state, position). Action = (keep, evict, quantize). Reward = accuracy − λ·latency.
2. **Expand the grid** — more GPUs (B200, MI300X), more models, longer contexts.
3. **Improve the universal equation** — if we get LOO below 10%, we skip benchmarking new hardware.
4. **Beyond KV cache** — precision selection, speculative decoding, distributed inference decisions.

**The punchline:** We built a physics engine for LLM inference. 979 real measurements, validated against three design properties. Now we train the agent.

**Speaker notes:**
The simulator is done — it's the lookup table with 0.4% error and ρ=0.999. Michael can import it directly into the RL training loop. The agent sees the hardware spec, makes a decision about what to evict, and gets a reward based on real measured latency. We've validated that the three properties hold: monotonicity (always consistent rewards), hardware sensitivity (different optimal policies per GPU), and crossover correctness (the agent learns when eviction is worth it vs when it's not). The $20 and one overnight run to build this table means we can iterate quickly — add a new model or GPU, rerun the benchmark, update the table. The universal equation sits alongside as a rough predictor for hardware we haven't benchmarked yet, but for RL training the lookup table is the ground truth.

---

## Appendix: Benchmark Infrastructure

**For questions about methodology:**

- All runs: FA2 attention, bf16, manual prefill+decode loop (unified timing for all strategies)
- 512 decode steps per run, 1 warmup + 3 timed repeats
- Incremental JSONL save after every config (survives crashes)
- Conservative VRAM headroom (max(8GB, 15% of GPU) + warmup memory check at 92%)
- kvpress DecodingPress hooks for strategy runs — works in manual forward loop, verified on all models
- SnapKV and ExpectedAttn compute their own attention scores (don't need materialized attention weights), compatible with FA2
- Isotonic regression via sklearn for monotonicity guarantee

**The code:** `scripts/benchmark_full_sweep.py`, `src/hwprop/lookup_table.py`
