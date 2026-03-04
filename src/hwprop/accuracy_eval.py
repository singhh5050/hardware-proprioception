"""Real LLM accuracy evaluation with kvpress KV cache strategies.

Runs concrete KV cache compression strategies (StreamingLLM, H2O, SnapKV, etc.)
on MATH problems using a real model (e.g., Qwen2.5-Math-7B-Instruct) and scores
answers against ground truth.

This module handles:
  - MATH-500 dataset loading and answer extraction
  - Strategy registry (12 strategies covering baselines, window, H2O, SnapKV)
  - Generation with kvpress DecodingPress
  - Answer scoring via \\boxed{} extraction
  - JSONL I/O for results
  - Plotting (accuracy by strategy, accuracy vs budget, Pareto)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MathTask:
    """A single MATH evaluation problem."""

    task_id: str
    problem: str
    ground_truth_solution: str
    ground_truth_answer: str
    level: str | None = None


@dataclass(frozen=True)
class AccuracyResult:
    """Result of evaluating one strategy on one task."""

    strategy_name: str
    task_id: str
    generated_text: str
    extracted_answer: str | None
    ground_truth_answer: str
    correct: bool
    tokens_generated: int
    prompt_tokens: int
    cache_size_at_end: int
    peak_memory_mb: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AccuracyResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class StrategyConfig:
    """Describes how to set up KV cache compression for generation."""

    name: str
    description: str
    budget_tokens: int | None  # target cache size (None = unlimited)
    # Set up lazily — kvpress imports only happen when actually generating
    press_factory: Callable | None = None  # () -> press context manager
    quantized: bool = False


# ---------------------------------------------------------------------------
# MATH-500 loading
# ---------------------------------------------------------------------------
def load_math_tasks(
    num_tasks: int = 200,
    seed: int = 42,
    dataset_name: str = "HuggingFaceH4/MATH-500",
) -> list[MathTask]:
    """Load MATH problems from Hugging Face, sample with fixed seed.

    Returns a deterministic subset of MathTask objects.
    """
    hf_home = os.path.join(os.getcwd(), ".hf_home")
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required. Install with: pip install datasets"
        ) from exc

    try:
        ds = load_dataset(dataset_name, split="test")
    except (ValueError, KeyError):
        # Some datasets only have "train" split
        ds = load_dataset(dataset_name, split="train")
    import numpy as np

    rng = np.random.default_rng(seed)
    n = min(num_tasks, len(ds))
    indices = rng.choice(len(ds), size=n, replace=False)

    tasks = []
    for idx in indices:
        row = ds[int(idx)]
        problem = str(row.get("problem", ""))
        solution = str(row.get("solution", ""))
        # Prefer the direct "answer" field if available (MATH-500 has it),
        # otherwise extract from the solution text.
        answer = str(row["answer"]) if "answer" in row else (extract_boxed_answer(solution) or "")
        level = row.get("level")

        tasks.append(
            MathTask(
                task_id=f"math_{int(idx):04d}",
                problem=problem,
                ground_truth_solution=solution,
                ground_truth_answer=answer,
                level=str(level) if level is not None else None,
            )
        )
    return tasks


# ---------------------------------------------------------------------------
# Answer scoring
# ---------------------------------------------------------------------------
def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} from text, handling nested braces."""
    # Find all \boxed{ positions
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    # Take the last match and find the matching closing brace
    last_match = matches[-1]
    start = last_match.end()  # position after \boxed{
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1

    if depth != 0:
        return None
    return text[start : pos - 1]


def normalize_math_answer(answer: str) -> str:
    """Normalize a math answer for comparison."""
    s = answer.strip()
    # Remove \text{...} wrappers
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    # Remove \mathrm{...} wrappers
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    # Remove \left and \right
    s = s.replace("\\left", "").replace("\\right", "")
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Remove trailing period
    s = s.rstrip(".")
    # Remove dollar signs
    s = s.replace("$", "")
    return s


def is_correct(generated_text: str, ground_truth_answer: str) -> bool:
    """Check if generated text contains the correct answer."""
    extracted = extract_boxed_answer(generated_text)
    if extracted is None:
        return False
    return normalize_math_answer(extracted) == normalize_math_answer(ground_truth_answer)


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------
def get_strategies(decision_interval: int = 64) -> dict[str, StrategyConfig]:
    """Build all 12 evaluation strategies.

    Strategies are constructed with lazy factories so that kvpress imports
    only happen when actually generating (not when running CPU-only tests).
    """
    strategies: dict[str, StrategyConfig] = {}

    # 1. full_cache — no compression
    strategies["full_cache"] = StrategyConfig(
        name="full_cache",
        description="No compression (accuracy ceiling)",
        budget_tokens=None,
    )

    # 2. full_cache_int8 — quantize all KV to INT8 via HQQ
    # Uses cache_implementation="quantized" in generate() rather than
    # constructing QuantizedCache directly (which requires model config).
    # INT8+HQQ avoids logit spikes with Qwen2.5's GQA heads (INT4 causes
    # degenerate repetition). residual_length=128 keeps sink tokens unquantized.
    strategies["full_cache_int8"] = StrategyConfig(
        name="full_cache_int8",
        description="Quantize all KV to INT8 (HQQ, residual=128)",
        budget_tokens=None,
        quantized=True,
    )

    # 3-6. StreamingLLM windows
    for size, label in [(128, "window_128"), (256, "window_256"),
                        (512, "window_512"), (1024, "window_1024")]:
        target = size + 4  # 4 sink tokens + window

        def _make_streaming_factory(ts=target, di=decision_interval):
            def factory():
                from kvpress import DecodingPress, StreamingLLMPress
                return DecodingPress(
                    StreamingLLMPress(n_sink=4),
                    target_size=ts,
                    compression_interval=di,
                )
            return factory

        strategies[label] = StrategyConfig(
            name=label,
            description=f"StreamingLLM: 4 sinks + last {size}",
            budget_tokens=target,
            press_factory=_make_streaming_factory(),
        )

    # 7-10. H2O (Heavy Hitter Oracle) — observed attention
    for size, label in [(128, "h2o_128"), (256, "h2o_256"),
                        (512, "h2o_512"), (1024, "h2o_1024")]:
        def _make_h2o_factory(ts=size, di=decision_interval):
            def factory():
                from kvpress import DecodingPress, ObservedAttentionPress
                return DecodingPress(
                    ObservedAttentionPress(),
                    target_size=ts,
                    compression_interval=di,
                )
            return factory

        strategies[label] = StrategyConfig(
            name=label,
            description=f"H2O: keep {size} highest-attention tokens",
            budget_tokens=size,
            press_factory=_make_h2o_factory(),
        )

    # 11. SnapKV at 512
    def _snapkv_factory():
        from kvpress import DecodingPress, SnapKVPress
        return DecodingPress(
            SnapKVPress(window_size=32),
            target_size=512,
            compression_interval=decision_interval,
        )

    strategies["snapkv_512"] = StrategyConfig(
        name="snapkv_512",
        description="SnapKV scoring, keep 512",
        budget_tokens=512,
        press_factory=_snapkv_factory,
    )

    # 12. ExpectedAttention at 512
    def _expected_attn_factory():
        from kvpress import DecodingPress, ExpectedAttentionPress
        return DecodingPress(
            ExpectedAttentionPress(),
            target_size=512,
            compression_interval=decision_interval,
        )

    strategies["expected_attn_512"] = StrategyConfig(
        name="expected_attn_512",
        description="Expected attention scoring, keep 512",
        budget_tokens=512,
        press_factory=_expected_attn_factory,
    )

    return strategies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _needs_eager_attention(strategies: dict[str, StrategyConfig]) -> bool:
    """Check if any strategy requires materialized attention weights.

    ObservedAttentionPress (H2O) asserts attentions is not None — this only
    works with attn_implementation="eager" (SDPA/FlashAttention don't
    materialize attention weight tensors).
    """
    return any(name.startswith("h2o_") for name in strategies)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------
def format_math_prompt(tokenizer, problem: str):
    """Format a MATH problem using Qwen2.5-Math-Instruct chat template.

    Returns input_ids tensor (on CPU).
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Please reason step by step, and put your final answer "
                "within \\boxed{}."
            ),
        },
        {"role": "user", "content": problem},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt").input_ids


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_with_strategy(
    model,
    tokenizer,
    input_ids,
    strategy: StrategyConfig,
    max_new_tokens: int = 2048,
) -> tuple[str, int, int]:
    """Generate text using the given strategy.

    Returns (generated_text, tokens_generated, cache_size_at_end).

    Three generation paths:
      A) kvpress press — registers forward hooks via context manager.
         DecodingPress overrides forward_hook to handle decoding-phase
         compression. Validated by smoke_test_kvpress.py before use.
         If the context manager doesn't work, the alternative is
         kvpress's KVPressTextGenerationPipeline (Q&A oriented API).
      B) Quantized cache — cache_implementation="quantized" in generate()
      C) Vanilla — plain model.generate()
    """
    import torch

    input_ids = input_ids.to(model.device)
    prompt_len = input_ids.shape[1]

    if strategy.press_factory is not None:
        # Path A: kvpress press (strategies 3-12)
        press = strategy.press_factory()
        with press(model):
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        tokens_generated = output_ids.shape[1] - prompt_len
        cache_size = strategy.budget_tokens or (prompt_len + tokens_generated)
    elif strategy.quantized:
        # Path B: INT8 quantized KV cache via transformers generate() API
        # INT8+HQQ avoids logit spikes with Qwen2.5's 2 GQA heads that cause
        # degenerate repetition under INT4. residual_length=128 keeps recent
        # tokens (including attention sinks) in full precision.
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            cache_implementation="quantized",
            cache_config={"nbits": 8, "backend": "hqq", "residual_length": 128},
        )
        tokens_generated = output_ids.shape[1] - prompt_len
        cache_size = prompt_len + tokens_generated
    else:
        # Path C: vanilla generation (full_cache)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        tokens_generated = output_ids.shape[1] - prompt_len
        cache_size = prompt_len + tokens_generated

    generated_text = tokenizer.decode(
        output_ids[0, prompt_len:], skip_special_tokens=True
    )
    return generated_text, tokens_generated, cache_size


# ---------------------------------------------------------------------------
# Top-level accuracy eval
# ---------------------------------------------------------------------------
def run_accuracy_eval(
    model_name: str,
    tasks: list[MathTask],
    strategies: dict[str, StrategyConfig] | None = None,
    decision_interval: int = 64,
    max_new_tokens: int = 2048,
    device: str = "auto",
) -> list[AccuracyResult]:
    """Run accuracy evaluation across strategies and tasks.

    Loads model once, iterates strategies x tasks, scores answers.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if strategies is None:
        strategies = get_strategies(decision_interval)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    attn_kwargs = {}
    if _needs_eager_attention(strategies):
        attn_kwargs["attn_implementation"] = "eager"
        print("NOTE: Using eager attention (required for H2O strategies)")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
        **attn_kwargs,
    )
    model.eval()
    print(f"Model loaded on {model.device}")

    return run_accuracy_eval_with_model(
        model, tokenizer, tasks, strategies, max_new_tokens
    )


def run_accuracy_eval_with_model(
    model,
    tokenizer,
    tasks: list[MathTask],
    strategies: dict[str, StrategyConfig],
    max_new_tokens: int = 2048,
) -> list[AccuracyResult]:
    """Run accuracy eval with an already-loaded model."""
    import torch

    results: list[AccuracyResult] = []
    total = len(strategies) * len(tasks)
    count = 0

    for strat_name, strategy in strategies.items():
        correct_count = 0
        print(f"\n--- Strategy: {strat_name} ({strategy.description}) ---")

        for i, task in enumerate(tasks):
            count += 1
            input_ids = format_math_prompt(tokenizer, task.problem)
            prompt_len = input_ids.shape[1]

            # Track peak memory
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            try:
                generated_text, tokens_generated, cache_size = generate_with_strategy(
                    model, tokenizer, input_ids, strategy, max_new_tokens
                )
            except Exception as e:
                print(f"  [{count}/{total}] {task.task_id}: ERROR — {e}")
                results.append(
                    AccuracyResult(
                        strategy_name=strat_name,
                        task_id=task.task_id,
                        generated_text=f"ERROR: {e}",
                        extracted_answer=None,
                        ground_truth_answer=task.ground_truth_answer,
                        correct=False,
                        tokens_generated=0,
                        prompt_tokens=prompt_len,
                        cache_size_at_end=0,
                        peak_memory_mb=0.0,
                    )
                )
                continue

            extracted = extract_boxed_answer(generated_text)
            correct = is_correct(generated_text, task.ground_truth_answer)
            if correct:
                correct_count += 1

            peak_mb = 0.0
            if torch.cuda.is_available():
                peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            results.append(
                AccuracyResult(
                    strategy_name=strat_name,
                    task_id=task.task_id,
                    generated_text=generated_text,
                    extracted_answer=extracted,
                    ground_truth_answer=task.ground_truth_answer,
                    correct=correct,
                    tokens_generated=tokens_generated,
                    prompt_tokens=prompt_len,
                    cache_size_at_end=cache_size,
                    peak_memory_mb=peak_mb,
                )
            )

            status = "OK" if correct else "WRONG"
            print(
                f"  [{count}/{total}] {task.task_id}: {status} "
                f"(extracted={extracted}, truth={task.ground_truth_answer}, "
                f"tokens={tokens_generated})"
            )

        acc = correct_count / len(tasks) if tasks else 0
        print(f"  >> {strat_name} accuracy: {acc:.1%} ({correct_count}/{len(tasks)})")

    return results


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------
def save_results(results: list[AccuracyResult], path: str) -> None:
    """Save results to JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")
    print(f"Saved {len(results)} results to {path}")


def load_results(path: str) -> list[AccuracyResult]:
    """Load results from JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(AccuracyResult.from_dict(json.loads(line)))
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _setup_matplotlib():
    """Configure matplotlib for headless environments."""
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_accuracy_by_strategy(
    results: list[AccuracyResult], output_path: str
) -> None:
    """Bar chart: strategy name -> mean accuracy."""
    plt = _setup_matplotlib()

    # Group by strategy
    by_strategy: dict[str, list[bool]] = {}
    for r in results:
        by_strategy.setdefault(r.strategy_name, []).append(r.correct)

    names = list(by_strategy.keys())
    accs = [sum(v) / len(v) for v in by_strategy.values()]

    # Color by strategy type
    colors = []
    for name in names:
        if name.startswith("full_cache"):
            colors.append("#2196F3")  # blue
        elif name.startswith("window"):
            colors.append("#FF9800")  # orange
        elif name.startswith("h2o"):
            colors.append("#4CAF50")  # green
        elif name.startswith("snapkv"):
            colors.append("#9C27B0")  # purple
        elif name.startswith("expected"):
            colors.append("#F44336")  # red
        else:
            colors.append("#607D8B")  # gray

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(names)), accs, color=colors, alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("MATH Accuracy by KV Cache Strategy")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.1%}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved accuracy plot to {output_path}")


def plot_accuracy_vs_budget(
    results: list[AccuracyResult], output_path: str
) -> None:
    """Line plot: budget in tokens -> accuracy, one line per strategy type."""
    plt = _setup_matplotlib()

    # Group by strategy family and budget
    families: dict[str, dict[int, list[bool]]] = {}
    budget_map = {
        "window_128": 132,
        "window_256": 260,
        "window_512": 516,
        "window_1024": 1028,
        "h2o_128": 128,
        "h2o_256": 256,
        "h2o_512": 512,
        "h2o_1024": 1024,
    }

    for r in results:
        if r.strategy_name not in budget_map:
            continue
        budget = budget_map[r.strategy_name]
        if r.strategy_name.startswith("window"):
            family = "StreamingLLM"
        else:
            family = "H2O"
        families.setdefault(family, {}).setdefault(budget, []).append(r.correct)

    fig, ax = plt.subplots(figsize=(8, 5))
    family_colors = {"StreamingLLM": "#FF9800", "H2O": "#4CAF50"}

    for family, budgets in families.items():
        xs = sorted(budgets.keys())
        ys = [sum(budgets[x]) / len(budgets[x]) for x in xs]
        ax.plot(
            xs, ys,
            marker="o",
            linewidth=2,
            markersize=6,
            label=family,
            color=family_colors.get(family, "#607D8B"),
        )

    ax.set_xlabel("Cache Budget (tokens)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Cache Budget")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved accuracy vs budget plot to {output_path}")


def plot_pareto(
    accuracy_results: list[AccuracyResult],
    latency_results: list[dict],
    output_path: str,
) -> None:
    """Scatter: accuracy (y) vs simulated latency (x), one series per hardware."""
    plt = _setup_matplotlib()
    import numpy as np

    # Compute per-strategy accuracy
    by_strategy: dict[str, list[bool]] = {}
    for r in accuracy_results:
        by_strategy.setdefault(r.strategy_name, []).append(r.correct)
    strategy_acc = {k: sum(v) / len(v) for k, v in by_strategy.items()}

    # Group latency by hardware
    by_hw: dict[str, list[dict]] = {}
    for lr in latency_results:
        hw_name = lr.get("hardware", "unknown")
        by_hw.setdefault(hw_name, []).append(lr)

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "p"]

    for i, (hw_name, rows) in enumerate(by_hw.items()):
        xs, ys = [], []
        for row in rows:
            strat = row.get("strategy", "")
            if strat in strategy_acc:
                xs.append(row.get("mean_latency_ms", 0))
                ys.append(strategy_acc[strat])
        if xs:
            marker = markers[i % len(markers)]
            ax.scatter(xs, ys, s=40, alpha=0.7, marker=marker, label=hw_name)

    ax.set_xlabel("Simulated Latency (ms/token)")
    ax.set_ylabel("Real Accuracy")
    ax.set_title("Pareto: Accuracy vs Latency across Hardware")
    ax.grid(alpha=0.3)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7,
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Pareto plot to {output_path}")
