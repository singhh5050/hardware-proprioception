#!/usr/bin/env python3
"""Generate accuracy eval visualizations from final merged results."""

import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
records = []
with open("accuracy_results_final.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

by_strat = defaultdict(list)
for r in records:
    by_strat[r["strategy_name"]].append(r)

# ---------------------------------------------------------------------------
# 1. Bar chart: accuracy by strategy (rescored)
# ---------------------------------------------------------------------------
# Order: baselines first, then by budget
order = [
    "full_cache", "full_cache_int8",
    "window_128", "window_256", "window_512", "window_1024",
    "h2o_128", "h2o_256", "h2o_512", "h2o_1024",
    "snapkv_512", "expected_attn_512",
]

accs = [sum(1 for r in by_strat[s] if r["correct"]) / len(by_strat[s]) for s in order]
orig_accs = [sum(1 for r in by_strat[s] if r.get("correct_original")) / len(by_strat[s]) for s in order]
sources = [by_strat[s][0].get("source", "") for s in order]

# Color by strategy family
family_colors = {
    "full_cache": "#2196F3",
    "full_cache_int8": "#2196F3",
    "window": "#FF9800",
    "h2o": "#4CAF50",
    "snapkv": "#9C27B0",
    "expected": "#F44336",
}

def get_color(name):
    for prefix, color in family_colors.items():
        if name.startswith(prefix):
            return color
    return "#607D8B"

colors = [get_color(s) for s in order]

fig, ax = plt.subplots(figsize=(12, 5.5))

# Rescored bars
bars = ax.bar(range(len(order)), accs, color=colors, alpha=0.85, label="Rescored (math-verify)")

# Original accuracy as a dot overlay
ax.scatter(range(len(order)), orig_accs, color="black", s=30, zorder=5,
           marker="_", linewidths=2, label="Original scorer")

# Annotate rerun strategies
for i, src in enumerate(sources):
    if "rerun" in src:
        ax.annotate("rerun", (i, accs[i] + 0.015), ha="center", fontsize=6,
                     color="#333", fontstyle="italic")

ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Accuracy")
ax.set_title("MATH-500 Accuracy by KV Cache Strategy (Qwen2.5-Math-7B-Instruct)")
ax.set_ylim(0, 1.0)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper left", fontsize=8)

# Value labels
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{acc:.1%}", ha="center", va="bottom", fontsize=7)

fig.tight_layout()
fig.savefig("accuracy_by_strategy.png", dpi=200)
plt.close(fig)
print("Saved accuracy_by_strategy.png")

# ---------------------------------------------------------------------------
# 2. Line plot: accuracy vs cache budget
# ---------------------------------------------------------------------------
budget_map = {
    "window_128": 132, "window_256": 260, "window_512": 516, "window_1024": 1028,
    "h2o_128": 128, "h2o_256": 256, "h2o_512": 512, "h2o_1024": 1024,
    "snapkv_512": 512,
}

families = defaultdict(dict)
for name, budget in budget_map.items():
    acc = sum(1 for r in by_strat[name] if r["correct"]) / len(by_strat[name])
    if name.startswith("window"):
        families["StreamingLLM"][budget] = acc
    elif name.startswith("h2o"):
        families["H2O"][budget] = acc
    elif name.startswith("snapkv"):
        families["SnapKV"][budget] = acc

# Add baselines as horizontal lines
baseline_acc = sum(1 for r in by_strat["full_cache"] if r["correct"]) / 200
int8_acc = sum(1 for r in by_strat["full_cache_int8"] if r["correct"]) / 200
ea_acc = sum(1 for r in by_strat["expected_attn_512"] if r["correct"]) / 200

fig, ax = plt.subplots(figsize=(8, 5))

family_styles = {
    "StreamingLLM": {"color": "#FF9800", "marker": "o"},
    "H2O": {"color": "#4CAF50", "marker": "s"},
    "SnapKV": {"color": "#9C27B0", "marker": "^"},
}

for family, budgets in families.items():
    xs = sorted(budgets.keys())
    ys = [budgets[x] for x in xs]
    style = family_styles[family]
    ax.plot(xs, ys, marker=style["marker"], linewidth=2, markersize=7,
            label=family, color=style["color"])

ax.axhline(y=baseline_acc, color="#2196F3", linestyle="--", alpha=0.6, label=f"full_cache ({baseline_acc:.1%})")
ax.axhline(y=int8_acc, color="#2196F3", linestyle=":", alpha=0.6, label=f"full_cache_int8 ({int8_acc:.1%})")

ax.set_xlabel("Cache Budget (tokens)")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs Cache Budget")
ax.set_ylim(0.2, 0.85)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("accuracy_vs_budget.png", dpi=200)
plt.close(fig)
print("Saved accuracy_vs_budget.png")

# ---------------------------------------------------------------------------
# 3. Rescorer impact: old vs new
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

x = range(len(order))
width = 0.35
bars1 = ax.bar([i - width/2 for i in x], orig_accs, width, label="Original scorer", color="#BDBDBD", alpha=0.8)
bars2 = ax.bar([i + width/2 for i in x], accs, width, label="math-verify rescored", color=[get_color(s) for s in order], alpha=0.85)

ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Accuracy")
ax.set_title("Impact of math-verify Rescoring")
ax.set_ylim(0, 1.0)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=8)

# Delta labels
for i, (orig, new) in enumerate(zip(orig_accs, accs)):
    delta = new - orig
    if abs(delta) > 0.001:
        ax.text(i + width/2, new + 0.01, f"+{delta:.1%}" if delta > 0 else f"{delta:.1%}",
                ha="center", fontsize=6, color="#333")

fig.tight_layout()
fig.savefig("rescorer_impact.png", dpi=200)
plt.close(fig)
print("Saved rescorer_impact.png")

print("\nDone! Generated 3 plots.")
