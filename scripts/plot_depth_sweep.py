"""
Plot erasure depth sweep results.

Reads results/depth_sweep_*.json and produces:
  figures/depth_sweep_<category>.png  — per-concept behavioral score vs depth
  figures/depth_sweep_mmlu.png        — MMLU vs depth (collateral damage)

Usage:
    python scripts/plot_depth_sweep.py
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = "results/figures"
DEPTHS = [0, 1, 2, 4, 8, 14, 20, 28]

CATEGORY_LABELS = {
    "factual":   "Factual MCQ Accuracy",
    "bias":      "Stereotype Gap (log-prob)",
    "stylistic": "Style Choice Accuracy",
}

CHANCE_LEVELS = {
    "factual":   0.25,   # 4-choice MCQ
    "bias":      0.0,    # gap = 0 means no preference
    "stylistic": 0.5,    # 2-choice
}


def load_sweep_files():
    files = glob.glob("results/depth_sweep_*.json")
    all_data = {}
    for f in files:
        suffix = os.path.basename(f).replace("depth_sweep_", "").replace(".json", "")
        with open(f) as fh:
            all_data[suffix] = json.load(fh)
    return all_data


def infer_category(concept, sweep_data):
    """Guess category from first test file found."""
    test_path = f"data/concept_test/{concept}.json"
    if os.path.exists(test_path):
        with open(test_path) as f:
            tests = json.load(f)
        return tests[0]["category"] if tests else "factual"
    return "factual"


def plot_category(category, concepts, sweep_data, depths):
    fig, ax = plt.subplots(figsize=(8, 5))

    for concept in concepts:
        scores = []
        for d in depths:
            val = sweep_data.get(str(d), {}).get(concept, {}).get("concept_test")
            scores.append(val if val is not None else float("nan"))
        ax.plot(depths, scores, marker="o", label=concept.replace("_", " "))

    chance = CHANCE_LEVELS.get(category, 0)
    ax.axhline(chance, color="gray", linestyle="--", linewidth=1, label="chance")

    ax.set_xlabel("Layers erased (0 to depth−1)")
    ax.set_ylabel(CATEGORY_LABELS.get(category, "Score"))
    ax.set_title(f"Erasure Depth vs. Behavioral Score — {category.capitalize()}")
    ax.set_xticks(depths)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = f"{FIGURES_DIR}/depth_sweep_{category}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_mmlu(all_data, depths):
    fig, ax = plt.subplots(figsize=(7, 4))

    for suffix, sweep_data in all_data.items():
        # MMLU is the same value for all concepts at a given depth
        mmlu_scores = []
        for d in depths:
            depth_data = sweep_data.get(str(d), {})
            vals = [v["mmlu"] for v in depth_data.values() if v.get("mmlu") is not None]
            mmlu_scores.append(vals[0] if vals else float("nan"))
        ax.plot(depths, mmlu_scores, marker="o", label=suffix)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="baseline")
    ax.set_xlabel("Layers erased (0 to depth−1)")
    ax.set_ylabel("MMLU Accuracy")
    ax.set_title("Erasure Depth vs. MMLU (Collateral Damage)")
    ax.set_xticks(depths)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    out = f"{FIGURES_DIR}/depth_sweep_mmlu.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    all_data = load_sweep_files()
    if not all_data:
        print("No depth_sweep_*.json files found in results/")
        return

    for suffix, sweep_data in all_data.items():
        # Get depths actually present
        depths = sorted(int(d) for d in sweep_data.keys())
        concepts = list(next(iter(sweep_data.values())).keys())
        category = infer_category(concepts[0], sweep_data)
        plot_category(category, concepts, sweep_data, depths)

    plot_mmlu(all_data, sorted(int(d) for d in next(iter(all_data.values())).keys()))


if __name__ == "__main__":
    main()
