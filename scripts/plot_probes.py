"""
Plot probe accuracy curves across layers for all concepts.

Reads results/probe_accuracies.json, produces:
  results/figures/probe_curves.png   — one line per concept, colored by category
  results/figures/peak_layers.png    — bar chart of peak layer per concept
  results/peak_layers.json           — { concept: peak_layer }
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

CATEGORY_COLOR = {
    "factual": "#2196F3",
    "bias": "#F44336",
    "stylistic": "#4CAF50",
}

CONCEPT_CATEGORY = {
    "capital_cities": "factual",
    "element_symbols": "factual",
    "inventor_invention": "factual",
    "country_language": "factual",
    "historical_dates": "factual",
    "gender_profession": "bias",
    "age_competence": "bias",
    "race_crime": "bias",
    "nationality_stereotype": "bias",
    "gender_emotion": "bias",
    "hemingway": "stylistic",
    "shakespeare": "stylistic",
    "legal_text": "stylistic",
    "scientific_writing": "stylistic",
    "news_wire": "stylistic",
}


def main():
    results_path = "results/probe_accuracies.json"
    fig_dir = "results/figures"
    os.makedirs(fig_dir, exist_ok=True)

    with open(results_path) as f:
        data = json.load(f)

    # --- probe curves ---
    fig, ax = plt.subplots(figsize=(12, 6))
    peak_layers = {}

    for concept, layer_accs in sorted(data.items()):
        layers = sorted(layer_accs.keys(), key=int)
        accs = [layer_accs[l] for l in layers]
        layers_int = [int(l) for l in layers]

        category = CONCEPT_CATEGORY.get(concept, "factual")
        color = CATEGORY_COLOR[category]
        ax.plot(layers_int, accs, label=concept, color=color, alpha=0.7, linewidth=1.5)

        peak = int(max(layer_accs, key=lambda k: layer_accs[k]))
        peak_layers[concept] = peak

    # legend by category
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=c, linewidth=2, label=cat)
        for cat, c in CATEGORY_COLOR.items()
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe accuracy (5-fold CV)")
    ax.set_title("Linear probe accuracy by layer — all concepts")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
    ax.set_ylim(0.4, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "probe_curves.png"), dpi=150)
    print(f"Saved probe_curves.png")
    plt.close()

    # --- peak layer bar chart ---
    concepts = sorted(peak_layers.keys())
    peaks = [peak_layers[c] for c in concepts]
    colors = [CATEGORY_COLOR[CONCEPT_CATEGORY.get(c, "factual")] for c in concepts]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(concepts)), peaks, color=colors)
    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels(concepts, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Peak layer")
    ax.set_title("Peak probe accuracy layer per concept")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, label=cat) for cat, c in CATEGORY_COLOR.items()
    ]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "peak_layers.png"), dpi=150)
    print(f"Saved peak_layers.png")
    plt.close()

    with open("results/peak_layers.json", "w") as f:
        json.dump(peak_layers, f, indent=2)
    print("Saved peak_layers.json")
    print("\nPeak layers:")
    for c in concepts:
        cat = CONCEPT_CATEGORY.get(c, "?")
        print(f"  [{cat:10s}] {c}: layer {peak_layers[c]}")


if __name__ == "__main__":
    main()
