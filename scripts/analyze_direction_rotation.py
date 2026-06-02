"""
Concept Direction Rotation Analysis.

For each concept, computes the cosine similarity between the probe direction
at layer L and the probe direction at layer 1 (the probe peak), for every L.

This measures how much concept directions ROTATE as information propagates
through the transformer. The rotation rate predicts erasure difficulty:

  - Stable directions (cos ≈ 1.0 at all layers): concept is redundantly
    encoded in the same subspace — erasing at any one layer is compensated
    (predicts factual inviolability)

  - Rotating / flipping directions (cos drifts or goes negative): concept
    is re-encoded differently at each layer — using the peak-layer direction
    for late-layer steering has wrong sign (explains gender_emotion flip)

Output:
  results/direction_rotation.json   — raw cosine similarities per concept/layer
  results/figures/direction_rotation_<category>.png

Usage:
    python scripts/analyze_direction_rotation.py
    python scripts/analyze_direction_rotation.py --weights_path results/probe_weights.json
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


CONCEPT_CATEGORY = {
    "capital_cities": "factual",   "element_symbols": "factual",
    "inventor_invention": "factual", "country_language": "factual",
    "historical_dates": "factual",
    "gender_profession": "bias",   "gender_emotion": "bias",
    "age_competence": "bias",      "race_crime": "bias",
    "nationality_stereotype": "bias",
    "hemingway": "stylistic",      "shakespeare": "stylistic",
    "legal_text": "stylistic",     "scientific_writing": "stylistic",
    "news_wire": "stylistic",
}

CATEGORY_COLOR = {"factual": "#3b82f6", "bias": "#ef4444", "stylistic": "#22c55e"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights_path", default="results/probe_weights.json")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--model_tag", default="llama3b",
                   help="Tag for output filenames (e.g. llama1b, mistral7b)")
    return p.parse_args()


def cosine(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    args = parse_args()
    figures_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    with open(args.weights_path) as f:
        raw = json.load(f)

    concepts = list(raw.keys())
    n_layers = max(len(raw[c].get("per_layer_coefs", {})) for c in concepts)
    print(f"Model: {args.model_tag}  |  Concepts: {len(concepts)}  |  Layers: {n_layers}")

    rotation = {}   # concept → {layer → cosine_sim with layer-1 direction}
    flip_layer = {} # concept → first layer where cosine goes negative

    for concept in concepts:
        w = raw[concept]
        pl = w.get("per_layer_coefs", {})
        if not pl:
            print(f"  {concept}: no per_layer_coefs, skipping")
            continue

        # Reference direction: probe peak layer (always layer 1 in our setup)
        peak = str(w["peak_layer"])
        ref_dir = np.array(pl[peak]["coef_scaled"], dtype=np.float32)

        sims = {}
        first_flip = None
        for layer_key, entry in pl.items():
            layer_idx = int(layer_key)
            dir_L = np.array(entry["coef_scaled"], dtype=np.float32)
            sim = cosine(ref_dir, dir_L)
            sims[layer_idx] = round(sim, 4)
            if sim < 0 and first_flip is None:
                first_flip = layer_idx

        rotation[concept] = sims
        flip_layer[concept] = first_flip

    # Save
    out = {"model_tag": args.model_tag, "rotation": rotation, "flip_layer": flip_layer}
    out_path = os.path.join(args.out_dir, f"direction_rotation_{args.model_tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")

    # Print summary
    print(f"\n{'Concept':<25} {'Category':<12} {'Min cos':>8} {'Flip layer':>11}")
    print("-" * 60)
    for concept in sorted(concepts, key=lambda c: CONCEPT_CATEGORY.get(c, "z")):
        if concept not in rotation:
            continue
        sims = rotation[concept]
        min_cos = min(sims.values())
        flip = flip_layer.get(concept, "none")
        cat = CONCEPT_CATEGORY.get(concept, "?")
        print(f"{concept:<25} {cat:<12} {min_cos:>8.4f} {str(flip):>11}")

    # --- Plots ---
    for category in ["factual", "bias", "stylistic"]:
        cat_concepts = [c for c in concepts if CONCEPT_CATEGORY.get(c) == category
                        and c in rotation]
        if not cat_concepts:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        layers = sorted(rotation[cat_concepts[0]].keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(cat_concepts)))

        for concept, color in zip(cat_concepts, colors):
            sims = [rotation[concept].get(l, float("nan")) for l in layers]
            ax.plot(layers, sims, marker="o", markersize=3, linewidth=2,
                    label=concept.replace("_", " "), color=color)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(1, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Cosine similarity with layer-1 direction", fontsize=11)
        ax.set_title(
            f"Concept Direction Rotation — {category.capitalize()} Concepts\n"
            f"({args.model_tag})",
            fontsize=13
        )
        ax.set_ylim(-1.1, 1.1)
        ax.legend(fontsize=8, ncol=2, loc="lower left")
        ax.grid(True, alpha=0.3)

        fname = f"{figures_dir}/direction_rotation_{category}_{args.model_tag}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")

    # Combined summary plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, category in zip(axes, ["factual", "bias", "stylistic"]):
        cat_concepts = [c for c in concepts if CONCEPT_CATEGORY.get(c) == category
                        and c in rotation]
        layers = sorted(rotation[cat_concepts[0]].keys()) if cat_concepts else []
        color = CATEGORY_COLOR[category]
        all_sims = []
        for concept in cat_concepts:
            sims = [rotation[concept].get(l, float("nan")) for l in layers]
            ax.plot(layers, sims, linewidth=1.5, alpha=0.7,
                    label=concept.replace("_", " "), color=color)
            all_sims.append(sims)

        if all_sims:
            mean_sims = np.nanmean(all_sims, axis=0)
            ax.plot(layers, mean_sims, linewidth=3, color=color,
                    label="mean", linestyle="--")

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_xlabel("Layer")
        ax.set_title(f"{category.capitalize()}", fontsize=13,
                     color=color, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)

    axes[0].set_ylabel("cos(direction_L, direction_1)", fontsize=11)
    fig.suptitle(f"Concept Direction Rotation Across Layers — {args.model_tag}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fname = f"{figures_dir}/direction_rotation_summary_{args.model_tag}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


if __name__ == "__main__":
    main()
