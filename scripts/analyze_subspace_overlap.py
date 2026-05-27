"""
Analyze concept subspace overlap.

Computes pairwise cosine similarity between concept directions and
reports which concepts share representational space.

Also checks: does erasing concept A partially erase concept B?
(cross-erasure probe accuracy)

Output: results/subspace_overlap.json + figures/subspace_overlap.png

Usage:
    python scripts/analyze_subspace_overlap.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from erase import load_probe_weights


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def probe_score_after_erasure(coef_scaled_target, coef_scaled_eraser, X_scaled, y):
    """
    What accuracy does the target probe achieve after erasing the eraser direction?
    Projects X_scaled along eraser direction, then scores with target probe.
    """
    v = np.array(coef_scaled_eraser, dtype=np.float32)
    v /= np.linalg.norm(v) + 1e-8
    X_erased = X_scaled - (X_scaled @ v)[:, None] * v
    logits = X_erased @ np.array(coef_scaled_target, dtype=np.float32)
    preds = (logits > 0).astype(int)
    return float((preds == y).mean())


def main():
    weights_path = "results/probe_weights.json"
    act_dir = "activations"
    out_dir = "results"
    figures_dir = "results/figures"
    os.makedirs(figures_dir, exist_ok=True)

    probe_weights = load_probe_weights(weights_path)
    concepts = sorted(probe_weights.keys())
    n = len(concepts)

    # --- Pairwise cosine similarity of concept directions ---
    sim_matrix = np.zeros((n, n))
    for i, c1 in enumerate(concepts):
        for j, c2 in enumerate(concepts):
            sim_matrix[i, j] = cosine_sim(
                probe_weights[c1]["coef_scaled"],
                probe_weights[c2]["coef_scaled"],
            )

    # --- Cross-erasure: does erasing A change B's probe accuracy? ---
    cross_erasure = {}
    for eraser in concepts:
        cross_erasure[eraser] = {}
        eraser_dir = probe_weights[eraser]["coef_scaled"]
        eraser_layer = probe_weights[eraser]["peak_layer"]

        for target in concepts:
            target_layer = probe_weights[target]["peak_layer"]
            if target_layer != eraser_layer:
                cross_erasure[eraser][target] = None
                continue

            npz_path = os.path.join(act_dir, f"{target}.npz")
            if not os.path.exists(npz_path):
                cross_erasure[eraser][target] = None
                continue

            npz = np.load(npz_path, allow_pickle=False)
            X = npz["X"][:, target_layer, :].astype(np.float32)
            y = npz["y"].astype(np.int32)
            mean = np.array(probe_weights[target]["scaler_mean"], dtype=np.float32)
            scale = np.array(probe_weights[target]["scaler_scale"], dtype=np.float32)
            X_scaled = (X - mean) / scale

            acc = probe_score_after_erasure(
                probe_weights[target]["coef_scaled"],
                eraser_dir,
                X_scaled, y,
            )
            cross_erasure[eraser][target] = round(acc, 4)

    # --- Save results ---
    results = {
        "cosine_similarity": {
            c1: {c2: round(float(sim_matrix[i, j]), 4)
                 for j, c2 in enumerate(concepts)}
            for i, c1 in enumerate(concepts)
        },
        "cross_erasure_probe_acc": cross_erasure,
    }
    with open(os.path.join(out_dir, "subspace_overlap.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results/subspace_overlap.json")

    # Print most similar pairs
    print("\n=== TOP CONCEPT DIRECTION SIMILARITIES (|cos| > 0.1) ===")
    pairs = []
    for i, c1 in enumerate(concepts):
        for j, c2 in enumerate(concepts):
            if j <= i:
                continue
            s = abs(sim_matrix[i, j])
            if s > 0.1:
                pairs.append((s, c1, c2))
    for s, c1, c2 in sorted(pairs, reverse=True):
        print(f"  {c1:<25} ↔ {c2:<25}  cos={s:.4f}")

    # Print cross-erasure surprises (erasing A drops B's probe acc significantly)
    print("\n=== CROSS-ERASURE EFFECTS (acc drop > 0.05) ===")
    for eraser in concepts:
        for target in concepts:
            if eraser == target:
                continue
            acc = cross_erasure[eraser].get(target)
            if acc is None:
                continue
            baseline = float((np.array(probe_weights[target]["coef_scaled"]) @
                               np.array(probe_weights[target]["coef_scaled"])) > 0)
            # Baseline is always ~1.0; flag if acc dropped noticeably
            if acc < 0.90:
                print(f"  Erasing {eraser:<25} → {target:<25} probe acc = {acc:.3f}")

    # --- Heatmap ---
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.abs(sim_matrix), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(concepts, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(concepts, fontsize=8)
    ax.set_title("|Cosine similarity| between concept probe directions")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = f"{figures_dir}/subspace_overlap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
