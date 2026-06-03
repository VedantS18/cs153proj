"""
Compute mean-difference contrast directions for RepE-style steering.

For each concept, computes:
    v_contrast = mean(positive_activations) - mean(negative_activations)

at the peak probe layer. This is the "concept direction" used for constant
steering (RepE), which is different from the probe direction (discriminative)
and produces actual behavioral change by persistently shifting representations.

Saves to results/contrast_directions.json:
{
  concept: {
    "peak_layer": int,
    "v_contrast": [float, ...],    # unit vector, original activation space
    "v_contrast_scaled": [float, ...],  # unit vector, scaled space
    "mean_pos": [float, ...],
    "mean_neg": [float, ...],
  }
}
"""

import argparse
import json
import os
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--act_dir", default="activations")
    p.add_argument("--weights_path", default="results/probe_weights.json")
    p.add_argument("--out_dir", default="results")
    return p.parse_args()


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.weights_path) as f:
        probe_weights = json.load(f)

    contrast_dirs = {}

    print(f"{'Concept':<25} {'Peak L':>7} {'|v_contrast|':>14} {'cos(probe,contrast)':>20}")
    print("-" * 70)

    for concept, w in probe_weights.items():
        npz_path = os.path.join(args.act_dir, f"{concept}.npz")
        if not os.path.exists(npz_path):
            print(f"  {concept}: no activations, skipping")
            continue

        npz = np.load(npz_path, allow_pickle=False)
        X = npz["X"][:, w["peak_layer"], :].astype(np.float32)
        y = npz["y"].astype(np.int32)

        # Mean difference in original activation space
        mean_pos = X[y == 1].mean(axis=0)
        mean_neg = X[y == 0].mean(axis=0)
        v_contrast = mean_pos - mean_neg
        v_contrast_unit = unit(v_contrast)

        # Also compute in scaled space for consistency with probe
        mean_scale = np.array(w["scaler_mean"], dtype=np.float32)
        scale = np.array(w["scaler_scale"], dtype=np.float32)
        X_scaled = (X - mean_scale) / scale
        mean_pos_s = X_scaled[y == 1].mean(axis=0)
        mean_neg_s = X_scaled[y == 0].mean(axis=0)
        v_contrast_scaled = unit(mean_pos_s - mean_neg_s)

        # Cosine similarity between probe direction and contrast direction
        probe_dir = np.array(w.get("coef_scaled", w["coef"]), dtype=np.float32)
        cos_sim = float(np.dot(unit(probe_dir), v_contrast_scaled))

        contrast_dirs[concept] = {
            "peak_layer": w["peak_layer"],
            "v_contrast": v_contrast_unit.tolist(),
            "v_contrast_scaled": v_contrast_scaled.tolist(),
            "mean_pos": mean_pos.tolist(),
            "mean_neg": mean_neg.tolist(),
            "cos_probe_contrast": round(cos_sim, 4),
        }

        print(f"  {concept:<25} {w['peak_layer']:>7}  "
              f"{np.linalg.norm(v_contrast):>12.2f}  {cos_sim:>18.4f}")

    out_path = os.path.join(args.out_dir, "contrast_directions.json")
    with open(out_path, "w") as f:
        json.dump(contrast_dirs, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\nKey insight: cos(probe, contrast) measures alignment.")
    print("If cos ≈ 1.0: probe and contrast point same direction → similar behavior")
    print("If cos ≈ 0.5: partial overlap → contrast direction captures more signal")
    print("If cos ≈ 0.0: orthogonal → very different signals")


if __name__ == "__main__":
    main()
