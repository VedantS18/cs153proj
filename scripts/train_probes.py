"""
Train a logistic regression probe at every layer for every concept.

Reads activations/<concept>.npz, runs 5-fold CV per layer,
outputs results/probe_accuracies.json:
  { concept: { layer_idx: mean_accuracy, ... }, ... }
"""

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--act_dir", default="activations")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--max_iter", type=int, default=1000)
    return p.parse_args()


def probe_layer(X_layer, y, n_folds, max_iter):
    """5-fold CV accuracy for a single (n_examples, hidden_dim) layer."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    accs = []
    for train_idx, val_idx in skf.split(X_layer, y):
        X_tr, X_val = X_layer[train_idx], X_layer[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        clf = LogisticRegression(max_iter=max_iter, C=1.0, solver="lbfgs")
        clf.fit(X_tr, y_tr)
        accs.append(clf.score(X_val, y_val))

    return float(np.mean(accs))


def fit_probe(X_layer, y, max_iter):
    """Fit a single probe on all data, return (clf, scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_layer)
    clf = LogisticRegression(max_iter=max_iter, C=1.0, solver="lbfgs")
    clf.fit(X_scaled, y)
    return clf, scaler


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    npz_files = sorted(f for f in os.listdir(args.act_dir) if f.endswith(".npz"))
    print(f"Found {len(npz_files)} concept activation files")

    results = {}
    probe_weights = {}  # { concept: { "layer": int, "coef": [...], "mean": [...], "scale": [...] } }

    for fname in npz_files:
        concept = fname.replace(".npz", "")
        data = np.load(os.path.join(args.act_dir, fname))
        X = data["X"].astype(np.float32)  # (N, n_layers, hidden_dim)
        y = data["y"].astype(np.int32)    # (N,)

        n_layers = X.shape[1]
        print(f"  {concept}: {X.shape[0]} examples, {n_layers} layers", flush=True)

        layer_accs = {}
        for layer_idx in range(n_layers):
            acc = probe_layer(X[:, layer_idx, :], y, args.n_folds, args.max_iter)
            layer_accs[layer_idx] = acc
            print(f"    layer {layer_idx:2d}: {acc:.3f}", flush=True)

        peak_layer = max(layer_accs, key=layer_accs.get)
        print(f"    peak layer: {peak_layer} ({layer_accs[peak_layer]:.3f})")
        results[concept] = layer_accs

        # Fit final probe on all data at peak layer — save weights for nullspace projection
        clf, scaler = fit_probe(X[:, peak_layer, :], y, args.max_iter)
        coef_scaled = clf.coef_[0]
        coef_orig = coef_scaled / (scaler.scale_ + 1e-8)
        coef_orig_normalized = coef_orig / (np.linalg.norm(coef_orig) + 1e-8)
        coef_scaled_normalized = coef_scaled / (np.linalg.norm(coef_scaled) + 1e-8)

        # Top-K PCA subspace of positive class at peak layer (scaled space).
        # These span the concept subspace for rank-K erasure.
        X_peak_scaled = (X[:, peak_layer, :].astype(np.float32) - scaler.mean_) / scaler.scale_
        X_pos = X_peak_scaled[y == 1]
        X_pos_centered = X_pos - X_pos.mean(0)
        _, _, Vt = np.linalg.svd(X_pos_centered, full_matrices=False)
        subspace_dirs = Vt[:8].tolist()  # top-8 PCA directions in scaled space

        # Per-layer probe directions (scaled space) for per-layer erasure.
        per_layer_coefs = {}
        for li in range(X.shape[1]):
            clf_l, scaler_l = fit_probe(X[:, li, :], y, args.max_iter)
            c = clf_l.coef_[0]
            per_layer_coefs[li] = {
                "coef_scaled": (c / (np.linalg.norm(c) + 1e-8)).tolist(),
                "scaler_mean":  scaler_l.mean_.tolist(),
                "scaler_scale": scaler_l.scale_.tolist(),
            }

        probe_weights[concept] = {
            "peak_layer":      int(peak_layer),
            "coef":            coef_orig_normalized.tolist(),
            "coef_scaled":     coef_scaled_normalized.tolist(),
            "scaler_mean":     scaler.mean_.tolist(),
            "scaler_scale":    scaler.scale_.tolist(),
            "subspace_dirs":   subspace_dirs,        # (8, hidden_dim) top PCA dirs, scaled space
            "per_layer_coefs": per_layer_coefs,      # {layer_idx: {coef_scaled, mean, scale}}
        }

    out_path = os.path.join(args.out_dir, "probe_accuracies.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    weights_path = os.path.join(args.out_dir, "probe_weights.json")
    with open(weights_path, "w") as f:
        json.dump(probe_weights, f)
    print(f"Saved {weights_path}")


if __name__ == "__main__":
    main()
