"""
Nullspace projection concept erasure.

Loads probe_weights.json and registers a forward hook on the model's
residual stream at each concept's peak layer. The hook projects out the
concept direction: h' = h - (h·v) * v  where v is the unit concept vector.

Usage:
    from scripts.erase import load_eraser, apply_erasure, remove_erasure

    model, hooks = apply_erasure(model, probe_weights, concepts=["capital_cities"])
    # ... run inference ...
    remove_erasure(hooks)
"""

import json
import numpy as np
import torch


def load_probe_weights(path="results/probe_weights.json"):
    with open(path) as f:
        raw = json.load(f)
    weights = {}
    for concept, data in raw.items():
        weights[concept] = {
            "peak_layer": data["peak_layer"],
            "coef": torch.tensor(data["coef"], dtype=torch.float32),
            "coef_scaled": data.get("coef_scaled"),   # list; used for probe eval in scaled space
            "scaler_mean": torch.tensor(data["scaler_mean"], dtype=torch.float32),
            "scaler_scale": torch.tensor(data["scaler_scale"], dtype=torch.float32),
        }
    return weights


def _make_hook_subspace(directions, device):
    """
    Hook that projects out a K-dimensional subspace.
    directions: (K, hidden_dim) tensor of orthonormal vectors.
    """
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        V = directions.to(device=h.device, dtype=h.dtype)  # (K, H)
        # h' = h - V^T (V h^T)  projected out for each token
        proj = (h @ V.T) @ V   # (B, T, H)
        h_erased = h - proj
        if isinstance(output, tuple):
            return (h_erased,) + output[1:]
        return h_erased
    return hook


def _make_hook_rank1(concept_dir, device):
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        v = concept_dir.to(device=h.device, dtype=h.dtype)
        proj = (h @ v).unsqueeze(-1) * v
        h_erased = h - proj
        if isinstance(output, tuple):
            return (h_erased,) + output[1:]
        return h_erased
    return hook


def apply_erasure(model, probe_weights, concepts=None, erase_layers=None,
                  rank=1, per_layer_dirs=False, mlp_only=False):
    """
    Register nullspace projection hooks for the given concepts.

    erase_layers:    list of layer indices to erase at, or None → peak layer only.
    rank:            number of directions to project out (1 = standard, K = subspace).
                     Uses top-K PCA directions of positive class (subspace_dirs).
    per_layer_dirs:  if True, use each layer's own probe direction instead of peak-layer dir.
                     Requires probe_weights to have 'per_layer_coefs'.
    mlp_only:        if True, hook the MLP sublayer output instead of full residual stream.

    Returns list of hook handles.
    """
    if concepts is None:
        concepts = list(probe_weights.keys())

    model_layers = model.model.layers
    n_layers = len(model_layers)
    device = next(model.parameters()).device

    hooks = []
    for concept in concepts:
        if concept not in probe_weights:
            print(f"Warning: no probe weights for {concept}, skipping")
            continue
        w = probe_weights[concept]
        target_layers = erase_layers if erase_layers is not None else [w["peak_layer"]]

        for layer_idx in target_layers:
            if layer_idx >= n_layers:
                continue

            # Choose which module to hook
            layer = model_layers[layer_idx]
            target_module = layer.mlp if mlp_only else layer

            # Build the projection directions
            if rank > 1 and "subspace_dirs" in w:
                dirs = torch.tensor(w["subspace_dirs"][:rank], dtype=torch.float32)
                # Gram-Schmidt to ensure orthonormality
                dirs = torch.linalg.qr(dirs.T)[0].T
                hook_fn = _make_hook_subspace(dirs, device)
            elif per_layer_dirs and "per_layer_coefs" in w:
                pl = w["per_layer_coefs"].get(layer_idx) or w["per_layer_coefs"].get(str(layer_idx))
                if pl is None:
                    continue
                v = torch.tensor(pl["coef_scaled"], dtype=torch.float32)
                hook_fn = _make_hook_rank1(v, device)
            else:
                hook_fn = _make_hook_rank1(w["coef"], device)

            hook = target_module.register_forward_hook(hook_fn)
            hooks.append(hook)

        mode_str = f"rank={rank}" + (" per-layer" if per_layer_dirs else "") + (" mlp-only" if mlp_only else "")
        print(f"  [{mode_str}] {concept} @ layers {target_layers}")

    return hooks


def remove_erasure(hooks):
    for h in hooks:
        h.remove()
