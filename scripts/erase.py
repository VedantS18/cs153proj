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


def _make_hook(concept_dir, device):
    """
    Returns a forward hook that projects out concept_dir from the hidden states.
    concept_dir: (hidden_dim,) unit vector on CPU — moved to device at hook time.
    """
    def hook(module, input, output):
        # output is (hidden_states, ...) or just hidden_states depending on layer type
        h = output[0] if isinstance(output, tuple) else output
        v = concept_dir.to(device=h.device, dtype=h.dtype)
        # project out: h' = h - (h @ v) * v
        proj = (h @ v).unsqueeze(-1) * v  # (B, T, H)
        h_erased = h - proj
        if isinstance(output, tuple):
            return (h_erased,) + output[1:]
        return h_erased
    return hook


def apply_erasure(model, probe_weights, concepts=None, erase_layers=None):
    """
    Register nullspace projection hooks for the given concepts.

    erase_layers: list of layer indices to erase at, or None to use each
                  concept's peak layer only (original single-layer behaviour).
                  When set, the same concept direction (from the peak layer probe)
                  is projected out at every specified layer.

    Returns list of hook handles — pass to remove_erasure() when done.
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
        concept_dir = w["coef"]  # unit-normalised direction from peak-layer probe

        target_layers = erase_layers if erase_layers is not None else [w["peak_layer"]]

        for layer_idx in target_layers:
            if layer_idx >= n_layers:
                continue
            hook = model_layers[layer_idx].register_forward_hook(
                _make_hook(concept_dir, device)
            )
            hooks.append(hook)

        print(f"  Erasure hooks registered: {concept} @ layers {target_layers}")

    return hooks


def remove_erasure(hooks):
    for h in hooks:
        h.remove()
