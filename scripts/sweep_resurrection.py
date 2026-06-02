"""
Concept resurrection curve.

Applies erasure at the probe peak layer (layer 1), then measures probe
accuracy at every subsequent layer by hooking the residual stream output.

This answers: after the concept is erased at layer 1, at which layer does
the model reconstruct it? The "resurrection layer" reveals the mechanism
behind the behavioral null result — the concept comes back before output.

Compares:
  baseline_probe[L]  — probe accuracy at layer L without any erasure
  erased_probe[L]    — probe accuracy at layer L after erasing at layer 1

Output: results/resurrection_curve.json
{
  concept: {
    "probe_peak_layer": int,
    "baseline": { layer_idx: accuracy },
    "post_erasure": { layer_idx: accuracy },   # after erasing at probe peak
  }
}

Usage:
    python scripts/sweep_resurrection.py
    python scripts/sweep_resurrection.py --concepts capital_cities gender_profession
    python scripts/sweep_resurrection.py --n_examples 100
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from erase import load_probe_weights, remove_erasure


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    p.add_argument("--weights_path", default="results/probe_weights.json")
    p.add_argument("--data_dir", default="data/concepts")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--concepts", nargs="*", default=None)
    p.add_argument("--n_examples", type=int, default=80,
                   help="Examples per concept (40 pos + 40 neg)")
    p.add_argument("--max_length", type=int, default=64)
    return p.parse_args()


def score_probe_on_acts(acts, coef_scaled, mean, scale):
    """Binary probe accuracy: (n, hidden_dim) → scalar."""
    X = acts.astype(np.float32)
    X_scaled = (X - mean) / scale
    logits = X_scaled @ coef_scaled
    return logits  # return raw logits for threshold at 0


def get_per_layer_coef(probe_weights, concept, layer_idx):
    """Get coef_scaled, mean, scale for a specific layer from extended weights."""
    w = probe_weights[concept]
    pl = w.get("per_layer_coefs", {})
    entry = pl.get(layer_idx) or pl.get(str(layer_idx))
    if entry is None:
        # Fall back to peak-layer scaler with peak-layer coef
        return (
            np.array(w["coef_scaled"], dtype=np.float32),
            w["scaler_mean"].numpy(),
            w["scaler_scale"].numpy(),
        )
    return (
        np.array(entry["coef_scaled"], dtype=np.float32),
        np.array(entry["scaler_mean"], dtype=np.float32),
        np.array(entry["scaler_scale"], dtype=np.float32),
    )


@torch.no_grad()
def collect_all_layer_acts(model, tokenizer, texts, device, max_length):
    """
    Run forward passes on texts, collecting last-token hidden states at every layer.
    Returns: (n_texts, n_layers, hidden_dim) numpy array.
    """
    n_layers = len(model.model.layers)
    all_acts = None

    for text in texts:
        enc = tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=max_length).to(device)

        captured = {}

        def make_capture_hook(layer_idx):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = h[0, -1, :].float().cpu().numpy()
            return hook

        hooks = [
            model.model.layers[i].register_forward_hook(make_capture_hook(i))
            for i in range(n_layers)
        ]
        model(**enc)
        for h in hooks:
            h.remove()

        row = np.stack([captured[i] for i in range(n_layers)])  # (n_layers, H)
        if all_acts is None:
            all_acts = row[None]
        else:
            all_acts = np.concatenate([all_acts, row[None]], axis=0)

    return all_acts  # (n_texts, n_layers, H)


@torch.no_grad()
def collect_all_layer_acts_erased(model, tokenizer, texts, device, max_length,
                                   probe_weights, concept):
    """
    Same as collect_all_layer_acts but with erasure hook at the probe peak layer.
    Captures post-erasure activations at every subsequent layer.
    """
    n_layers = len(model.model.layers)
    erase_layer = probe_weights[concept]["peak_layer"]
    concept_dir = probe_weights[concept]["coef"].to(device)
    all_acts = None

    for text in texts:
        enc = tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=max_length).to(device)

        captured = {}

        def make_capture_hook(layer_idx):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = h[0, -1, :].float().cpu().numpy()
            return hook

        # Erasure hook at peak layer
        def erase_hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            v = concept_dir.to(dtype=h.dtype)
            h_erased = h - (h @ v).unsqueeze(-1) * v
            if isinstance(output, tuple):
                return (h_erased,) + output[1:]
            return h_erased

        hooks = [
            model.model.layers[i].register_forward_hook(make_capture_hook(i))
            for i in range(n_layers)
        ]
        erase_handle = model.model.layers[erase_layer].register_forward_hook(erase_hook)

        model(**enc)

        for h in hooks:
            h.remove()
        erase_handle.remove()

        row = np.stack([captured[i] for i in range(n_layers)])
        if all_acts is None:
            all_acts = row[None]
        else:
            all_acts = np.concatenate([all_acts, row[None]], axis=0)

    return all_acts


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()

    probe_weights = load_probe_weights(args.weights_path)
    concepts = args.concepts or list(probe_weights.keys())
    n_layers = len(model.model.layers)
    n_each = args.n_examples // 2

    results = {}

    for concept in concepts:
        print(f"\n{'='*60}\n{concept}\n{'='*60}")

        data_path = os.path.join(args.data_dir, f"{concept}.jsonl")
        if not os.path.exists(data_path):
            print(f"  No data file, skipping")
            continue

        import json as _json
        lines = [_json.loads(l) for l in open(data_path)]
        pos = [l["text"] for l in lines if l["label"] == 1][:n_each]
        neg = [l["text"] for l in lines if l["label"] == 0][:n_each]
        texts = pos + neg
        labels = np.array([1] * len(pos) + [0] * len(neg))
        erase_layer = probe_weights[concept]["peak_layer"]

        print(f"  {len(texts)} examples, erasing at layer {erase_layer}")

        # Baseline: no erasure
        print("  Collecting baseline activations...")
        acts_baseline = collect_all_layer_acts(
            model, tokenizer, texts, device, args.max_length)

        # Post-erasure activations
        print("  Collecting post-erasure activations...")
        acts_erased = collect_all_layer_acts_erased(
            model, tokenizer, texts, device, args.max_length,
            probe_weights, concept)

        baseline_accs = {}
        erased_accs = {}

        for layer_idx in range(n_layers):
            coef, mean, scale = get_per_layer_coef(probe_weights, concept, layer_idx)

            logits_base = score_probe_on_acts(acts_baseline[:, layer_idx, :], coef, mean, scale)
            acc_base = float((( logits_base > 0).astype(int) == labels).mean())
            baseline_accs[layer_idx] = round(acc_base, 4)

            logits_erased = score_probe_on_acts(acts_erased[:, layer_idx, :], coef, mean, scale)
            acc_erased = float(((logits_erased > 0).astype(int) == labels).mean())
            erased_accs[layer_idx] = round(acc_erased, 4)

        # Find resurrection layer: first layer post-erasure where probe recovers > 75%
        resurrection_layer = None
        for layer_idx in range(erase_layer + 1, n_layers):
            if erased_accs[layer_idx] > 0.75:
                resurrection_layer = layer_idx
                break

        results[concept] = {
            "probe_peak_layer": erase_layer,
            "resurrection_layer": resurrection_layer,
            "baseline": baseline_accs,
            "post_erasure": erased_accs,
        }

        print(f"  Probe peak: layer {erase_layer}  |  "
              f"Resurrection: layer {resurrection_layer}")
        print(f"  Post-erasure probe at layer {erase_layer}: "
              f"{erased_accs[erase_layer]:.3f} "
              f"(baseline: {baseline_accs[erase_layer]:.3f})")
        if resurrection_layer:
            print(f"  Concept recovered at layer {resurrection_layer}: "
                  f"{erased_accs[resurrection_layer]:.3f}")

    out_path = os.path.join(args.out_dir, "resurrection_curve.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # Summary table
    print("\n=== RESURRECTION SUMMARY ===")
    print(f"{'Concept':<25} {'Erase layer':>12} {'Resurrection':>13} {'Layers to recover':>18}")
    print("-" * 72)
    for concept, r in results.items():
        gap = (r["resurrection_layer"] - r["probe_peak_layer"]
               if r["resurrection_layer"] else "never")
        print(f"{concept:<25} {r['probe_peak_layer']:>12}  "
              f"{str(r['resurrection_layer']):>13}  {str(gap):>18}")


if __name__ == "__main__":
    main()
