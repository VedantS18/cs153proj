"""
Compute contrast directions from CrowS-Pairs sentences directly.

Instead of using handcrafted internal prompts, extract activations from the
actual CrowS-Pairs sentence pairs and compute:
    v_contrast = mean(sent_more activations) - mean(sent_less activations)

This guarantees the direction is aligned with what CrowS-Pairs measures,
making bidirectional steering (amplify + debias) symmetric and reliable.

Output: results/crows_directions.json
"""

import argparse
import csv
import io
import json
import os
import urllib.request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BIAS_TYPE_TO_CONCEPT = {
    "gender":      ["gender_profession", "gender_emotion"],
    "race-color":  ["race_crime"],
    "age":         ["age_competence"],
    "nationality": ["nationality_stereotype"],
}

CROWS_URL = ("https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/"
             "data/crows_pairs_anonymized.csv")


def load_crows():
    with urllib.request.urlopen(CROWS_URL, timeout=30) as r:
        return list(csv.DictReader(io.StringIO(r.read().decode())))


@torch.no_grad()
def get_mean_activation(model, tokenizer, sentences, layer_idx, device, max_length=64):
    """Mean residual stream activation at layer_idx over a list of sentences."""
    vecs = []
    for sent in sentences:
        enc = tokenizer(sent, return_tensors="pt", truncation=True,
                        max_length=max_length).to(device)
        acts = []

        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            # mean over token positions
            acts.append(h.float().mean(dim=1).squeeze(0).cpu())

        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        model(**enc)
        handle.remove()

        if acts:
            vecs.append(acts[0])

    if not vecs:
        return None
    return torch.stack(vecs).mean(dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    p.add_argument("--out", default="results/crows_directions.json")
    p.add_argument("--max_pairs", type=int, default=300)
    p.add_argument("--peak_layers", default="results/peak_layers.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()
    n_layers = len(model.model.layers)

    # Load peak layers from probe training
    peak_layers = {}
    if os.path.exists(args.peak_layers):
        peak_layers = json.load(open(args.peak_layers))

    rows = load_crows()
    print(f"Loaded {len(rows)} CrowS-Pairs rows")

    results = {}

    for bias_type, concepts in BIAS_TYPE_TO_CONCEPT.items():
        type_rows = [r for r in rows if r["bias_type"] == bias_type][:args.max_pairs]
        if not type_rows:
            print(f"  No rows for {bias_type}, skipping")
            continue

        sent_more = [r["sent_more"] for r in type_rows]
        sent_less = [r["sent_less"] for r in type_rows]
        print(f"\n{bias_type}: {len(type_rows)} pairs")

        for concept in concepts:
            # Use peak layer from probe training, fall back to layer 1
            peak = peak_layers.get(concept, {}).get("peak_layer", 1)
            peak = min(peak, n_layers - 1)
            print(f"  {concept} @ layer {peak} ...", end=" ", flush=True)

            mean_more = get_mean_activation(model, tokenizer, sent_more, peak, device)
            mean_less = get_mean_activation(model, tokenizer, sent_less, peak, device)

            if mean_more is None or mean_less is None:
                print("skipped")
                continue

            v = mean_more - mean_less
            norm = v.norm().item()
            v_unit = (v / (norm + 1e-8)).tolist()

            # Also compute cosine with existing contrast direction if available
            results[concept] = {
                "peak_layer": peak,
                "bias_type": bias_type,
                "n_pairs": len(type_rows),
                "v_contrast": v_unit,
                "v_contrast_norm": norm,
            }
            print(f"done (norm={norm:.3f})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nSaved {args.out}")
    print(f"Concepts: {list(results.keys())}")


if __name__ == "__main__":
    main()
