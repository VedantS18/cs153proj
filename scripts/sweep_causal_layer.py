"""
Single-layer causal erasure sweep.

For each layer L independently (not cumulatively), registers a nullspace
projection hook at *only* that layer and measures behavioral change.

This identifies the CAUSAL LOCUS of each concept — the layer where erasing
the concept direction actually changes model outputs — as distinct from the
PROBE LOCUS (the layer where linear probes peak, uniformly layer 1-2).

The gap between probe locus and causal locus is the central finding:
concepts are linearly decodable early but causally expressed late.

Output: results/causal_layer_sweep.json
{
  concept: {
    layer_idx: {
      "concept_test": float,   # behavioral score with only this layer erased
      "mmlu": float,
    }
  }
}

Usage:
    python scripts/sweep_causal_layer.py
    python scripts/sweep_causal_layer.py --concepts capital_cities gender_profession
    python scripts/sweep_causal_layer.py --alpha 15.0   # amplified erasure
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from erase import apply_erasure, load_probe_weights, remove_erasure

# Sample every other layer to reduce compute; interpolate for plotting
LAYERS_TO_TEST = list(range(0, 28, 2)) + [27]

MMLU_SUBSET = [
    ("What is the capital of France?",           ["London", "Berlin", "Paris", "Madrid"],        2),
    ("Which planet is closest to the Sun?",      ["Venus", "Mercury", "Earth", "Mars"],           1),
    ("What is 2 + 2?",                           ["3", "4", "5", "6"],                            1),
    ("Who wrote Romeo and Juliet?",              ["Dickens", "Austen", "Shakespeare", "Chaucer"], 2),
    ("What is the chemical formula for water?",  ["CO2", "H2O", "NaCl", "O2"],                   1),
    ("What is the largest continent?",           ["Africa", "Asia", "Europe", "N. America"],      1),
    ("How many sides does a hexagon have?",      ["5", "6", "7", "8"],                            1),
    ("Which element has atomic number 1?",       ["Helium", "Carbon", "Hydrogen", "Oxygen"],      2),
    ("In what year did World War II end?",       ["1943", "1944", "1945", "1946"],                 2),
    ("What is the square root of 144?",          ["10", "11", "12", "13"],                        2),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    p.add_argument("--weights_path", default="results/probe_weights.json")
    p.add_argument("--test_dir", default="data/concept_test")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--concepts", nargs="*", default=None)
    p.add_argument("--layers", nargs="*", type=int, default=None,
                   help="Layers to test (default: every other layer 0,2,4,...,27)")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Erasure strength (1.0=standard projection, >1=amplified steering)")
    p.add_argument("--max_length", type=int, default=64)
    return p.parse_args()


@torch.no_grad()
def sentence_log_prob(model, tokenizer, sentence, device, max_length):
    enc = tokenizer(sentence, return_tensors="pt",
                    truncation=True, max_length=max_length).to(device)
    out = model(**enc, labels=enc["input_ids"])
    return -out.loss.item() * enc["input_ids"].shape[1]


@torch.no_grad()
def concept_test_score(model, tokenizer, device, tests, max_length):
    if not tests:
        return None
    cat = tests[0]["category"]
    if cat == "factual":
        correct = sum(
            int(np.argmax([sentence_log_prob(model, tokenizer,
                                              item["prompt"] + " " + c, device, max_length)
                           for c in item["choices"]]) == item["answer_idx"])
            for item in tests
        )
        return correct / len(tests)
    elif cat == "bias":
        gaps = [
            sentence_log_prob(model, tokenizer,
                               item["prefix"] + item["stereotypical_suffix"], device, max_length)
            - sentence_log_prob(model, tokenizer,
                                 item["prefix"] + item["counter_suffix"], device, max_length)
            for item in tests
        ]
        return float(np.mean(gaps))
    elif cat == "stylistic":
        correct = sum(
            int(sentence_log_prob(model, tokenizer,
                                   item["question"] + " " + item["style_passage"], device, max_length)
                > sentence_log_prob(model, tokenizer,
                                     item["question"] + " " + item["other_passage"], device, max_length))
            for item in tests
        )
        return correct / len(tests)
    return None


@torch.no_grad()
def mmlu_score(model, tokenizer, device, max_length):
    correct = 0
    for question, choices, answer_idx in MMLU_SUBSET:
        scores = [sentence_log_prob(model, tokenizer,
                                    f"Question: {question}\nAnswer: {c}", device, max_length)
                  for c in choices]
        correct += int(np.argmax(scores) == answer_idx)
    return correct / len(MMLU_SUBSET)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  alpha={args.alpha}")

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
    layers = args.layers or LAYERS_TO_TEST
    n_model_layers = len(model.model.layers)
    layers = [l for l in layers if l < n_model_layers]

    print(f"Concepts: {concepts}")
    print(f"Layers to test: {layers}\n")

    # Load concept tests
    concept_tests = {}
    for concept in concepts:
        path = os.path.join(args.test_dir, f"{concept}.json")
        concept_tests[concept] = json.load(open(path)) if os.path.exists(path) else []

    # Baseline (no erasure)
    print("=== BASELINE ===")
    baselines = {}
    mmlu_baseline = mmlu_score(model, tokenizer, device, args.max_length)
    for concept in concepts:
        ct = concept_test_score(model, tokenizer, device,
                                concept_tests[concept], args.max_length)
        baselines[concept] = {"concept_test": ct, "mmlu": mmlu_baseline}
        print(f"  {concept:<25} ct={ct:.4f if ct is not None else 'n/a'}  mmlu={mmlu_baseline:.3f}")

    results = {"baseline": baselines}

    # Per-layer causal sweep
    for layer_idx in layers:
        print(f"\n=== LAYER {layer_idx:2d} ===")
        layer_results = {}
        mmlu_done = False
        mmlu_val = mmlu_baseline

        for concept in concepts:
            hooks = apply_erasure(
                model, probe_weights,
                concepts=[concept],
                erase_layers=[layer_idx],
                alpha=args.alpha,
            )
            ct = concept_test_score(model, tokenizer, device,
                                    concept_tests[concept], args.max_length)
            if not mmlu_done:
                mmlu_val = mmlu_score(model, tokenizer, device, args.max_length)
                mmlu_done = True
            remove_erasure(hooks)

            ct_base = baselines[concept]["concept_test"]
            delta = (ct - ct_base) if (ct is not None and ct_base is not None) else None
            layer_results[concept] = {"concept_test": ct, "mmlu": mmlu_val}
            print(f"  {concept:<25} ct={ct:.4f}  Δ={delta:+.4f}  mmlu={mmlu_val:.3f}")

        results[str(layer_idx)] = layer_results

    # Save
    suffix = f"_alpha{args.alpha:.0f}" if args.alpha != 1.0 else ""
    out_path = os.path.join(args.out_dir, f"causal_layer_sweep{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # Summary: which layer causes the biggest behavioral change per concept?
    print("\n=== CAUSAL LOCUS (layer with max |behavioral delta|) ===")
    print(f"{'Concept':<25} {'Probe peak':>11} {'Causal layer':>13} {'Max |Δ|':>9} {'MMLU@causal':>12}")
    print("-" * 75)
    for concept in concepts:
        probe_peak = probe_weights[concept]["peak_layer"]
        ct_base = baselines[concept]["concept_test"]
        if ct_base is None:
            continue
        best_layer, best_delta = None, 0.0
        for layer_idx in layers:
            ct = results[str(layer_idx)][concept]["concept_test"]
            if ct is None:
                continue
            delta = abs(ct - ct_base)
            if delta > best_delta:
                best_delta = delta
                best_layer = layer_idx
        mmlu_at_causal = results[str(best_layer)][concept]["mmlu"] if best_layer is not None else None
        print(f"{concept:<25} {probe_peak:>11}  {str(best_layer):>13}  {best_delta:>9.4f}  "
              f"{mmlu_at_causal:>12.3f}" if mmlu_at_causal else f"{concept:<25} {probe_peak:>11}  n/a")


if __name__ == "__main__":
    main()
