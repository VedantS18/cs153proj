"""
Steering strength (alpha) sweep at the causal locus.

Tests alpha values [1, 2, 5, 10, 15, 20] at both the probe peak layer (layer 1)
and at late layers (20, 24, 27) where causal effects are expected to be larger.

alpha=1.0  → standard nullspace projection (our baseline)
alpha>1.0  → overproject: h' = h - alpha*(h·v)v  (steer away from concept)

This answers: at what steering strength does behavioral erasure occur?
And is the causal-layer erasure more efficient (lower alpha needed) than
probe-peak-layer erasure?

Output: results/alpha_sweep.json
{
  concept: {
    layer_idx: {
      alpha: { "concept_test": float, "mmlu": float }
    }
  }
}

Usage:
    python scripts/sweep_alpha.py
    python scripts/sweep_alpha.py --concepts capital_cities gender_profession hemingway
    python scripts/sweep_alpha.py --test_layers 1 20 27 --alphas 1 5 10 20
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

# Probe peak layer (1) + candidate causal layers
DEFAULT_TEST_LAYERS = [1, 20, 24, 27]
DEFAULT_ALPHAS = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]

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
    p.add_argument("--test_layers", nargs="*", type=int, default=None)
    p.add_argument("--alphas", nargs="*", type=float, default=None)
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
    test_layers = args.test_layers or DEFAULT_TEST_LAYERS
    alphas = args.alphas or DEFAULT_ALPHAS

    print(f"Concepts: {concepts}")
    print(f"Test layers: {test_layers}")
    print(f"Alphas: {alphas}\n")

    concept_tests = {}
    for concept in concepts:
        path = os.path.join(args.test_dir, f"{concept}.json")
        concept_tests[concept] = json.load(open(path)) if os.path.exists(path) else []

    # Baseline
    print("=== BASELINE ===")
    baselines = {}
    mmlu_base = mmlu_score(model, tokenizer, device, args.max_length)
    for concept in concepts:
        ct = concept_test_score(model, tokenizer, device,
                                concept_tests[concept], args.max_length)
        baselines[concept] = {"concept_test": ct, "mmlu": mmlu_base}
        print(f"  {concept:<25} ct={ct:.4f}  mmlu={mmlu_base:.3f}")

    results = {"baseline": baselines}

    for layer_idx in test_layers:
        layer_key = str(layer_idx)
        results[layer_key] = {}

        for alpha in alphas:
            alpha_key = str(alpha)
            print(f"\n=== Layer {layer_idx}  alpha={alpha} ===")
            alpha_results = {}
            mmlu_done = False
            mmlu_val = mmlu_base

            for concept in concepts:
                hooks = apply_erasure(
                    model, probe_weights,
                    concepts=[concept],
                    erase_layers=[layer_idx],
                    alpha=alpha,
                )
                ct = concept_test_score(model, tokenizer, device,
                                        concept_tests[concept], args.max_length)
                if not mmlu_done:
                    mmlu_val = mmlu_score(model, tokenizer, device, args.max_length)
                    mmlu_done = True
                remove_erasure(hooks)

                ct_base = baselines[concept]["concept_test"]
                delta = (ct - ct_base) if ct is not None and ct_base is not None else None
                alpha_results[concept] = {"concept_test": ct, "mmlu": mmlu_val}
                print(f"  {concept:<25} ct={ct:.4f}  Δ={delta:+.4f}  mmlu={mmlu_val:.3f}")

            results[layer_key][alpha_key] = alpha_results

    out_path = os.path.join(args.out_dir, "alpha_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # Summary: for each concept, find the alpha at layer 27 that first changes behavior
    print("\n=== ERASURE THRESHOLD (first alpha where |Δ| > 0.05 at each layer) ===")
    print(f"{'Concept':<25}" + "".join(f"  L={l:>2}" for l in test_layers))
    print("-" * (25 + 8 * len(test_layers)))
    for concept in concepts:
        ct_base = baselines[concept]["concept_test"]
        if ct_base is None:
            continue
        row = f"{concept:<25}"
        for layer_idx in test_layers:
            layer_key = str(layer_idx)
            threshold_alpha = None
            for alpha in alphas:
                ct = results[layer_key][str(alpha)][concept]["concept_test"]
                if ct is not None and abs(ct - ct_base) > 0.05:
                    threshold_alpha = alpha
                    break
            row += f"  {'α='+str(threshold_alpha) if threshold_alpha else 'none':>6}"
        print(row)


if __name__ == "__main__":
    main()
