"""
Sweep erasure depth: how many layers must we erase before behavior changes?

For each depth setting, registers nullspace projection hooks at layers 0..depth-1
(always starting from layer 0) and measures:
  - Concept-specific behavioral score (MCQ acc / stereotype gap / style acc)
  - MMLU accuracy (collateral damage)

The probe direction used at every layer is the one from each concept's peak layer.

Outputs results/depth_sweep.json:
{
  depth: {
    concept: {
      "concept_test": float,
      "mmlu": float,
    }
  }
}

Usage:
    python scripts/sweep_erasure_depth.py
    python scripts/sweep_erasure_depth.py --concepts capital_cities gender_profession hemingway
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


# Depths to sweep: erase layers 0..depth-1
DEPTHS = [0, 1, 2, 4, 8, 14, 20, 28]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    p.add_argument("--weights_path", default="results/probe_weights.json")
    p.add_argument("--test_dir", default="data/concept_test")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--concepts", nargs="*", default=None)
    p.add_argument("--depths", nargs="*", type=int, default=None,
                   help="Override depth list (default: 0 1 2 4 8 14 20 28)")
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
    category = tests[0]["category"]

    if category == "factual":
        correct = 0
        for item in tests:
            scores = [sentence_log_prob(model, tokenizer,
                                        item["prompt"] + " " + c, device, max_length)
                      for c in item["choices"]]
            correct += int(np.argmax(scores) == item["answer_idx"])
        return correct / len(tests)

    elif category == "bias":
        gaps = []
        for item in tests:
            lp_s = sentence_log_prob(model, tokenizer,
                                      item["prefix"] + item["stereotypical_suffix"],
                                      device, max_length)
            lp_c = sentence_log_prob(model, tokenizer,
                                      item["prefix"] + item["counter_suffix"],
                                      device, max_length)
            gaps.append(lp_s - lp_c)
        return float(np.mean(gaps))

    elif category == "stylistic":
        correct = 0
        for item in tests:
            lp_s = sentence_log_prob(model, tokenizer,
                                      item["question"] + " " + item["style_passage"],
                                      device, max_length)
            lp_o = sentence_log_prob(model, tokenizer,
                                      item["question"] + " " + item["other_passage"],
                                      device, max_length)
            correct += int(lp_s > lp_o)
        return correct / len(tests)

    return None


# Small MMLU subset for collateral damage check
MMLU_SUBSET = [
    ("What is the capital of France?",            ["London", "Berlin", "Paris", "Madrid"],    2),
    ("Which planet is closest to the Sun?",       ["Venus", "Mercury", "Earth", "Mars"],       1),
    ("What is 2 + 2?",                            ["3", "4", "5", "6"],                        1),
    ("Who wrote Romeo and Juliet?",               ["Dickens", "Austen", "Shakespeare", "Chaucer"], 2),
    ("What is the chemical formula for water?",   ["CO2", "H2O", "NaCl", "O2"],               1),
    ("What is the largest continent?",            ["Africa", "Asia", "Europe", "North America"], 1),
    ("How many sides does a hexagon have?",       ["5", "6", "7", "8"],                        1),
    ("Which element has atomic number 1?",        ["Helium", "Carbon", "Hydrogen", "Oxygen"],  2),
    ("In what year did World War II end?",        ["1943", "1944", "1945", "1946"],             2),
    ("What is the square root of 144?",           ["10", "11", "12", "13"],                    2),
]


@torch.no_grad()
def mmlu_score(model, tokenizer, device, max_length=64):
    correct = 0
    for question, choices, answer_idx in MMLU_SUBSET:
        scores = []
        for choice in choices:
            prompt = f"Question: {question}\nAnswer: {choice}"
            enc = tokenizer(prompt, return_tensors="pt",
                            truncation=True, max_length=max_length).to(device)
            out = model(**enc, labels=enc["input_ids"])
            scores.append(-out.loss.item())
        correct += int(np.argmax(scores) == answer_idx)
    return correct / len(MMLU_SUBSET)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model: {args.model}")

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
    depths = args.depths or DEPTHS
    n_model_layers = len(model.model.layers)
    print(f"Model has {n_model_layers} layers")
    print(f"Concepts: {concepts}")
    print(f"Depths:   {depths}\n")

    # Load concept tests once
    concept_tests = {}
    for concept in concepts:
        path = os.path.join(args.test_dir, f"{concept}.json")
        if os.path.exists(path):
            with open(path) as f:
                concept_tests[concept] = json.load(f)
        else:
            print(f"  Warning: no test file for {concept}")
            concept_tests[concept] = []

    results = {}

    for depth in depths:
        erase_layers = list(range(depth)) if depth > 0 else []
        print(f"\n{'='*60}")
        print(f"DEPTH {depth}  (erasing layers {erase_layers if erase_layers else 'none'})")
        print(f"{'='*60}")

        depth_results = {}

        # Baseline depth=0: no hooks, score everything once
        if depth == 0:
            for concept in concepts:
                ct = concept_test_score(model, tokenizer, device,
                                        concept_tests[concept], args.max_length)
                depth_results[concept] = {"concept_test": ct, "mmlu": None}
            # MMLU once at depth 0
            mmlu = mmlu_score(model, tokenizer, device)
            for concept in concepts:
                depth_results[concept]["mmlu"] = mmlu
            print(f"  Baseline MMLU: {mmlu:.3f}")
        else:
            mmlu_done = False
            for concept in concepts:
                hooks = apply_erasure(model, probe_weights,
                                      concepts=[concept],
                                      erase_layers=erase_layers)
                ct = concept_test_score(model, tokenizer, device,
                                        concept_tests[concept], args.max_length)
                if not mmlu_done:
                    mmlu = mmlu_score(model, tokenizer, device)
                    mmlu_done = True
                remove_erasure(hooks)

                depth_results[concept] = {"concept_test": ct, "mmlu": mmlu}
                ct_str = f"{ct:.4f}" if ct is not None else "n/a"
                print(f"  {concept:<25} ct={ct_str}  mmlu={mmlu:.3f}")

        results[str(depth)] = depth_results

    out_path = os.path.join(args.out_dir, "depth_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # Summary table: concept_test at each depth
    print("\n=== CONCEPT TEST SCORE BY DEPTH ===")
    header = f"{'Concept':<25} " + " ".join(f"d={d:>2}" for d in depths)
    print(header)
    print("-" * len(header))
    for concept in concepts:
        row = f"{concept:<25} "
        for depth in depths:
            val = results[str(depth)][concept]["concept_test"]
            row += f"{val:>6.3f} " if val is not None else "   n/a "
        print(row)

    print("\n=== MMLU BY DEPTH ===")
    for depth in depths:
        mmlu = results[str(depth)][concepts[0]]["mmlu"]
        print(f"  depth={depth:>2}: {mmlu:.3f}")


if __name__ == "__main__":
    main()
