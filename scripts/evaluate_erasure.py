"""
Evaluate concept erasure: measure effectiveness and collateral damage.

For each concept:
  1. Probe accuracy before/after erasure (did we suppress the concept?)
  2. MMLU subset accuracy before/after erasure (collateral damage)

Outputs results/erasure_eval.json:
  {
    concept: {
      "probe_acc_before": float,
      "probe_acc_after": float,
      "mmlu_acc_before": float,
      "mmlu_acc_after": float,
    }
  }
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from erase import apply_erasure, load_probe_weights, remove_erasure


# Small MMLU-style questions to measure collateral damage.
# Format: (question, choices, correct_idx)
MMLU_SUBSET = [
    ("What is the capital of France?", ["London", "Berlin", "Paris", "Madrid"], 2),
    ("Which planet is closest to the Sun?", ["Venus", "Mercury", "Earth", "Mars"], 1),
    ("What is 2 + 2?", ["3", "4", "5", "6"], 1),
    ("Who wrote Romeo and Juliet?", ["Dickens", "Austen", "Shakespeare", "Chaucer"], 2),
    ("What is the chemical formula for water?", ["CO2", "H2O", "NaCl", "O2"], 1),
    ("What is the largest continent?", ["Africa", "Asia", "Europe", "North America"], 1),
    ("How many sides does a hexagon have?", ["5", "6", "7", "8"], 1),
    ("What is the speed of light approximately?", ["3×10^8 m/s", "3×10^6 m/s", "3×10^5 m/s", "3×10^4 m/s"], 0),
    ("Which element has atomic number 1?", ["Helium", "Carbon", "Hydrogen", "Oxygen"], 2),
    ("In what year did World War II end?", ["1943", "1944", "1945", "1946"], 2),
    ("What is the square root of 144?", ["10", "11", "12", "13"], 2),
    ("What language is spoken in Brazil?", ["Spanish", "Portuguese", "French", "English"], 1),
    ("Who developed the theory of relativity?", ["Newton", "Bohr", "Einstein", "Planck"], 2),
    ("What is the boiling point of water in Celsius?", ["90", "95", "100", "105"], 2),
    ("How many bones are in the adult human body?", ["196", "206", "216", "226"], 1),
    ("What is the chemical symbol for gold?", ["Go", "Gd", "Au", "Ag"], 2),
    ("Which gas makes up most of Earth's atmosphere?", ["Oxygen", "Carbon dioxide", "Nitrogen", "Argon"], 2),
    ("What is the powerhouse of the cell?", ["Nucleus", "Ribosome", "Mitochondria", "Golgi"], 2),
    ("In which year did the French Revolution begin?", ["1776", "1783", "1789", "1799"], 2),
    ("What is the hardest natural substance?", ["Gold", "Iron", "Diamond", "Quartz"], 2),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    p.add_argument("--act_dir", default="activations")
    p.add_argument("--weights_path", default="results/probe_weights.json")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--concepts", nargs="*", default=None,
                   help="Subset of concepts to evaluate (default: all)")
    p.add_argument("--max_length", type=int, default=64)
    return p.parse_args()


def probe_accuracy(probe_weights, concept, act_dir, erased=False):
    """
    Measure probe accuracy on stored activations, optionally with nullspace projection applied.

    output_hidden_states captures activations before forward hooks fire, so we
    apply the projection directly to the stored numpy arrays instead of re-running
    through the model. This is mathematically equivalent to what the hook does.
    """
    w = probe_weights[concept]
    layer_idx = w["peak_layer"]
    coef = w["coef"].numpy()       # unit concept direction vector
    mean = w["scaler_mean"].numpy()
    scale = w["scaler_scale"].numpy()

    npz = np.load(os.path.join(act_dir, f"{concept}.npz"), allow_pickle=False)
    X = npz["X"][:, layer_idx, :].astype(np.float32)  # (N, hidden_dim)
    y = npz["y"].astype(np.int32)

    if erased:
        # Nullspace projection: remove the concept direction
        # x_erased = x - (x · v) * v  → x_erased · v = 0 exactly
        proj = (X @ coef)[:, None] * coef
        X = X - proj

    X_scaled = (X - mean) / scale
    logits = X_scaled @ coef
    preds = (logits > 0).astype(int)
    return float((preds == y).mean())


@torch.no_grad()
def mmlu_accuracy(model, tokenizer, device, max_length=128):
    """Score MMLU subset with multiple-choice log-prob scoring."""
    correct = 0
    for question, choices, answer_idx in MMLU_SUBSET:
        scores = []
        for choice in choices:
            prompt = f"Question: {question}\nAnswer: {choice}"
            enc = tokenizer(prompt, return_tensors="pt",
                            truncation=True, max_length=max_length).to(device)
            out = model(**enc, labels=enc["input_ids"])
            scores.append(-out.loss.item())  # higher = more likely
        pred = int(np.argmax(scores))
        correct += int(pred == answer_idx)
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
    print(f"Evaluating {len(concepts)} concepts")

    # Baseline MMLU — compute once before any erasure
    print("\nComputing baseline MMLU...")
    mmlu_before = mmlu_accuracy(model, tokenizer, device)
    print(f"Baseline MMLU: {mmlu_before:.3f}")

    results = {}
    for concept in concepts:
        print(f"\n--- {concept} ---")

        # Probe accuracy on stored activations — no model needed
        probe_before = probe_accuracy(probe_weights, concept, args.act_dir, erased=False)
        probe_after  = probe_accuracy(probe_weights, concept, args.act_dir, erased=True)
        print(f"  Probe acc before: {probe_before:.3f}")
        print(f"  Probe acc after:  {probe_after:.3f}")

        # MMLU with erasure hook active on the model
        hooks = apply_erasure(model, probe_weights, concepts=[concept])
        mmlu_after = mmlu_accuracy(model, tokenizer, device)
        remove_erasure(hooks)
        print(f"  MMLU before:      {mmlu_before:.3f}")
        print(f"  MMLU after:       {mmlu_after:.3f}")

        results[concept] = {
            "probe_acc_before": probe_before,
            "probe_acc_after": probe_after,
            "mmlu_acc_before": mmlu_before,
            "mmlu_acc_after": mmlu_after,
            "probe_delta": probe_after - probe_before,
            "mmlu_delta": mmlu_after - mmlu_before,
        }
        print(f"  Erasure effect:   {results[concept]['probe_delta']:+.3f} probe | "
              f"{results[concept]['mmlu_delta']:+.3f} MMLU")

    out_path = os.path.join(args.out_dir, "erasure_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # Summary table
    print("\n=== SUMMARY ===")
    print(f"{'Concept':<25} {'Probe Δ':>10} {'MMLU Δ':>10}")
    print("-" * 47)
    for concept, r in sorted(results.items()):
        print(f"{concept:<25} {r['probe_delta']:>+10.3f} {r['mmlu_delta']:>+10.3f}")


if __name__ == "__main__":
    main()
