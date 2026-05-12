"""
Qualitative inspection of erasure — shows what actually changes in model behavior.

For each concept, runs two types of checks:
  1. Probe scores on individual example sentences (before/after erasure)
  2. Model completions to concept-specific prompts (before/after erasure)

Usage:
    python scripts/inspect_erasure.py --concept capital_cities
    python scripts/inspect_erasure.py --concept gender_profession
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


# Concept-specific generation prompts — what we ask the model before/after erasure
PROMPTS = {
    "capital_cities": [
        "The capital city of France is",
        "Tokyo is the capital of",
        "Q: What is the capital of Germany? A:",
        "Berlin is to Germany as Paris is to",
    ],
    "element_symbols": [
        "The chemical symbol for gold is",
        "Au is the symbol for the element",
        "Q: What is the chemical symbol for iron? A:",
    ],
    "inventor_invention": [
        "The telephone was invented by",
        "Alexander Graham Bell is famous for inventing",
        "Q: Who invented the lightbulb? A:",
    ],
    "country_language": [
        "In Brazil, people speak",
        "The official language of Japan is",
        "Q: What language do they speak in Germany? A:",
    ],
    "historical_dates": [
        "World War II ended in",
        "The Berlin Wall fell in",
        "Q: In what year did the French Revolution begin? A:",
    ],
    "gender_profession": [
        "The nurse said that",
        "The engineer solved the problem using",
        "The CEO announced that",
        "The secretary organized",
    ],
    "gender_emotion": [
        "When she heard the news, she",
        "When he heard the news, he",
        "The woman reacted to the criticism by",
        "The man reacted to the criticism by",
    ],
    "age_competence": [
        "The young intern quickly figured out",
        "The elderly employee struggled to",
        "The recent graduate picked up",
    ],
    "race_crime": [
        "The neighborhood became safer after",
        "People felt uneasy when",
    ],
    "nationality_stereotype": [
        "The German engineer was known for being",
        "The French chef prepared",
        "The American tourist",
    ],
    "hemingway": [
        "He drank the wine.",
        "The old man looked at the sea.",
        "Continue this story in the same style: He walked into the bar.",
    ],
    "shakespeare": [
        "Complete this in Shakespearean style: Thou art more lovely",
        "What light through yonder window",
        "To be, or not to be,",
    ],
    "legal_text": [
        "The party of the first part hereby agrees to",
        "Notwithstanding the foregoing, the licensee shall",
        "In the event of a breach of contract,",
    ],
    "scientific_writing": [
        "The experimental results demonstrate",
        "A double-blind randomized controlled trial was conducted to",
        "The null hypothesis was rejected at",
    ],
    "news_wire": [
        "WASHINGTON — The White House announced",
        "NEW YORK (Reuters) — Markets fell",
        "LONDON — The prime minister",
    ],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    p.add_argument("--concept", required=True)
    p.add_argument("--weights_path", default="results/probe_weights.json")
    p.add_argument("--act_dir", default="activations")
    p.add_argument("--max_new_tokens", type=int, default=30)
    p.add_argument("--n_examples", type=int, default=6,
                   help="Number of individual sentences to show probe scores for")
    return p.parse_args()


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, device):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def probe_scores_on_examples(probe_weights, concept, act_dir, n=6):
    """Show probe logit scores on individual sentences, before and after erasure."""
    w = probe_weights[concept]
    layer_idx = w["peak_layer"]
    coef = w["coef"].numpy()
    mean = w["scaler_mean"].numpy()
    scale = w["scaler_scale"].numpy()

    npz = np.load(os.path.join(act_dir, f"{concept}.npz"), allow_pickle=False)
    X = npz["X"][:, layer_idx, :].astype(np.float32)
    y = npz["y"].astype(np.int32)

    texts = []
    with open(f"data/concepts/{concept}.jsonl") as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    # Show n/2 positives and n/2 negatives
    pos_idx = np.where(y == 1)[0][:n // 2]
    neg_idx = np.where(y == 0)[0][:n // 2]
    indices = np.concatenate([pos_idx, neg_idx])

    print(f"\n{'─'*70}")
    print(f"PROBE SCORES ON INDIVIDUAL EXAMPLES  (layer {layer_idx})")
    print(f"{'─'*70}")
    print(f"  Score > 0 → probe predicts POSITIVE (concept present)")
    print(f"  Score < 0 → probe predicts NEGATIVE (concept absent)")
    print(f"  After erasure the score should be ~0 for everything\n")

    coef_s = np.array(probe_weights[concept]["coef_scaled"], dtype=np.float32)

    for i in indices:
        x = X[i]
        label = "POS" if y[i] == 1 else "NEG"

        x_scaled = (x - mean) / scale

        # Before erasure
        score_before = float(x_scaled @ coef_s)

        # After erasure — project in scaled space
        x_scaled_erased = x_scaled - (x_scaled @ coef_s) * coef_s
        score_after = float(x_scaled_erased @ coef_s)

        text_short = texts[i][:65] + "..." if len(texts[i]) > 65 else texts[i]
        print(f"  [{label}] \"{text_short}\"")
        print(f"        before: {score_before:+.4f}  →  after: {score_after:+.4f}")
        print()


def main():
    args = parse_args()

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

    if args.concept not in probe_weights:
        print(f"No probe weights for {args.concept}. Available: {list(probe_weights.keys())}")
        return

    # --- Part 1: probe scores on stored examples ---
    probe_scores_on_examples(probe_weights, args.concept, args.act_dir, args.n_examples)

    # --- Part 2: model completions before/after erasure ---
    prompts = PROMPTS.get(args.concept, [f"Tell me about {args.concept}:"])

    print(f"\n{'─'*70}")
    print(f"MODEL COMPLETIONS  (max {args.max_new_tokens} new tokens)")
    print(f"{'─'*70}\n")

    for prompt in prompts:
        print(f"  PROMPT: \"{prompt}\"")

        # Before
        completion_before = generate(model, tokenizer, prompt, args.max_new_tokens, device)
        print(f"  BEFORE: {completion_before}")

        # After
        hooks = apply_erasure(model, probe_weights, concepts=[args.concept])
        completion_after = generate(model, tokenizer, prompt, args.max_new_tokens, device)
        remove_erasure(hooks)
        print(f"  AFTER:  {completion_after}")
        print()


if __name__ == "__main__":
    main()
