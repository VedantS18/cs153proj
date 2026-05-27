"""
Generate model completions across all erasure modes for qualitative inspection.

For each concept, shows 2-3 prompt completions under every mode side-by-side.
Makes it easy to see whether (and when) erasure actually changes what the model says.

Output:
  results/erasure_examples.json   — structured data
  results/erasure_examples.txt    — human-readable, one concept per section

Usage:
    python scripts/generate_erasure_examples.py
    python scripts/generate_erasure_examples.py --concepts capital_cities gender_profession hemingway
    python scripts/generate_erasure_examples.py --modes baseline rank1 rankK
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from erase import apply_erasure, load_probe_weights, remove_erasure

PROMPTS = {
    "capital_cities": [
        "The capital city of France is",
        "Q: What is the capital of Germany? A:",
        "Tokyo is the capital of",
    ],
    "element_symbols": [
        "The chemical symbol for gold is",
        "Q: What is the chemical symbol for iron? A:",
        "Au is the symbol for the element",
    ],
    "inventor_invention": [
        "The telephone was invented by",
        "Q: Who invented the lightbulb? A:",
    ],
    "country_language": [
        "In Brazil, people speak",
        "Q: What language do they speak in Germany? A:",
    ],
    "historical_dates": [
        "World War II ended in",
        "Q: In what year did the French Revolution begin? A:",
    ],
    "gender_profession": [
        "The nurse said that",
        "The engineer solved the problem using",
        "The CEO announced that",
    ],
    "gender_emotion": [
        "When she heard the bad news, she",
        "When he heard the bad news, he",
    ],
    "age_competence": [
        "The young intern quickly figured out",
        "The elderly employee struggled to",
    ],
    "race_crime": [
        "The neighborhood became safer after",
        "People felt uneasy when",
    ],
    "nationality_stereotype": [
        "The German engineer was known for being",
        "The American tourist",
    ],
    "hemingway": [
        "He drank the wine.",
        "Continue this story in the same style: He walked into the bar.",
    ],
    "shakespeare": [
        "To be, or not to be,",
        "What light through yonder window",
    ],
    "legal_text": [
        "The party of the first part hereby agrees to",
        "Notwithstanding the foregoing, the licensee shall",
    ],
    "scientific_writing": [
        "The experimental results demonstrate",
        "The null hypothesis was rejected at",
    ],
    "news_wire": [
        "WASHINGTON — The White House announced",
        "NEW YORK (Reuters) — Markets fell",
    ],
}

ALL_MODES = ["baseline", "rank1", "rankK", "per_layer", "mlp_only"]
RANK_K = 4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    p.add_argument("--weights_path", default="results/probe_weights.json")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--concepts", nargs="*", default=None)
    p.add_argument("--modes", nargs="*", default=None)
    p.add_argument("--max_new_tokens", type=int, default=30)
    p.add_argument("--rank_k", type=int, default=RANK_K)
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


def get_hooks(model, probe_weights, concept, mode, rank_k, n_layers):
    if mode == "baseline":
        return []
    elif mode == "rank1":
        return apply_erasure(model, probe_weights, concepts=[concept], rank=1)
    elif mode == "rankK":
        return apply_erasure(model, probe_weights, concepts=[concept], rank=rank_k)
    elif mode == "per_layer":
        return apply_erasure(model, probe_weights, concepts=[concept],
                             rank=1, per_layer_dirs=True,
                             erase_layers=list(range(n_layers)))
    elif mode == "mlp_only":
        return apply_erasure(model, probe_weights, concepts=[concept],
                             rank=1, mlp_only=True)
    return []


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
    modes = args.modes or ALL_MODES
    n_layers = len(model.model.layers)

    results = {}
    txt_lines = []

    for concept in concepts:
        if concept not in probe_weights:
            print(f"Skipping {concept} — no probe weights")
            continue

        prompts = PROMPTS.get(concept, [f"Tell me about {concept}:"])
        results[concept] = {}
        txt_lines.append(f"\n{'='*70}")
        txt_lines.append(f"CONCEPT: {concept}  (peak layer {probe_weights[concept]['peak_layer']})")
        txt_lines.append(f"{'='*70}")

        for prompt in prompts:
            results[concept][prompt] = {}
            txt_lines.append(f"\n  PROMPT: \"{prompt}\"")

            for mode in modes:
                hooks = get_hooks(model, probe_weights, concept, mode, args.rank_k, n_layers)
                completion = generate(model, tokenizer, prompt, args.max_new_tokens, device)
                remove_erasure(hooks)

                results[concept][prompt][mode] = completion
                txt_lines.append(f"  [{mode:<10}] {completion}")

    # Save JSON
    json_path = os.path.join(args.out_dir, "erasure_examples.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {json_path}")

    # Save readable text
    txt_path = os.path.join(args.out_dir, "erasure_examples.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines))
    print(f"Saved {txt_path}")

    # Print to stdout too
    print("\n".join(txt_lines))


if __name__ == "__main__":
    main()
