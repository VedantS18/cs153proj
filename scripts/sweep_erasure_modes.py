"""
Comprehensive erasure mode sweep.

Tests five erasure modes across all 15 concepts:
  baseline    — no erasure
  rank1       — project 1 direction at peak layer (original method)
  rankK       — project top-K PCA directions at peak layer (K=4)
  per_layer   — project each layer's own probe direction at every layer
  mlp_only    — project peak-layer direction from MLP sublayer only

For bias concepts (gender_profession, gender_emotion, age_competence,
race_crime, nationality_stereotype) also reports:
  - Stereotype gap (hand-crafted tests)
  - CrowS-Pairs stereo preference rate (external)
  - StereoSet Stereotype Score (external)

For factual: MCQ accuracy.
For stylistic: style choice accuracy.
Collateral damage: MMLU subset for every mode.

Output: results/mode_sweep.json

Usage:
    python scripts/sweep_erasure_modes.py
    python scripts/sweep_erasure_modes.py --concepts capital_cities gender_profession
    python scripts/sweep_erasure_modes.py --modes baseline rank1 rankK
"""

import argparse
import csv
import io
import json
import os
import sys
import urllib.request

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from erase import apply_erasure, load_probe_weights, remove_erasure

BIAS_CONCEPTS = {
    "gender_profession", "gender_emotion", "age_competence",
    "race_crime", "nationality_stereotype",
}

CONCEPT_TO_CROWS_TYPE = {
    "gender_profession": "gender",
    "gender_emotion": "gender",
    "race_crime": "race-color",
    "nationality_stereotype": "nationality",
    "age_competence": "age",
}

CONCEPT_TO_STEREOSET_DOMAINS = {
    "gender_profession": ["gender", "profession"],
    "gender_emotion": ["gender"],
    "race_crime": ["race"],
    "nationality_stereotype": [],
    "age_competence": [],
}

ALL_MODES = ["baseline", "rank1", "rankK", "per_layer", "mlp_only"]
RANK_K = 4

MMLU_SUBSET = [
    ("What is the capital of France?",           ["London", "Berlin", "Paris", "Madrid"],      2),
    ("Which planet is closest to the Sun?",      ["Venus", "Mercury", "Earth", "Mars"],         1),
    ("What is 2 + 2?",                           ["3", "4", "5", "6"],                          1),
    ("Who wrote Romeo and Juliet?",              ["Dickens", "Austen", "Shakespeare", "Chaucer"], 2),
    ("What is the chemical formula for water?",  ["CO2", "H2O", "NaCl", "O2"],                  1),
    ("What is the largest continent?",           ["Africa", "Asia", "Europe", "North America"], 1),
    ("How many sides does a hexagon have?",      ["5", "6", "7", "8"],                          1),
    ("Which element has atomic number 1?",       ["Helium", "Carbon", "Hydrogen", "Oxygen"],    2),
    ("In what year did World War II end?",       ["1943", "1944", "1945", "1946"],               2),
    ("What is the square root of 144?",          ["10", "11", "12", "13"],                      2),
]

_CROWS_CSV_URL = (
    "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/"
    "data/crows_pairs_anonymized.csv"
)
_crows_cache = None


def _load_crows_csv():
    global _crows_cache
    if _crows_cache is not None:
        return _crows_cache
    try:
        with urllib.request.urlopen(_CROWS_CSV_URL, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        _crows_cache = list(csv.DictReader(io.StringIO(content)))
        print(f"  Loaded {len(_crows_cache)} CrowS-Pairs rows")
    except Exception as e:
        print(f"  CrowS-Pairs download failed: {e}")
        _crows_cache = []
    return _crows_cache


_stereoset_cache = None


def _load_stereoset():
    global _stereoset_cache
    if _stereoset_cache is not None:
        return _stereoset_cache
    try:
        from datasets import load_dataset
        ds = load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation")
        _stereoset_cache = list(ds)
        print(f"  Loaded {len(_stereoset_cache)} StereoSet rows")
    except Exception as e:
        print(f"  StereoSet load failed: {e}")
        _stereoset_cache = []
    return _stereoset_cache


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    p.add_argument("--weights_path", default="results/probe_weights.json")
    p.add_argument("--test_dir", default="data/concept_test")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--concepts", nargs="*", default=None)
    p.add_argument("--modes", nargs="*", default=None,
                   help=f"Subset of modes to run (default: {ALL_MODES})")
    p.add_argument("--rank_k", type=int, default=RANK_K)
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--max_crows_pairs", type=int, default=200)
    return p.parse_args()


@torch.no_grad()
def sentence_log_prob(model, tokenizer, sentence, device, max_length):
    enc = tokenizer(sentence, return_tensors="pt",
                    truncation=True, max_length=max_length).to(device)
    out = model(**enc, labels=enc["input_ids"])
    return -out.loss.item() * enc["input_ids"].shape[1]


@torch.no_grad()
def mmlu_score(model, tokenizer, device, max_length):
    correct = 0
    for question, choices, answer_idx in MMLU_SUBSET:
        scores = [sentence_log_prob(model, tokenizer,
                                    f"Question: {question}\nAnswer: {c}",
                                    device, max_length)
                  for c in choices]
        correct += int(np.argmax(scores) == answer_idx)
    return correct / len(MMLU_SUBSET)


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
def crows_score(model, tokenizer, device, bias_type, max_pairs, max_length):
    rows = [r for r in _load_crows_csv() if r["bias_type"] == bias_type]
    if not rows:
        return None
    rows = rows[:max_pairs]
    n_stereo = sum(
        int(sentence_log_prob(model, tokenizer, r["sent_more"], device, max_length)
            > sentence_log_prob(model, tokenizer, r["sent_less"], device, max_length))
        for r in rows
    )
    return n_stereo / len(rows)


@torch.no_grad()
def stereoset_score(model, tokenizer, device, domains, max_length):
    rows = [r for r in _load_stereoset() if r["bias_type"] in domains]
    if not rows:
        return None, None
    STEREO, ANTI, UNRELA = 1, 0, 2
    stereo_wins = lm_correct = stereo_total = lm_total = 0
    for row in rows:
        ctx = row["context"]
        by_label = {lbl: s for s, lbl in
                    zip(row["sentences"]["sentence"], row["sentences"]["gold_label"])}
        if not all(k in by_label for k in (STEREO, ANTI, UNRELA)):
            continue
        lp_s = sentence_log_prob(model, tokenizer, ctx + " " + by_label[STEREO], device, max_length)
        lp_a = sentence_log_prob(model, tokenizer, ctx + " " + by_label[ANTI],   device, max_length)
        lp_u = sentence_log_prob(model, tokenizer, ctx + " " + by_label[UNRELA], device, max_length)
        stereo_wins += int(lp_s > lp_a); stereo_total += 1
        lm_correct += int(lp_s > lp_u) + int(lp_a > lp_u); lm_total += 2
    if stereo_total == 0:
        return None, None
    return 100.0 * stereo_wins / stereo_total, 100.0 * lm_correct / lm_total


def score_concept(model, tokenizer, device, concept, tests, probe_weights,
                  max_length, max_crows_pairs):
    """Score one concept — concept-specific test + bias benchmarks if applicable."""
    result = {}
    result["concept_test"] = concept_test_score(model, tokenizer, device, tests, max_length)

    if concept in BIAS_CONCEPTS:
        crows_type = CONCEPT_TO_CROWS_TYPE.get(concept)
        if crows_type:
            result["crows"] = crows_score(model, tokenizer, device,
                                           crows_type, max_crows_pairs, max_length)
        ss_domains = CONCEPT_TO_STEREOSET_DOMAINS.get(concept, [])
        if ss_domains:
            ss, lms = stereoset_score(model, tokenizer, device, ss_domains, max_length)
            result["stereoset_ss"]  = ss
            result["stereoset_lms"] = lms

    return result


def run_mode(model, tokenizer, device, mode, concepts, concept_tests,
             probe_weights, max_length, max_crows_pairs, rank_k, n_model_layers):
    """Run one erasure mode across all concepts, return results dict."""
    print(f"\n{'='*60}\nMODE: {mode}\n{'='*60}")
    mode_results = {}

    for concept in concepts:
        tests = concept_tests.get(concept, [])

        if mode == "baseline":
            r = score_concept(model, tokenizer, device, concept, tests,
                              probe_weights, max_length, max_crows_pairs)
        else:
            # Build erase kwargs
            kwargs = {}
            if mode == "rank1":
                kwargs = {"rank": 1, "erase_layers": None}
            elif mode == "rankK":
                kwargs = {"rank": rank_k, "erase_layers": None}
            elif mode == "per_layer":
                kwargs = {"rank": 1, "per_layer_dirs": True,
                          "erase_layers": list(range(n_model_layers))}
            elif mode == "mlp_only":
                kwargs = {"rank": 1, "mlp_only": True, "erase_layers": None}

            hooks = apply_erasure(model, probe_weights, concepts=[concept], **kwargs)
            r = score_concept(model, tokenizer, device, concept, tests,
                              probe_weights, max_length, max_crows_pairs)
            remove_erasure(hooks)

        mode_results[concept] = r
        ct = r.get("concept_test")
        crows = r.get("crows")
        ss = r.get("stereoset_ss")
        parts = [f"ct={ct:.4f}" if ct is not None else "ct=n/a"]
        if crows is not None: parts.append(f"crows={crows:.3f}")
        if ss   is not None: parts.append(f"ss={ss:.1f}")
        print(f"  {concept:<25} {' | '.join(parts)}")

    # MMLU once per mode (use first concept's hooks as representative)
    if mode != "baseline":
        if mode == "rank1":
            hooks = apply_erasure(model, probe_weights,
                                  concepts=concepts[:1], rank=1)
        elif mode == "rankK":
            hooks = apply_erasure(model, probe_weights,
                                  concepts=concepts[:1], rank=rank_k)
        elif mode == "per_layer":
            hooks = apply_erasure(model, probe_weights, concepts=concepts[:1],
                                  rank=1, per_layer_dirs=True,
                                  erase_layers=list(range(n_model_layers)))
        elif mode == "mlp_only":
            hooks = apply_erasure(model, probe_weights,
                                  concepts=concepts[:1], rank=1, mlp_only=True)
        mmlu = mmlu_score(model, tokenizer, device, max_length)
        remove_erasure(hooks)
    else:
        mmlu = mmlu_score(model, tokenizer, device, max_length)

    for concept in concepts:
        mode_results[concept]["mmlu"] = mmlu
    print(f"  MMLU: {mmlu:.3f}")
    return mode_results


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
    modes = args.modes or ALL_MODES
    n_model_layers = len(model.model.layers)
    print(f"Model layers: {n_model_layers} | Concepts: {len(concepts)} | Modes: {modes}\n")

    # Pre-load bias benchmarks once
    if any(c in BIAS_CONCEPTS for c in concepts):
        _load_crows_csv()
        _load_stereoset()

    # Load concept tests
    concept_tests = {}
    for concept in concepts:
        path = os.path.join(args.test_dir, f"{concept}.json")
        concept_tests[concept] = json.load(open(path)) if os.path.exists(path) else []

    results = {}
    for mode in modes:
        results[mode] = run_mode(
            model, tokenizer, device, mode, concepts, concept_tests,
            probe_weights, args.max_length, args.max_crows_pairs,
            args.rank_k, n_model_layers,
        )

    out_path = os.path.join(args.out_dir, "mode_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # Summary
    metrics = ["concept_test", "crows", "stereoset_ss", "mmlu"]
    for concept in concepts:
        print(f"\n--- {concept} ---")
        header = f"  {'mode':<12}" + "".join(f"  {m:>14}" for m in metrics)
        print(header)
        for mode in modes:
            r = results[mode].get(concept, {})
            row = f"  {mode:<12}"
            for m in metrics:
                v = r.get(m)
                row += f"  {v:>14.4f}" if isinstance(v, float) else f"  {'n/a':>14}"
            print(row)


if __name__ == "__main__":
    main()
