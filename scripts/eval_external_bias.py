"""
Evaluate concept erasure on external bias benchmarks: CrowS-Pairs and StereoSet.

Runs log-prob scoring on both benchmarks before/after erasure for the 5 bias concepts:
  gender_profession, gender_emotion, age_competence, race_crime, nationality_stereotype

CrowS-Pairs metric: % of pairs where model assigns P(more-stereotyped) > P(less-stereotyped)
  After erasure of relevant concept: this rate should drop toward 50%

StereoSet metric: Stereotype Score (SS) = % of intrasentence pairs where model prefers stereo
  After erasure: SS should approach 50 (neutral); LMS should stay high

Output: results/external_bias_eval.json

Usage:
    python scripts/eval_external_bias.py --concept gender_profession
    python scripts/eval_external_bias.py  # all 5 bias concepts
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


# CrowS-Pairs bias_type → our concept names (partial match is fine)
CROWS_CONCEPT_MAP = {
    "gender": ["gender_profession", "gender_emotion"],
    "race-color": ["race_crime"],
    "nationality": ["nationality_stereotype"],
    "age": ["age_competence"],
    "socioeconomic": [],
    "religion": [],
    "disability": [],
    "sexual-orientation": [],
    "physical-appearance": [],
}

# StereoSet domain → our concept names
STEREOSET_CONCEPT_MAP = {
    "gender": ["gender_profession", "gender_emotion"],
    "race": ["race_crime"],
    "profession": ["gender_profession"],
    "religion": [],
}

# Which concept applies to which external benchmark category
CONCEPT_TO_CROWS_TYPE = {
    "gender_profession": "gender",
    "gender_emotion": "gender",
    "race_crime": "race-color",
    "nationality_stereotype": "nationality",
    "age_competence": "age",
}

CONCEPT_TO_STEREOSET_DOMAIN = {
    "gender_profession": ["gender", "profession"],
    "gender_emotion": ["gender"],
    "race_crime": ["race"],
    "nationality_stereotype": [],
    "age_competence": [],
}

ALL_BIAS_CONCEPTS = [
    "gender_profession", "gender_emotion", "age_competence",
    "race_crime", "nationality_stereotype",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    p.add_argument("--weights_path", default="results/probe_weights.json")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--concepts", nargs="*", default=None)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--max_pairs", type=int, default=200,
                   help="Max CrowS-Pairs per concept category (speeds up eval)")
    return p.parse_args()


@torch.no_grad()
def sentence_log_prob(model, tokenizer, sentence, device, max_length):
    """Log-probability of the entire sentence under the model."""
    enc = tokenizer(sentence, return_tensors="pt",
                    truncation=True, max_length=max_length).to(device)
    out = model(**enc, labels=enc["input_ids"])
    # loss is mean NLL; multiply back by token count for total log-prob
    n_tokens = enc["input_ids"].shape[1]
    return -out.loss.item() * n_tokens


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
        reader = csv.DictReader(io.StringIO(content))
        _crows_cache = list(reader)
        print(f"    Loaded {len(_crows_cache)} CrowS-Pairs rows from GitHub")
    except Exception as e:
        print(f"    Could not download CrowS-Pairs CSV: {e}")
        _crows_cache = []
    return _crows_cache


def eval_crows_concept(model, tokenizer, device, bias_type, max_pairs, max_length):
    """
    Load CrowS-Pairs from raw CSV, filter to bias_type, return (n_stereo_preferred, n_total).
    Stereo-preferred = model assigns higher log-prob to more-stereotyped sentence.
    """
    rows = [r for r in _load_crows_csv() if r["bias_type"] == bias_type]
    if not rows:
        print(f"    No CrowS-Pairs rows for bias_type={bias_type}")
        return None, None

    rows = rows[:max_pairs]
    n_stereo = 0
    for row in rows:
        lp_more = sentence_log_prob(model, tokenizer, row["sent_more"], device, max_length)
        lp_less = sentence_log_prob(model, tokenizer, row["sent_less"], device, max_length)
        if lp_more > lp_less:
            n_stereo += 1

    return n_stereo, len(rows)


def eval_stereoset_concept(model, tokenizer, device, domains, max_length):
    """
    Load StereoSet intrasentence, filter to domains, return (stereotype_score, lm_score).
    SS = % of (stereo, anti) pairs where model prefers stereo.
    LMS = % of (stereo/anti, unrelated) pairs where model prefers meaningful sentence.

    StereoSet gold_label integers: 0=anti-stereotype, 1=stereotype, 2=unrelated
    """
    if not domains:
        return None, None

    try:
        from datasets import load_dataset
        ds = load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation")
    except Exception as e:
        print(f"    Could not load stereoset: {e}")
        return None, None

    rows = [r for r in ds if r["bias_type"] in domains]
    if not rows:
        return None, None

    # StereoSet label mapping (integer)
    STEREO = 1
    ANTI   = 0
    UNRELA = 2

    stereo_wins = 0
    stereo_total = 0
    lm_correct = 0
    lm_total = 0

    for row in rows:
        context   = row["context"]
        sents     = row["sentences"]["sentence"]
        labels    = row["sentences"]["gold_label"]

        by_label = {label: sent for sent, label in zip(sents, labels)}

        if STEREO not in by_label or ANTI not in by_label or UNRELA not in by_label:
            continue

        stereo_text = context + " " + by_label[STEREO]
        anti_text   = context + " " + by_label[ANTI]
        unrela_text = context + " " + by_label[UNRELA]

        lp_stereo = sentence_log_prob(model, tokenizer, stereo_text, device, max_length)
        lp_anti   = sentence_log_prob(model, tokenizer, anti_text,   device, max_length)
        lp_unrela = sentence_log_prob(model, tokenizer, unrela_text, device, max_length)

        stereo_wins  += int(lp_stereo > lp_anti)
        stereo_total += 1

        lm_correct += int(lp_stereo > lp_unrela)
        lm_correct += int(lp_anti   > lp_unrela)
        lm_total   += 2

    if stereo_total == 0:
        return None, None

    ss  = 100.0 * stereo_wins / stereo_total
    lms = 100.0 * lm_correct / lm_total
    return ss, lms


def run_concept(model, tokenizer, device, concept, probe_weights,
                max_pairs, max_length):
    """Returns dict with before/after scores for CrowS-Pairs and StereoSet."""
    crows_type   = CONCEPT_TO_CROWS_TYPE.get(concept)
    stereo_doms  = CONCEPT_TO_STEREOSET_DOMAIN.get(concept, [])

    result = {"concept": concept}

    # ---- CrowS-Pairs ----
    if crows_type:
        print(f"  CrowS-Pairs ({crows_type})...")
        n_before, n_total = eval_crows_concept(
            model, tokenizer, device, crows_type, max_pairs, max_length)

        if n_before is not None:
            hooks = apply_erasure(model, probe_weights, concepts=[concept])
            n_after, _ = eval_crows_concept(
                model, tokenizer, device, crows_type, max_pairs, max_length)
            remove_erasure(hooks)

            result["crows_stereo_rate_before"] = n_before / n_total
            result["crows_stereo_rate_after"]  = n_after  / n_total
            result["crows_n"]                  = n_total
            result["crows_delta"]              = (n_after - n_before) / n_total
            print(f"    Before: {n_before/n_total:.3f}  After: {n_after/n_total:.3f}")
        else:
            result["crows_stereo_rate_before"] = None
            result["crows_stereo_rate_after"]  = None

    # ---- StereoSet ----
    if stereo_doms:
        print(f"  StereoSet (domains={stereo_doms})...")
        ss_before, lms_before = eval_stereoset_concept(
            model, tokenizer, device, stereo_doms, max_length)

        if ss_before is not None:
            hooks = apply_erasure(model, probe_weights, concepts=[concept])
            ss_after, lms_after = eval_stereoset_concept(
                model, tokenizer, device, stereo_doms, max_length)
            remove_erasure(hooks)

            result["stereoset_ss_before"]  = ss_before
            result["stereoset_ss_after"]   = ss_after
            result["stereoset_lms_before"] = lms_before
            result["stereoset_lms_after"]  = lms_after
            result["stereoset_ss_delta"]   = ss_after - ss_before
            print(f"    SS before: {ss_before:.1f}  SS after: {ss_after:.1f}")
            print(f"    LMS before: {lms_before:.1f}  LMS after: {lms_after:.1f}")
        else:
            result["stereoset_ss_before"] = None
            result["stereoset_ss_after"]  = None

    return result


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
    concepts = args.concepts or ALL_BIAS_CONCEPTS
    print(f"Evaluating {len(concepts)} bias concepts on external benchmarks\n")

    results = {}
    for concept in concepts:
        if concept not in probe_weights:
            print(f"Skipping {concept} — no probe weights")
            continue
        print(f"\n--- {concept} ---")
        results[concept] = run_concept(
            model, tokenizer, device, concept, probe_weights,
            args.max_pairs, args.max_length,
        )

    out_path = os.path.join(args.out_dir, "external_bias_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'Concept':<25} {'CrowS Δ':>10} {'SS Δ':>8} {'LMS Δ':>8}")
    print("-" * 55)
    for concept, r in results.items():
        crows_d = f"{r['crows_delta']:+.3f}" if r.get("crows_delta") is not None else "  n/a"
        ss_d    = f"{r['stereoset_ss_delta']:+.1f}" if r.get("stereoset_ss_delta") is not None else " n/a"
        lms_d   = f"{r['stereoset_lms_after'] - r['stereoset_lms_before']:+.1f}" \
                  if r.get("stereoset_lms_before") is not None else " n/a"
        print(f"{concept:<25} {crows_d:>10} {ss_d:>8} {lms_d:>8}")


if __name__ == "__main__":
    main()
