"""
Generate a comprehensive experiment report tying all results together.

For each concept and model, produces:
  - Model and concept metadata
  - Probe localization (accuracy at each layer, peak layer)
  - Concept direction rotation across layers
  - Resurrection curve (probe recovery after erasure at peak layer)
  - Erasure results: behavioral score + CrowS-Pairs rate at each alpha/layer
  - LLM-as-judge completions: before/after with labels at each alpha
  - Summary: what worked, what didn't

Outputs:
  results/report_<tag>.json      -- full structured data
  results/report_<tag>.md        -- human-readable markdown

Usage:
    python scripts/generate_report.py --tag llama3b
    python scripts/generate_report.py --tag llama1b --results_dir results/llama1b
"""

import argparse
import json
import os
from datetime import datetime


CONCEPT_CATEGORY = {
    "capital_cities": "factual",   "element_symbols": "factual",
    "inventor_invention": "factual", "country_language": "factual",
    "historical_dates": "factual",
    "gender_profession": "bias",   "gender_emotion": "bias",
    "age_competence": "bias",      "race_crime": "bias",
    "nationality_stereotype": "bias",
    "hemingway": "stylistic",      "shakespeare": "stylistic",
    "legal_text": "stylistic",     "scientific_writing": "stylistic",
    "news_wire": "stylistic",
}


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="llama3b")
    p.add_argument("--results_dir", default="results",
                   help="Directory with result JSON files for this model")
    p.add_argument("--base_results_dir", default="results",
                   help="Directory with base results (probe_weights, resurrection, etc.)")
    p.add_argument("--out_dir", default="results")
    return p.parse_args()


def fmt(v, decimals=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def build_concept_report(concept, tag, results_dir, base_dir, all_data):
    """Build the full data dict for one concept."""
    r = {"concept": concept, "category": CONCEPT_CATEGORY.get(concept, "unknown")}

    # ── 1. Probe localization ──────────────────────────────────────────────
    probe_accs = all_data.get("probe_accuracies", {}).get(concept, {})
    pw = all_data.get("probe_weights", {}).get(concept, {})
    peak_layer = pw.get("peak_layer")
    r["probe_peak_layer"] = peak_layer
    r["probe_accuracy_at_peak"] = probe_accs.get(str(peak_layer)) or probe_accs.get(peak_layer)
    r["probe_accuracy_by_layer"] = {
        int(k): round(v, 4) for k, v in probe_accs.items()
    }

    # ── 2. Direction rotation ──────────────────────────────────────────────
    rot = all_data.get("direction_rotation", {}).get("rotation", {}).get(concept, {})
    flip = all_data.get("direction_rotation", {}).get("flip_layer", {}).get(concept)
    r["direction_rotation"] = {int(k): v for k, v in rot.items()}
    r["direction_flip_layer"] = flip
    if rot:
        r["min_cosine_similarity"] = min(rot.values())

    # ── 3. Resurrection curve ──────────────────────────────────────────────
    res = all_data.get("resurrection", {}).get(concept, {})
    if res:
        r["resurrection_layer"] = res.get("resurrection_layer")
        r["probe_post_erasure_by_layer"] = res.get("post_erasure", {})
        r["probe_baseline_by_layer"] = res.get("baseline", {})

    # ── 4. Alpha sweep (behavioral scores + CrowS-Pairs) ──────────────────
    alpha_data = all_data.get("alpha_sweep", {})
    baseline = alpha_data.get("baseline", {}).get(concept, {})
    r["behavioral_baseline"] = {
        "concept_test": baseline.get("concept_test"),
        "crows_stereo_rate": baseline.get("crows"),
        "mmlu": baseline.get("mmlu"),
    }

    alpha_results = []
    for layer_key in sorted(alpha_data.keys()):
        if layer_key == "baseline":
            continue
        layer_idx = int(layer_key)
        for alpha_key, concepts_data in alpha_data[layer_key].items():
            cd = concepts_data.get(concept, {})
            ct_base = baseline.get("concept_test")
            crows_base = baseline.get("crows")
            ct = cd.get("concept_test")
            crows = cd.get("crows")
            alpha_results.append({
                "layer": layer_idx,
                "alpha": float(alpha_key),
                "concept_test": ct,
                "concept_test_delta": round(ct - ct_base, 4) if ct is not None and ct_base is not None else None,
                "crows_stereo_rate": crows,
                "crows_delta": round(crows - crows_base, 4) if crows is not None and crows_base is not None else None,
                "mmlu": cd.get("mmlu"),
            })
    r["alpha_sweep"] = alpha_results

    # ── 5. LLM-as-judge completions ────────────────────────────────────────
    judge = all_data.get("judge_bias", {}).get(concept)
    if judge:
        r["judge_eval"] = {}
        for alpha_key, jdata in judge.items():
            if alpha_key == "baseline":
                continue
            r["judge_eval"][alpha_key] = {
                "stereotype_rate": jdata.get("stereotype_rate"),
                "neutral_rate": jdata.get("neutral_rate"),
                "counter_rate": jdata.get("counter_rate"),
                "examples": jdata.get("completions", [])[:5],  # first 5 examples
            }

    return r


def to_markdown(tag, reports, judge_validation_acc=None):
    lines = []
    lines.append(f"# Concept Erasure Experiment Report — {tag}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("This report documents the full concept erasure experiment pipeline:")
    lines.append("1. **Probe Localization** — where in the network is each concept linearly decodable?")
    lines.append("2. **Direction Rotation** — how does the concept direction change across layers?")
    lines.append("3. **Resurrection Curve** — after erasing at the probe peak, does the concept reconstitute?")
    lines.append("4. **Alpha Sweep** — at what steering strength does behavioral erasure occur?")
    lines.append("5. **LLM-as-Judge** — does the model actually produce less stereotyped text?")
    lines.append("")

    if judge_validation_acc is not None:
        lines.append(f"> **Judge validation accuracy:** {judge_validation_acc:.0%}")
        lines.append("")

    for category in ["factual", "bias", "stylistic"]:
        cat_reports = [r for r in reports if r["category"] == category]
        if not cat_reports:
            continue

        lines.append(f"---")
        lines.append(f"## {category.upper()} CONCEPTS")
        lines.append("")

        for r in cat_reports:
            concept = r["concept"]
            lines.append(f"### {concept.replace('_', ' ').title()}")
            lines.append("")

            # Probe localization
            lines.append("**Probe Localization**")
            lines.append(f"- Peak layer: {r.get('probe_peak_layer', 'n/a')}")
            lines.append(f"- Accuracy at peak: {fmt(r.get('probe_accuracy_at_peak'))}")
            by_layer = r.get("probe_accuracy_by_layer", {})
            if by_layer:
                layer_str = "  ".join(
                    f"L{l}:{fmt(a)}" for l, a in sorted(by_layer.items()) if l % 4 == 0
                )
                lines.append(f"- By layer (every 4th): {layer_str}")
            lines.append("")

            # Direction rotation
            lines.append("**Direction Rotation**")
            lines.append(f"- Min cosine similarity with layer-1 direction: {fmt(r.get('min_cosine_similarity'))}")
            lines.append(f"- First flip layer (cos < 0): {r.get('direction_flip_layer', 'never')}")
            lines.append("")

            # Resurrection curve
            if "resurrection_layer" in r:
                lines.append("**Resurrection Curve** (probe accuracy after erasing at peak layer)")
                lines.append(f"- Concept reconstitutes above 75% at layer: {r.get('resurrection_layer', 'never')}")
                post = r.get("probe_post_erasure_by_layer", {})
                if post:
                    post_str = "  ".join(
                        f"L{l}:{fmt(v)}" for l, v in sorted((int(k), v) for k, v in post.items()) if int(l) % 4 == 0
                    )
                    lines.append(f"- Post-erasure probe by layer: {post_str}")
                lines.append("")

            # Behavioral baseline
            bl = r.get("behavioral_baseline", {})
            lines.append("**Baseline (no erasure)**")
            lines.append(f"- Concept test score: {fmt(bl.get('concept_test'))}")
            if bl.get("crows_stereo_rate") is not None:
                lines.append(f"- CrowS-Pairs stereotype rate: {fmt(bl.get('crows_stereo_rate'))} (50% = unbiased)")
            lines.append(f"- MMLU: {fmt(bl.get('mmlu'))}")
            lines.append("")

            # Alpha sweep
            alpha_results = r.get("alpha_sweep", [])
            if alpha_results:
                lines.append("**Alpha Sweep** (steering strength vs. behavioral change)")
                lines.append("")
                lines.append("| Layer | Alpha | Concept Δ | CrowS Δ | MMLU |")
                lines.append("|---|---|---|---|---|")
                for ar in alpha_results:
                    lines.append(
                        f"| {ar['layer']} | {ar['alpha']} | "
                        f"{fmt(ar.get('concept_test_delta'))} | "
                        f"{fmt(ar.get('crows_delta'))} | "
                        f"{fmt(ar.get('mmlu'))} |"
                    )
                lines.append("")

            # Judge eval
            judge_eval = r.get("judge_eval", {})
            if judge_eval:
                lines.append("**LLM-as-Judge Stereotype Rate** (lower = more erasure)")
                lines.append("")
                lines.append("| Alpha | Stereo% | Neutral% | Counter% |")
                lines.append("|---|---|---|---|")
                for alpha_key in sorted(judge_eval.keys(), key=float):
                    je = judge_eval[alpha_key]
                    lines.append(
                        f"| {alpha_key} | "
                        f"{fmt(je.get('stereotype_rate'), 1)} | "
                        f"{fmt(je.get('neutral_rate'), 1)} | "
                        f"{fmt(je.get('counter_rate'), 1)} |"
                    )
                lines.append("")

                # Show example completions at baseline and max alpha
                alphas = sorted(judge_eval.keys(), key=float)
                for ak in [alphas[0], alphas[-1]]:
                    examples = judge_eval[ak].get("examples", [])[:3]
                    if examples:
                        lines.append(f"*Example completions at α={ak}:*")
                        for ex in examples:
                            lines.append(f"> **[{ex.get('label', '?')}]** Prompt: \"{ex.get('prompt', '')}\"")
                            lines.append(f">  → \"{ex.get('completion', '')}\"")
                        lines.append("")

            lines.append("")

    return "\n".join(lines)


def main():
    args = parse_args()
    rdir = args.results_dir
    bdir = args.base_results_dir
    os.makedirs(args.out_dir, exist_ok=True)

    # Load all result files
    all_data = {
        "probe_accuracies": load_json(f"{bdir}/probe_accuracies.json") or {},
        "probe_weights":    load_json(f"{bdir}/probe_weights.json") or {},
        "direction_rotation": load_json(f"{rdir}/direction_rotation_{args.tag}.json") or {},
        "resurrection":     load_json(f"{bdir}/resurrection_curve.json") or {},
        "alpha_sweep":      load_json(f"{rdir}/alpha_sweep_{args.tag}.json") or
                            load_json(f"{bdir}/alpha_sweep.json") or {},
        "judge_bias":       load_json(f"{rdir}/judge_bias_{args.tag}.json") or
                            load_json(f"{bdir}/judge_bias_{args.tag}.json") or {},
    }

    # Judge validation accuracy
    judge_val_acc = None
    judge_raw = all_data["judge_bias"]
    if "judge_validation_accuracy" in judge_raw:
        judge_val_acc = judge_raw["judge_validation_accuracy"]

    concepts = sorted(CONCEPT_CATEGORY.keys())
    reports = [
        build_concept_report(c, args.tag, rdir, bdir, all_data)
        for c in concepts
    ]

    # Save JSON
    out_json = {
        "tag": args.tag,
        "generated": datetime.now().isoformat(),
        "judge_validation_accuracy": judge_val_acc,
        "concepts": reports,
    }
    json_path = os.path.join(args.out_dir, f"report_{args.tag}.json")
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"Saved {json_path}")

    # Save Markdown
    md = to_markdown(args.tag, reports, judge_val_acc)
    md_path = os.path.join(args.out_dir, f"report_{args.tag}.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved {md_path}")

    # Print quick summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — {args.tag}")
    print(f"{'='*60}")
    print(f"{'Concept':<25} {'Peak L':>7} {'Flip L':>7} {'Resurr':>7} {'CrowS base':>11} {'Judge base':>11}")
    print("-" * 75)
    for r in reports:
        crows_b = r.get("behavioral_baseline", {}).get("crows_stereo_rate")
        judge_b = None
        if r.get("judge_eval"):
            alpha_keys = sorted(r["judge_eval"].keys(), key=float)
            judge_b = r["judge_eval"][alpha_keys[0]].get("stereotype_rate")
        print(
            f"{r['concept']:<25} "
            f"{str(r.get('probe_peak_layer', '?')):>7} "
            f"{str(r.get('direction_flip_layer', '-')):>7} "
            f"{str(r.get('resurrection_layer', '-')):>7} "
            f"{fmt(crows_b):>11} "
            f"{fmt(judge_b):>11}"
        )


if __name__ == "__main__":
    main()
