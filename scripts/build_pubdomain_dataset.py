"""
Rebuild stylistic concept dataset using real public-domain text.

Replaces synthetic Hemingway examples with actual text from
The Sun Also Rises (1926) and adds new author concepts:
  fitzgerald  — The Great Gatsby (1925)
  austen      — Pride and Prejudice (1813)
  dickens     — A Tale of Two Cities (1859)
  woolf       — Mrs Dalloway (1925)

Each concept uses sentences from its own work as positives and
sentences from other concepts as negatives (cross-style negatives
provide cleaner contrastive signal than generic prose).

Run AFTER fetch_pubdomain_styles.py has populated data/pubdomain/.
"""

import json
import random
import os
from pathlib import Path

random.seed(42)

NEW_CONCEPTS = ["hemingway", "fitzgerald", "austen", "dickens", "woolf"]

# Negatives for each new concept: pull from the most stylistically distant others
# and from the existing non-author concepts (legal, scientific, news_wire)
CROSS_NEG_SOURCES = {
    "hemingway":  ["fitzgerald", "austen", "dickens", "woolf"],  # verbose vs sparse
    "fitzgerald": ["hemingway", "dickens"],                       # sparse vs lyrical
    "austen":     ["hemingway", "woolf", "dickens"],
    "dickens":    ["hemingway", "fitzgerald", "woolf"],
    "woolf":      ["hemingway", "dickens", "austen"],
}

# Extra non-literary negatives for each — keeps the boundary clean
EXTRA_NEGATIVES = {
    "hemingway": [
        "The magnificent iridescence of the twilight sky filled her heart with ineffable longing.",
        "In the labyrinthine corridors of the ancient palace she contemplated her existence.",
        "His mercurial temperament rendered him simultaneously captivating and insufferable.",
        "She gazed upon the resplendent tapestry of stars, each a testament to indifference.",
        "The experimental results demonstrate a statistically significant correlation.",
        "Notwithstanding the foregoing, the licensee shall retain all rights not expressly granted.",
        "WASHINGTON — The president announced Monday that he would sign the bill into law.",
        "The committee voted to approve the proposed budget changes by a narrow margin.",
        "She updated the spreadsheet and shared it with the rest of the team.",
        "He checked the weather forecast before deciding whether to bring an umbrella.",
    ],
    "fitzgerald": [
        "He paid the bill. They went out.",
        "The coffee was cold. He did not mind.",
        "She left. He did not watch her go.",
        "The road was empty. He drove fast.",
        "They ate and slept.",
        "LONDON — The prime minister addressed Parliament on the ongoing trade dispute.",
        "The party of the first part hereby agrees to indemnify and hold harmless.",
        "The experimental results demonstrate a statistically significant difference.",
        "She ordered a coffee and checked her messages before the meeting started.",
        "He set an alarm for seven and went to bed early.",
    ],
    "austen": [
        "He drank the wine. It was good.",
        "So we beat on, boats against the current, borne back into the past.",
        "It was the best of times, it was the worst of times.",
        "The null hypothesis was rejected at a significance level of p < 0.05.",
        "Pursuant to Section 3(b), the obligor shall remit payment within thirty days.",
        "NEW YORK (Reuters) — Markets fell sharply amid concerns over rising inflation.",
        "She updated the spreadsheet and sent the revised version to the team.",
        "The package arrived one day earlier than expected.",
        "He finished the assignment and submitted it through the online portal.",
        "The maintenance team repaired the elevator late in the afternoon.",
    ],
    "dickens": [
        "She drank. He watched.",
        "In my younger and more vulnerable years my father gave me advice.",
        "The experimental results demonstrate a statistically significant correlation.",
        "The party of the first part hereby agrees to indemnify the second part.",
        "SAN FRANCISCO — The company disclosed a data breach affecting millions.",
        "She ordered a coffee and checked her phone before the meeting.",
        "He set an alarm and went to bed without saying anything.",
        "The quarterly report shows an increase in user engagement.",
        "The flight was delayed three hours due to fog at the destination.",
        "She attended the networking event and exchanged contact information.",
    ],
    "woolf": [
        "He drank the wine. It was cold.",
        "It was the best of times, it was the worst of times, it was the age of wisdom.",
        "It is a truth universally acknowledged, that a single man in possession of a fortune wants a wife.",
        "The null hypothesis was rejected at the standard significance level.",
        "Pursuant to Section 3, the obligor shall remit payment within thirty days.",
        "BEIJING (AP) — Chinese officials confirmed the delegation would arrive ahead of schedule.",
        "She updated the spreadsheet and emailed it to the rest of the team.",
        "He checked the weather app and decided to bring an umbrella.",
        "The quarterly earnings report exceeded analyst expectations.",
        "She set an alarm for six and turned off the light.",
    ],
}

CATEGORY = "stylistic"


def load_pubdomain(concept: str) -> list[str]:
    path = Path("data/pubdomain") / f"{concept}.json"
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping {concept}")
        return []
    data = json.load(open(path))
    return data.get("sentences", [])


def build_pairs(concept: str, all_sentences: dict, n_pos: int = 200, n_neg: int = 200) -> list[dict]:
    pos_pool = all_sentences[concept]
    if not pos_pool:
        return []

    # Negatives: cross-concept sentences + extra negatives
    neg_pool = []
    for src in CROSS_NEG_SOURCES.get(concept, []):
        neg_pool.extend(all_sentences.get(src, []))
    neg_pool.extend(EXTRA_NEGATIVES.get(concept, []))

    random.shuffle(neg_pool)

    pairs = []
    for text in random.choices(pos_pool, k=n_pos):
        pairs.append({"text": text, "label": 1, "concept": concept, "category": CATEGORY})
    for text in random.choices(neg_pool, k=n_neg) if neg_pool else []:
        pairs.append({"text": text, "label": 0, "concept": concept, "category": CATEGORY})

    random.shuffle(pairs)
    return pairs


def main():
    # Load all pubdomain sentences
    all_sentences = {}
    for concept in NEW_CONCEPTS:
        sentences = load_pubdomain(concept)
        all_sentences[concept] = sentences
        print(f"  {concept}: {len(sentences)} sentences from {Path('data/pubdomain/' + concept + '.json')}")

    out_dir = Path("data/concepts")
    out_dir.mkdir(parents=True, exist_ok=True)

    for concept in NEW_CONCEPTS:
        pairs = build_pairs(concept, all_sentences)
        if not pairs:
            print(f"  SKIP {concept} — no data")
            continue
        out_path = out_dir / f"{concept}.jsonl"
        with open(out_path, "w") as f:
            for item in pairs:
                f.write(json.dumps(item) + "\n")
        pos = sum(1 for p in pairs if p["label"] == 1)
        neg = sum(1 for p in pairs if p["label"] == 0)
        print(f"  Wrote {out_path}: {pos} pos, {neg} neg")

    print("\nDone. Run train_probes.py for each new concept next.")


if __name__ == "__main__":
    main()
