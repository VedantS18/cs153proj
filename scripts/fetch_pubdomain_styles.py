"""
Fetch public domain literary texts from Project Gutenberg and extract
sentence-level training examples for stylistic concept probes.

Sources (all US public domain):
  Hemingway  — The Sun Also Rises (1926)     PG 67138
  Fitzgerald — The Great Gatsby (1925)       PG 64317
  Austen     — Pride and Prejudice (1813)    PG 1342
  Dickens    — A Tale of Two Cities (1859)   PG 98
  Woolf      — Mrs Dalloway (1925)           PG 5765

Output: data/pubdomain/<concept>.json   (list of sentence strings)
"""

import re
import ssl
import json
import random
import urllib.request
from pathlib import Path

random.seed(42)

SOURCES = {
    "hemingway": {
        "pg_id": 67138,
        "title": "The Sun Also Rises (Hemingway, 1926)",
        # Hemingway: short declarative, concrete, sparse adjectives
        "min_len": 20, "max_len": 120,
        "prefer_short": True,
        "reject_patterns": [r"^\s*[A-Z\s]{5,}\s*$", r"^Chapter", r"^Book\s"],
    },
    "fitzgerald": {
        "pg_id": 64317,
        "title": "The Great Gatsby (Fitzgerald, 1925)",
        # Fitzgerald: medium-long, lyrical, colour words, first-person
        "min_len": 40, "max_len": 220,
        "prefer_short": False,
        "reject_patterns": [r"^\s*[A-Z\s]{5,}\s*$", r"^Chapter"],
    },
    "austen": {
        "pg_id": 1342,
        "title": "Pride and Prejudice (Austen, 1813)",
        "min_len": 35, "max_len": 200,
        "prefer_short": False,
        "reject_patterns": [r"^\s*[A-Z\s]{5,}\s*$", r"^Chapter", r"^Vol\b"],
    },
    "dickens": {
        "pg_id": 98,
        "title": "A Tale of Two Cities (Dickens, 1859)",
        "min_len": 40, "max_len": 220,
        "prefer_short": False,
        "reject_patterns": [r"^\s*[A-Z\s]{5,}\s*$", r"^Book\s", r"^Chapter"],
    },
    "woolf": {
        "pg_id": 5765,
        "title": "Mrs Dalloway (Woolf, 1925)",
        "min_len": 40, "max_len": 250,
        "prefer_short": False,
        "reject_patterns": [r"^\s*[A-Z\s]{5,}\s*$"],
    },
}

PG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def fetch_text(pg_id: int) -> str:
    url = PG_URL.format(id=pg_id)
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(url, timeout=30, context=ctx) as r:
        raw = r.read().decode("utf-8", errors="ignore")
    # Strip PG header / footer
    start = re.search(r"\*\*\* START OF [^\n]+\n", raw)
    end   = re.search(r"\*\*\* END OF [^\n]+\n",   raw)
    if start:
        raw = raw[start.end():]
    if end:
        raw = raw[:end.start()]
    return raw


def extract_sentences(text: str, cfg: dict, n: int = 60) -> list[str]:
    # Collapse whitespace, split on sentence boundaries
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Simple sentence splitter: split on .!? followed by space+capital
    chunks = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'(])', text)

    sentences = []
    for chunk in chunks:
        s = chunk.strip()
        s = re.sub(r"\s+", " ", s)
        # Basic filters
        if len(s) < cfg["min_len"] or len(s) > cfg["max_len"]:
            continue
        if any(re.search(p, s) for p in cfg["reject_patterns"]):
            continue
        # Skip lines that are mostly caps (headings)
        if sum(1 for c in s if c.isupper()) > len(s) * 0.4:
            continue
        # Skip dialogue-only (starts and ends with quote)
        if s.startswith('"') and s.endswith('"'):
            continue
        # Must end with sentence-final punctuation
        if s[-1] not in ".!?":
            continue
        sentences.append(s)

    if not sentences:
        return []

    if cfg["prefer_short"]:
        # For Hemingway: weight toward shorter sentences
        sentences.sort(key=len)
        pool = sentences[:max(n * 3, len(sentences) // 2)]
    else:
        pool = sentences

    random.shuffle(pool)
    return pool[:n]


def main():
    out_dir = Path("data/pubdomain")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for concept, cfg in SOURCES.items():
        print(f"\n{'='*50}")
        print(f"Fetching: {cfg['title']}")
        try:
            text = fetch_text(cfg["pg_id"])
            print(f"  Downloaded {len(text):,} chars")
            sentences = extract_sentences(text, cfg, n=60)
            print(f"  Extracted {len(sentences)} sentences")
            if sentences:
                print(f"  Sample: {sentences[0][:100]}")
                print(f"  Sample: {sentences[1][:100]}")
            results[concept] = {
                "source": cfg["title"],
                "sentences": sentences,
            }
            out_path = out_dir / f"{concept}.json"
            with open(out_path, "w") as f:
                json.dump(results[concept], f, indent=2)
            print(f"  Saved {out_path}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[concept] = {"source": cfg["title"], "sentences": []}

    # Summary
    print("\n\n=== Summary ===")
    for concept, data in results.items():
        print(f"  {concept:15s} {len(data['sentences']):3d} sentences  ({data['source']})")

    return results


if __name__ == "__main__":
    main()
