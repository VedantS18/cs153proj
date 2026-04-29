"""
Extract residual stream activations for every concept.

For each sentence in data/concepts/<concept>.jsonl, runs a forward pass
through the model and saves the last-token hidden state at every layer.

Output: activations/<concept>.npz
  - X: float32 array of shape (n_examples, n_layers, hidden_dim)
  - y: int32 label array of shape (n_examples,)
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B",
                   help="HF model name or local path")
    p.add_argument("--data_dir", default="data/concepts")
    p.add_argument("--out_dir", default="activations")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=64)
    return p.parse_args()


def load_concept(path):
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(item["label"])
    return texts, labels


@torch.no_grad()
def extract(model, tokenizer, texts, batch_size, max_length, device):
    """Returns float32 array (n, n_layers, hidden_dim) — last token at each layer."""
    all_hidden = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        out = model(**enc, output_hidden_states=True)
        # hidden_states: tuple of (n_layers+1) tensors, each (B, seq_len, hidden_dim)
        # index 0 = embedding layer, 1..n = transformer layers
        # take last non-padding token for each example
        seq_lens = enc["attention_mask"].sum(dim=1) - 1  # (B,)

        # stack all layers: (n_layers+1, B, seq_len, hidden)
        stacked = torch.stack(out.hidden_states, dim=0)  # (L+1, B, T, H)
        # gather last token: (L+1, B, H)
        last = stacked[:, torch.arange(len(batch)), seq_lens, :]  # (L+1, B, H)
        last = last.permute(1, 0, 2)  # (B, L+1, H)
        all_hidden.append(last.cpu().float().numpy())

    return np.concatenate(all_hidden, axis=0)  # (N, L+1, H)


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
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()

    concept_files = sorted(f for f in os.listdir(args.data_dir) if f.endswith(".jsonl"))
    print(f"Found {len(concept_files)} concepts")

    for fname in concept_files:
        concept = fname.replace(".jsonl", "")
        out_path = os.path.join(args.out_dir, f"{concept}.npz")
        if os.path.exists(out_path):
            print(f"  {concept}: already exists, skipping")
            continue

        texts, labels = load_concept(os.path.join(args.data_dir, fname))
        print(f"  {concept}: {len(texts)} examples", flush=True)

        X = extract(model, tokenizer, texts, args.batch_size, args.max_length, device)
        y = np.array(labels, dtype=np.int32)
        np.savez_compressed(out_path, X=X, y=y)
        print(f"    saved {out_path}  shape={X.shape}")

    print("Done.")


if __name__ == "__main__":
    main()
