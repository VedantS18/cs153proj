"""
Local inference server for live RepE steering demo.

Loads Llama-3.2-3B + contrast directions once on startup,
then serves POST /steer requests from the Next.js UI.

Usage:
    python scripts/serve_steer.py
    python scripts/serve_steer.py --port 8787 --contrast results/contrast_directions.json

The server runs at http://localhost:8787 and accepts CORS from localhost:3000.
"""

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# FastAPI / uvicorn
try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Install: pip install fastapi uvicorn pydantic")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(__file__))
from erase import apply_repe_steering, load_contrast_directions, remove_erasure

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",    default="meta-llama/Llama-3.2-3B")
parser.add_argument("--contrast", default="results/contrast_directions.json")
parser.add_argument("--port",     type=int, default=8787)
parser.add_argument("--host",     default="127.0.0.1")
parser.add_argument("--max_new_tokens", type=int, default=120)
args = parser.parse_args()

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading tokenizer: {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model: {args.model}  (this takes ~30s)")
t0 = time.time()
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    device_map=device,
)
model.eval()
print(f"Model loaded on {device} in {time.time()-t0:.1f}s")

print(f"Loading contrast directions: {args.contrast}")
contrast_dirs = load_contrast_directions(args.contrast)
print(f"  {len(contrast_dirs)} concepts: {sorted(contrast_dirs.keys())}")

n_layers = len(model.model.layers)
STEER_LAYERS = 4          # how many layers to steer simultaneously
LAYER_FRAC   = 0.85       # steer at the top 85% of the model

# ── API ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="RepE Steering Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


class SteerRequest(BaseModel):
    text:   str
    style:  str
    alpha:  float = 10.0


class SteerResponse(BaseModel):
    original:  str
    steered:   str
    style:     str
    alpha:     float
    latency_ms: float


def generate(prompt: str, hooks: list, max_new_tokens: int = args.max_new_tokens) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the new tokens
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


@app.get("/health")
def health():
    return {"status": "ok", "concepts": sorted(contrast_dirs.keys()), "device": device}


@app.post("/steer", response_model=SteerResponse)
def steer(req: SteerRequest):
    if req.style not in contrast_dirs:
        from fastapi import HTTPException
        raise HTTPException(400, f"Unknown style '{req.style}'. Available: {sorted(contrast_dirs.keys())}")

    prompt = req.text.strip()
    if not prompt:
        from fastapi import HTTPException
        raise HTTPException(400, "text must not be empty")

    t0 = time.time()

    # Baseline (no steering)
    original = generate(prompt, [])

    # Steered — apply at multiple layers around the concept's peak layer.
    # Divide alpha by layer count so total effective strength matches the alpha value.
    peak = contrast_dirs[req.style]["peak_layer"]
    layer_start = max(0, min(peak, int(n_layers * LAYER_FRAC) - STEER_LAYERS))
    per_layer_alpha = req.alpha / STEER_LAYERS
    hooks = []
    for layer_idx in range(layer_start, layer_start + STEER_LAYERS):
        h = apply_repe_steering(
            model, contrast_dirs,
            concepts=[req.style],
            alpha=per_layer_alpha,
            layer=layer_idx,
            subtract=False,  # inject style
        )
        hooks.extend(h)

    try:
        steered = generate(prompt, hooks)
    finally:
        remove_erasure(hooks)

    return SteerResponse(
        original=original,
        steered=steered,
        style=req.style,
        alpha=req.alpha,
        latency_ms=round((time.time() - t0) * 1000, 1),
    )


if __name__ == "__main__":
    print(f"\nStarting server at http://{args.host}:{args.port}")
    print("UI should call POST http://localhost:8787/steer")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
