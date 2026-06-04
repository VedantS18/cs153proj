"""
Local inference server for live RepE steering demo.

Endpoints:
  GET  /health
  POST /steer   — single-concept style injection
  POST /blend   — multi-concept vector arithmetic
  POST /scan    — probe all 21 concept directions on input text
  POST /debias  — subtract a bias direction; compare completions

Usage:
    python scripts/serve_steer.py
    python scripts/serve_steer.py --port 8787 --contrast results/contrast_directions.json
"""

import argparse, json, os, sys, time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Install: pip install fastapi uvicorn pydantic"); sys.exit(1)

sys.path.insert(0, os.path.dirname(__file__))
from erase import apply_repe_steering, load_contrast_directions, remove_erasure

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",          default="meta-llama/Llama-3.2-3B")
parser.add_argument("--contrast",       default="results/contrast_directions.json")
parser.add_argument("--port",           type=int, default=8787)
parser.add_argument("--host",           default="127.0.0.1")
parser.add_argument("--max_new_tokens", type=int, default=120)
args = parser.parse_args()

# ── Model load ────────────────────────────────────────────────────────────────
print(f"Loading tokenizer: {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model: {args.model}  (~30s first time)")
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

n_layers   = len(model.model.layers)
STEER_LAYERS = 4
LAYER_FRAC   = 0.85

CONCEPT_CATEGORY = {
    "hemingway": "style", "shakespeare": "style", "jk_rowling": "style",
    "cormac_mccarthy": "style", "legal_text": "style", "scientific_writing": "style",
    "news_wire": "style", "fitzgerald": "style", "austen": "style",
    "dickens": "style", "woolf": "style",
    "age_competence": "bias", "gender_profession": "bias", "race_crime": "bias",
    "gender_emotion": "bias", "nationality_stereotype": "bias",
    "capital_cities": "factual", "country_language": "factual",
    "element_symbols": "factual", "historical_dates": "factual",
    "inventor_invention": "factual",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
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
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def steer_hooks(concepts: list[str], alphas: list[float], subtract: bool = False) -> list:
    """Return forward hooks for multi-concept steering."""
    hooks = []
    for concept, alpha in zip(concepts, alphas):
        if concept not in contrast_dirs:
            continue
        peak = contrast_dirs[concept]["peak_layer"]
        layer_start = max(0, min(peak, int(n_layers * LAYER_FRAC) - STEER_LAYERS))
        per_layer = alpha / STEER_LAYERS
        for li in range(layer_start, layer_start + STEER_LAYERS):
            h = apply_repe_steering(
                model, contrast_dirs,
                concepts=[concept],
                alpha=per_layer,
                layer=li,
                subtract=subtract,
            )
            hooks.extend(h)
    return hooks


def scan_activations(text: str) -> dict[str, float]:
    """Forward pass; return dot-product score of last-token hidden state with each concept direction."""
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Group by peak layer to minimize hooks
    layer_to_concepts: dict[int, list[str]] = {}
    for c, cd in contrast_dirs.items():
        li = cd["peak_layer"]
        layer_to_concepts.setdefault(li, []).append(c)

    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for li in layer_to_concepts:
        def make_hook(idx):
            def hook(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[idx] = h[0, -1, :].detach().cpu().float()
            return hook
        hooks.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()

    scores = {}
    for concept, cd in contrast_dirs.items():
        li = cd["peak_layer"]
        if li not in captured:
            scores[concept] = 0.0
            continue
        h  = captured[li]
        v  = cd["v_contrast"].cpu().float()
        # True cosine similarity: normalise both h and v
        h_norm = h  / (h.norm()  + 1e-8)
        v_norm = v  / (v.norm()  + 1e-8)
        scores[concept] = round(float(h_norm @ v_norm), 4)
    return scores


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="RepE Steering Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "concepts": sorted(contrast_dirs.keys()), "device": device}


# ── /steer ────────────────────────────────────────────────────────────────────
class SteerRequest(BaseModel):
    text:  str
    style: str
    alpha: float = 5.0

class SteerResponse(BaseModel):
    original: str; steered: str; style: str; alpha: float; latency_ms: float

@app.post("/steer", response_model=SteerResponse)
def steer(req: SteerRequest):
    if req.style not in contrast_dirs:
        raise HTTPException(400, f"Unknown style '{req.style}'")
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text must not be empty")
    t0 = time.time()
    prompt = text.rstrip() + " "
    hooks = steer_hooks([req.style], [req.alpha])
    try:
        steered = generate(prompt, hooks)
    finally:
        remove_erasure(hooks)
    return SteerResponse(
        original=text, steered=steered,
        style=req.style, alpha=req.alpha,
        latency_ms=round((time.time()-t0)*1000, 1),
    )


# ── /blend ────────────────────────────────────────────────────────────────────
class BlendRequest(BaseModel):
    text:     str
    concepts: list[str]   # e.g. ["hemingway", "legal_text"]
    alphas:   list[float] # per-concept strength

class BlendResponse(BaseModel):
    original: str; steered: str; concepts: list[str]; alphas: list[float]; latency_ms: float

@app.post("/blend", response_model=BlendResponse)
def blend(req: BlendRequest):
    if len(req.concepts) != len(req.alphas):
        raise HTTPException(400, "concepts and alphas must have the same length")
    for c in req.concepts:
        if c not in contrast_dirs:
            raise HTTPException(400, f"Unknown concept '{c}'")
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text must not be empty")
    t0 = time.time()
    prompt = text.rstrip() + " "
    hooks = steer_hooks(req.concepts, req.alphas)
    try:
        steered = generate(prompt, hooks)
    finally:
        remove_erasure(hooks)
    return BlendResponse(
        original=text, steered=steered,
        concepts=req.concepts, alphas=req.alphas,
        latency_ms=round((time.time()-t0)*1000, 1),
    )


# ── /scan ─────────────────────────────────────────────────────────────────────
class ScanRequest(BaseModel):
    text: str

class ScanResponse(BaseModel):
    scores:     dict[str, float]   # concept → activation score
    categories: dict[str, str]     # concept → style|bias|factual
    latency_ms: float

@app.post("/scan", response_model=ScanResponse)
def scan(req: ScanRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text must not be empty")
    t0 = time.time()
    scores = scan_activations(text)
    return ScanResponse(
        scores=scores,
        categories=CONCEPT_CATEGORY,
        latency_ms=round((time.time()-t0)*1000, 1),
    )


# ── /debias ───────────────────────────────────────────────────────────────────
# Measures log-probability of stereotype vs counter-stereotype sentences,
# before and after erasing the bias direction. This is exactly what CrowS-Pairs
# measures — not completion quality, but model preference over full sentences.

def sentence_logprob(text: str, hooks: list = []) -> float:
    """Mean per-token log-probability of a sentence under the model (with optional hooks)."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        out = model(**inputs, labels=input_ids)
    return float(-out.loss)   # loss = mean NLL, so -loss = mean log-prob

class DebiasRequest(BaseModel):
    stereo:   str   # stereotyped sentence (full)
    counter:  str   # counter-stereotyped sentence (full, same structure)
    concept:  str
    alpha:    float = 20.0

class DebiasResponse(BaseModel):
    stereo_logprob:          float
    counter_logprob:         float
    stereo_logprob_erased:   float
    counter_logprob_erased:  float
    gap_before:              float   # stereo - counter  (positive = model prefers stereo)
    gap_after:               float
    gap_reduction_pct:       float   # % reduction in gap
    concept:                 str
    alpha:                   float
    latency_ms:              float

@app.post("/debias", response_model=DebiasResponse)
def debias(req: DebiasRequest):
    if req.concept not in contrast_dirs:
        raise HTTPException(400, f"Unknown concept '{req.concept}'")
    if CONCEPT_CATEGORY.get(req.concept) != "bias":
        raise HTTPException(400, f"'{req.concept}' is not a bias concept")
    t0 = time.time()

    # Baseline log-probs
    lp_stereo   = sentence_logprob(req.stereo.strip())
    lp_counter  = sentence_logprob(req.counter.strip())

    # Log-probs with bias direction erased
    hooks = steer_hooks([req.concept], [req.alpha], subtract=True)
    try:
        lp_stereo_e  = sentence_logprob(req.stereo.strip(),   hooks)
        lp_counter_e = sentence_logprob(req.counter.strip(),  hooks)
    finally:
        remove_erasure(hooks)

    gap_before = lp_stereo  - lp_counter
    gap_after  = lp_stereo_e - lp_counter_e
    reduction  = (1 - abs(gap_after) / (abs(gap_before) + 1e-8)) * 100 if gap_before != 0 else 0.0

    return DebiasResponse(
        stereo_logprob=round(lp_stereo, 4),
        counter_logprob=round(lp_counter, 4),
        stereo_logprob_erased=round(lp_stereo_e, 4),
        counter_logprob_erased=round(lp_counter_e, 4),
        gap_before=round(gap_before, 4),
        gap_after=round(gap_after, 4),
        gap_reduction_pct=round(reduction, 1),
        concept=req.concept,
        alpha=req.alpha,
        latency_ms=round((time.time()-t0)*1000, 1),
    )


if __name__ == "__main__":
    print(f"\nStarting server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
