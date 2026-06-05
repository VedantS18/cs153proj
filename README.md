# NeuralStyle: Mapping and Steering Concept Directions in LLM Activation Space

**CS 153 · Stanford · 2026 · Vedant Srinivas · vedants8@stanford.edu**

**Track:** Research + Application/Product (hybrid)

---

## Why I Built This

Language models are deployed in hiring systems, medical tools, and writing assistants used by hundreds of millions of people — yet we have almost no visibility into what they have actually learned. You can inspect outputs. You cannot look inside.

The interpretability tools that exist today are mostly post-hoc: you look at what a model produces and infer what it learned. What I wanted was a research tool that lets you look directly at the model's representation space — find where a concept actually lives as a geometric direction in the activation space — and then interact with that direction causally, at inference time, without any retraining.

The intended applications are **AI safety and interpretability**: giving researchers a concrete way to probe what a model has internalized, measure it quantitatively, and test interventions on it. The secondary application is writing tools — once you have a concept direction, you can steer style without fine-tuning.

The bottleneck I identified: published steering vector work (RepE, activation addition) looks at one or two concepts in isolation. Nobody had done a systematic analysis of how concept directions relate to each other geometrically, and whether that geometry predicts which interventions will work before you run them. That's what this project does.

---

## Key Findings

| Finding | Result |
|---|---|
| Style cosine std vs. random | **0.17 vs. 0.018** — nearly 10× more structured |
| Top aligned concept pair | legal ↔ scientific: **+0.584** (shared formal register) |
| Top opposed pair | scientific ↔ Shakespeare: **−0.381** |
| Bias direction geometry | Near-random; no interpretable pairwise structure |
| Probe / contrast direction alignment (bias) | **0.16 cosine** — nearly orthogonal; erasing the probe direction removes the wrong thing |
| Age/competence erasure (CrowS-Pairs) | 71.3% → 47% (−24 pp at α=20) |
| Age/competence erasure (LLM judge) | ~71% → **0%** at α=10 |
| Gender/emotion erasure (LLM judge) | ~66% → **0%** at α=15 |
| Gender/profession erasure (LLM judge) | ~66% → **0%** at α=5 |
| Nationality erasure (LLM judge) | ~58% → **0%** at α=10 |
| Race/crime erasure | Only partial (~21%) — geometry predicts this in advance |
| Style resurrection after layer-1 erasure | Reconstructed by layer 3 in all style concepts |
| Bias resurrection after layer-1 erasure | Stays suppressed — shallow encoding to begin with |
| Cross-model replication | Structure holds on Llama 3B and Mistral 7B independently |
| Style concept stable layer | Hemingway: layer 21/28; bias concepts: layer 5/28 |

**The resurrection finding is the most surprising:** the model actively fights to preserve style concepts downstream. Erasing the direction at a single layer does not stick — the network rebuilds it. Bias concepts don't resurrect, but this is evidence they were shallowly encoded to begin with, which is another geometric explanation for why behavioral erasure is noisier.

---

## How It Works

### 1 — Extract concept directions

For each concept, labeled positive and negative examples are collected:

- **Stylistic (11):** Real passages from Project Gutenberg (Hemingway PG-67138, Fitzgerald PG-64317, Austen PG-1342, Dickens PG-98, Woolf PG-5765) plus Shakespeare, McCarthy, Rowling, legal contracts, scientific abstracts, news wire copy
- **Bias (5):** Sentence pairs from CrowS-Pairs (age/competence, gender/profession, gender/emotion, race/crime, nationality stereotype)
- **Factual (5):** Matched pairs for capital cities, country languages, element symbols, historical dates, inventor-invention pairs

A forward pass through Llama 3.2-3B (28 transformer layers, 3072-dimensional hidden states) extracts residual stream activations at each layer. A linear probe (logistic regression) is trained on those activations to find the direction separating positive from negative examples. The **contrast direction** is computed separately as:

```
v̂_contrast = unit( mean(positive activations) − mean(negative activations) )
```

This is the vector used for steering — not the probe weight. For bias concepts, the probe weight and the contrast direction have only 0.16 cosine similarity. They are nearly orthogonal. Erasing what the probe identifies is not the same as erasing what governs model behavior — a finding with direct practical implications for auditing tools.

### 2 — Analyze geometry

Pairwise cosine similarity across all 21 contrast directions produces a 21×21 matrix. Concepts that share register (legal and scientific writing) are geometrically close (+0.58). Concepts that oppose in tone (Hemingway and legal) are opposed (−0.31). Rowling and Shakespeare cluster together (+0.57). Bias directions are near-random relative to everything.

The pairwise cosine std across the 11 style directions is **0.17**. Random unit vectors in 3072D have std of **0.018**. The structure is not an artifact of the method.

### 3 — Steer at inference

At generation time, a forward hook modifies the residual stream at layers 23–26:

```
h' = h + α · v̂        # style injection (add)
h' = h − α · v̂        # bias erasure (subtract)
```

No weights are modified. No fine-tuning. No system prompt change. The model continues generation from the modified hidden state. `α` controls strength — style concepts collapse above α≈20–40; bias erasure is effective at α=10–15 before coherence degrades.

### 4 — Evaluate

- **CrowS-Pairs**: 300+ sentence pairs; measure fraction where model assigns higher log-probability to the stereotyped sentence. 50% = unbiased.
- **LLM-as-judge**: 18 neutral prompts per bias concept completed twice (baseline + erased). An independent Claude Haiku judge labels each completion as stereotyped or counter-stereotyped.
- **MMLU**: General knowledge benchmark at moderate alpha to confirm the intervention isn't broadly degrading the model.
- **Ablation (erasure method sweep)**: Five methods compared — baseline, rank-1 nullspace, rank-K subspace, per-layer RepE, MLP-only. Per-layer RepE performs best; MLP-only worsens some metrics.

---

## Benchmark Results

### Bias Erasure (LLM-as-Judge)

| Concept | Baseline | α=5 | α=10 | α=15 | α=20 |
|---|---|---|---|---|---|
| Age / Competence | ~71% | 5.6% | **0%** | 0% | 0% |
| Gender / Profession | ~66% | **0%** | 0% | 6.7% | 4.4% |
| Gender / Emotion | ~66% | 3.3% | 3.3% | **0%** | — |
| Nationality | ~58% | **0%** | 0% | 5.6% | 5.6% |
| Race / Crime | ~58% | 25% | 20.8% | 29.2% | 25% |

Race/crime is the honest outlier. The contrast direction for race/crime has the lowest cosine alignment with the CrowS-Pairs behavioral direction of any bias concept. The geometry predicts this in advance — before running any generation experiment.

### Bias Erasure (CrowS-Pairs, all alphas)

Age/competence drops from 71.3% → 47% at α=20 (−24 pp). Same direction, sign flipped, amplifies from 71% → 85% — a 38pp controllable range. MMLU stays intact at moderate alpha (intervention is specific, not broadly degrading).

### Style Steering

All 11 style concepts steer cleanly at α=10–15. Precomputed examples at α=5/10/20/40 show the progression. The demo uses live generation on the GPU cluster — not cached outputs.

Vector arithmetic: legal + scientific reinforce (cosine +0.58); Hemingway + legal fight (cosine −0.31). The network graph predicts both outcomes before any generation is run. The geometry is predictive.

---

## Use Cases

**AI safety / interpretability** — auditing tool for deployed models. Run any text through the concept scanner to measure which social and stylistic directions are active in the model's representation. Measure stereotype preference quantitatively without touching model weights. The probe/contrast direction misalignment finding means naive probe-based auditing gives false confidence — this project shows a better approach.

**Bias measurement and reduction** — a way to quantify what a model has encoded about social groups without retraining, and to test whether an intervention actually changes the behavioral direction or just changes a proxy.

**Writing tools** — precise style control without fine-tuning. At production scale, adding a 3072-dimensional vector at inference time adds microseconds per forward pass. The blend panel shows you can simultaneously inject two style directions — aligned ones reinforce, opposing ones fight.

**Research** — the geometric structure finding (style vs. bias directionality) is testable on any transformer model. The resurrection experiment is a concrete diagnostic for whether a concept has deep vs. shallow encoding. Both are reusable methods.

---

## Architecture

```
Stanford Farmshare GPU cluster (oat-01)
└── Slurm job: scripts/serve_steer.py
    ├── Llama 3.2-3B-Instruct loaded on CUDA
    ├── 21 contrast direction vectors loaded from disk
    ├── GET  /health
    ├── POST /steer    — single-concept style injection
    ├── POST /blend    — two-vector arithmetic (α₁·v̂₁ + α₂·v̂₂)
    ├── POST /scan     — 21-probe cosine activation scan on input text
    └── POST /debias   — log-probability gap on sentence pairs

SSH tunnel: localhost:8787 → oat-01:8787

Local machine
└── Next.js (ui/)
    ├── app/page.tsx             — 5-tab layout
    ├── app/api/steer/           — proxy routes to inference server
    └── components/
        ├── HowItWorks.tsx       — residual stream diagram + concept overview
        ├── StyleInput.tsx       — live steering + precomputed α examples
        ├── StyleBlend.tsx       — vector arithmetic panel
        ├── ConceptNetwork.tsx   — D3 network graph + 21×21 heatmap
        ├── ConceptScanner.tsx   — live concept activation probe
        ├── ExplainTab.tsx       — geometry section with layer stability
        ├── ResurrectionChart.tsx — post-erasure reconstruction curves
        ├── BiasCompletions.tsx  — before/after completions with judge labels
        ├── BiasLab.tsx          — live log-probability gap measurement
        └── MeasureTab.tsx       — benchmark charts + LLM-judge results
```

---

## Repository Structure

```
cs153proj/
├── scripts/
│   ├── fetch_pubdomain_styles.py       download Project Gutenberg texts
│   ├── build_dataset.py                labeled training data (bias + factual)
│   ├── build_pubdomain_dataset.py      labeled training data (style, with negatives)
│   ├── extract_activations.py          forward pass → residual stream activations
│   ├── train_probes.py                 linear probe training per concept per layer
│   ├── compute_contrast_directions.py  mean-difference contrast vectors
│   ├── analyze_subspace_overlap.py     21×21 pairwise cosine similarity
│   ├── sweep_repe.py                   RepE alpha sweep (CrowS-Pairs + MMLU)
│   ├── sweep_resurrection.py           post-erasure probe tracking by layer
│   ├── sweep_erasure_modes.py          ablation: 5 erasure methods compared
│   ├── judge_bias_erasure.py           LLM-as-judge generation evaluation
│   ├── generate_report.py              aggregate all results → JSON report
│   └── serve_steer.py                  FastAPI inference server
├── jobs/                               SLURM job scripts for Farmshare
│   ├── run_full_pipeline.sh            end-to-end: data → probes → analysis
│   ├── run_repe.sh                     alpha sweep
│   ├── run_resurrection.sh             resurrection experiment
│   ├── run_mode_sweep.sh               erasure method ablation
│   ├── run_judge_bias.sh               LLM judge evaluation
│   ├── run_mistral_comparison.sh       Mistral 7B replication
│   └── run_steer_server.sh             launch inference server on GPU node
├── results/
│   ├── report_llama3b.json             full experiment report (all concepts)
│   ├── subspace_overlap.json           21×21 cosine similarity matrix
│   ├── repe_sweep.json                 CrowS-Pairs stereotype rates vs alpha
│   ├── direction_rotation_llama3b.json layer-by-layer direction rotation
│   ├── mode_sweep_*.json               erasure method ablation results
│   └── mistral/                        Mistral 7B replication results
└── ui/
    ├── app/                            Next.js app router + API proxy routes
    ├── components/                     React components (see Architecture above)
    └── public/data/                    precomputed JSON served to the UI
```

---

## Reproducing the Research

### Requirements

- Python 3.10+
- PyTorch with CUDA (tested on A30, 24GB VRAM)
- `transformers`, `scikit-learn`, `numpy`, `fastapi`, `uvicorn`
- HuggingFace access token with Llama 3.2 gated model access
- SLURM cluster recommended (tested on Stanford Farmshare)

### Full pipeline

```bash
git clone https://github.com/vedants8/cs153proj
cd cs153proj
pip install -r requirements.txt
export HF_TOKEN=your_token_here

# 1. Fetch Gutenberg texts for style concepts
python scripts/fetch_pubdomain_styles.py

# 2. Build labeled datasets
python scripts/build_pubdomain_dataset.py
python scripts/build_dataset.py

# 3. Extract residual stream activations (requires GPU)
python scripts/extract_activations.py

# 4. Train linear probes
python scripts/train_probes.py

# 5. Compute contrast directions
python scripts/compute_contrast_directions.py

# 6. Run analysis
python scripts/analyze_subspace_overlap.py
python scripts/sweep_repe.py
python scripts/sweep_resurrection.py
python scripts/sweep_erasure_modes.py
python scripts/judge_bias_erasure.py

# 7. Generate consolidated report
python scripts/generate_report.py
```

On Farmshare, each step has a corresponding SLURM job in `jobs/`. The full pipeline chains them end-to-end:

```bash
sbatch jobs/run_full_pipeline.sh
```

### Running the demo locally

```bash
# On your GPU machine: start the inference server
python scripts/serve_steer.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --directions results/contrast_directions.json \
  --port 8787

# In a second terminal: start the UI
cd ui && npm install && npm run dev
# → http://localhost:3000
```

The UI proxies `/api/steer` to `localhost:8787`. The green **LIVE inference** badge turns on when the server is reachable.

---

## Limitations

- **Race/crime erasure is only partial.** The contrast direction for this concept is not well-aligned with the CrowS-Pairs behavioral direction. The geometry predicts this — but does not fix it.
- **LLM judge validation accuracy is 38%** on this dataset. The directional results (stereotype rate decreasing) are robust to this noise, but absolute numbers should be treated as approximate.
- **Style resurrection** means single-layer erasure does not work. Multi-layer intervention (layers 23–26 simultaneously) is used throughout, but further reconstruction may occur at layers outside this window.
- **3B model limits style quality.** Larger models produce cleaner style transfer; the 3B model was used because it fits in a single Farmshare GPU slot.
- **CrowS-Pairs is US-centric.** Nationality stereotype and cross-cultural bias concepts may not generalize to other cultural contexts.
- **Probe training data is small.** 200 positive / 200 negative examples per concept. Larger datasets would sharpen the directions.

---

## What's Next

1. **70B-scale evaluation** — whether the style/bias geometry gap holds at 70B+ parameters and whether style steering remains coherent at that scale
2. **Automated concept discovery** — finding interpretable directions without labeled data using sparse autoencoders or contrastive self-supervised methods
3. **Production inference layer** — real-time style and bias auditing as middleware on every API call, not a research one-off
4. **Causal mediation analysis** — which attention heads and MLP layers are causally responsible for style resurrection vs. passive carry-through

---

## Data Sources

| Source | Used For | License |
|---|---|---|
| [Project Gutenberg](https://www.gutenberg.org) | Style concept training data | Public domain |
| [CrowS-Pairs](https://github.com/nyu-mll/crows-pairs) | Bias concept training data + evaluation benchmark | CC BY 4.0 |
| Legal contracts (public domain) | `legal_text` concept direction | Public domain |
| arXiv abstracts (public) | `scientific_writing` concept direction | arXiv non-exclusive license |
| Reuters news wire (public) | `news_wire` concept direction | Public domain samples |

---

## AI Usage Disclosure

**Claude Code** (Anthropic) was the primary development tool throughout this project:

- All UI components were built with Claude Code: React/Next.js components, D3 network graph, recharts visualizations, dark-mode design system
- FastAPI inference server and SLURM job scripts were written with Claude Code assistance
- Data transformation scripts and the report generation pipeline were largely written with Claude Code
- The `results/report_llama3b.json` and all experiment outputs are from actual GPU runs on Stanford Farmshare — no results were generated or fabricated by AI

**All research design, experiment hypotheses, analysis interpretation, and scientific findings are original.** Claude Code was used as a coding tool, not as a research collaborator. The geometric analysis, subspace overlap interpretation, resurrection experiment design, and the core finding (style vs. bias directionality) were designed and interpreted by the author. Commit history documents the development process, including iterations and dead ends (e.g., early probe-based erasure that turned out to target the wrong direction, leading to the probe/contrast direction misalignment finding).

---

## Citations

- Zou, A. et al. "Representation Engineering: A Top-Down Approach to AI Transparency." arXiv 2023.
- Nangia, N. et al. "CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models." EMNLP 2020.
- Meta AI. "Llama 3.2." 2024.
- Jiang, A. et al. "Mistral 7B." arXiv 2023.
- Project Gutenberg texts: PG-67138 (Hemingway), PG-64317 (Fitzgerald), PG-1342 (Austen), PG-98 (Dickens), PG-5765 (Woolf).
