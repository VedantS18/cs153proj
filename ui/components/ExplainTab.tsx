"use client";

import { useEffect, useState } from "react";
import ConceptNetwork from "./ConceptNetwork";

const CAT_COLOR: Record<string, string> = {
  style: "#a78bfa", bias: "#f87171", factual: "#fbbf24",
};
const SHORT: Record<string, string> = {
  hemingway: "Hemingway", shakespeare: "Shakespeare", jk_rowling: "Rowling",
  cormac_mccarthy: "McCarthy", legal_text: "Legal", scientific_writing: "Scientific",
  news_wire: "News Wire", fitzgerald: "Fitzgerald", austen: "Austen",
  dickens: "Dickens", woolf: "Woolf",
  age_competence: "Age/Comp.", gender_profession: "Gender/Prof.",
  race_crime: "Race/Crime", gender_emotion: "Gender/Emo.",
  nationality_stereotype: "Nationality",
  capital_cities: "Capitals", country_language: "Languages",
  element_symbols: "Elements", historical_dates: "Hist. Dates",
  inventor_invention: "Inventors",
};

type StabilityEntry = { concept: string; category: string; stable_layer: number };

function LayerStability() {
  const [data, setData] = useState<StabilityEntry[]>([]);
  useEffect(() => {
    fetch("/data/layer_stability.json").then(r => r.json()).then(setData).catch(() => null);
  }, []);

  if (!data.length) return null;

  const maxLayer = 28;
  const byCategory = {
    bias:    data.filter(d => d.category === "bias"),
    factual: data.filter(d => d.category === "factual"),
    style:   data.filter(d => d.category === "style"),
  };

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground leading-relaxed max-w-2xl">
        Each concept direction undergoes a full rotation in the early layers as the model processes the input,
        then settles into a stable direction. The layer at which it stabilises tells you how deeply the concept
        is encoded. Bias and factual concepts stabilise in the first few layers. Style concepts take much longer —
        Hemingway doesn&apos;t settle until layer 21 out of 28. That persistence across layers is part of why
        style directions are more geometrically structured and easier to steer reliably.
      </p>
      {(["bias","factual","style"] as const).map(cat => (
        <div key={cat}>
          <div className="text-xs font-mono mb-1.5" style={{ color: CAT_COLOR[cat] }}>
            {cat === "style" ? "Stylistic" : cat === "bias" ? "Bias" : "Factual"}
          </div>
          <div className="space-y-1">
            {byCategory[cat].map(d => {
              const pct = (d.stable_layer / maxLayer) * 100;
              return (
                <div key={d.concept} className="flex items-center gap-3">
                  <div className="w-24 text-right text-xs font-mono text-muted-foreground/60 flex-shrink-0">
                    {SHORT[d.concept] ?? d.concept}
                  </div>
                  <div className="flex-1 relative h-4 flex items-center">
                    <div className="w-full h-2 rounded-sm bg-white/5" />
                    <div className="absolute left-0 h-2 rounded-sm transition-all"
                      style={{ width: `${pct}%`, background: CAT_COLOR[cat] + (cat === "style" ? "cc" : "66") }} />
                  </div>
                  <div className="text-xs font-mono w-16 flex-shrink-0"
                    style={{ color: CAT_COLOR[cat] + (cat === "style" ? "" : "99") }}>
                    layer {d.stable_layer}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ))}
      <div className="text-xs text-muted-foreground/40 font-mono">
        Bar length = stable layer / 28 total layers. Longer bar = concept encoded deeper into the network.
      </div>
    </div>
  );
}

export default function ExplainTab() {
  return (
    <div className="pt-10 space-y-10">

      {/* header */}
      <div>
        <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-muted-foreground mb-3">
          <span className="w-4 h-px bg-primary/60" />
          Concept Direction Geometry
        </div>
        <h2 className="text-3xl font-bold tracking-tight mb-2">
          How concepts are organized in activation space
        </h2>
        <p className="text-sm text-muted-foreground max-w-2xl leading-relaxed">
          Each trained probe defines a direction in the model&apos;s 3072-dimensional residual stream.
          The network below maps the cosine similarity between every pair of those directions —
          nodes are concepts, edges connect pairs with meaningful alignment (|cos| ≥ 0.10).
          Click any node to see its training examples and nearest neighbors.
        </p>
      </div>

      {/* network graph + heatmap */}
      <ConceptNetwork />

      {/* three callout cards */}
      <div className="grid grid-cols-3 gap-4">
        <div className="rounded-xl border border-border bg-card p-5">
          <div className="text-4xl font-bold mb-2" style={{ color: CAT_COLOR.style }}>17×</div>
          <div className="text-sm font-semibold mb-2">Stylistic directions are geometrically structured</div>
          <div className="text-xs text-muted-foreground leading-relaxed">
            Pairwise cosine std across the 5 stylistic directions is 0.32. Random unit vectors
            in 3072D have std of 0.018. Stylistic concepts occupy a coherent subspace
            with interpretable relationships — legal and scientific share a direction (+0.58),
            while Hemingway and legal are nearly opposed (−0.31).
          </div>
        </div>

        <div className="rounded-xl border border-border bg-card p-5">
          <div className="text-4xl font-bold mb-2" style={{ color: "oklch(0.72 0.20 270)" }}>0.16</div>
          <div className="text-sm font-semibold mb-2">Probe direction ≠ behavior direction for bias</div>
          <div className="text-xs text-muted-foreground leading-relaxed">
            The probe weight vector (what the classifier learns) has only 0.16 cosine overlap
            with the contrast direction that governs model behavior on CrowS-Pairs.
            Bias probes are nearly orthogonal to the directions that actually matter —
            erasing the probe direction removes the wrong thing.
          </div>
        </div>

        <div className="rounded-xl border border-border bg-card p-5">
          <div className="text-4xl font-bold mb-2" style={{ color: CAT_COLOR.factual }}>1B = 3B</div>
          <div className="text-sm font-semibold mb-2">Geometric structure replicates across model sizes</div>
          <div className="text-xs text-muted-foreground leading-relaxed">
            Llama 1B and 3B independently learn the same pairwise similarity structure.
            Sign matches across all top pairs; top pair (legal ↔ scientific) scores
            +0.58 in 3B and +0.44 in 1B. The structure reflects the training data,
            not model-specific features.
          </div>
        </div>
      </div>

      {/* layer stability */}
      <div className="rounded-2xl border border-border bg-card p-6">
        <div className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-1">
          Where in the network each concept lives
        </div>
        <div className="text-sm font-semibold mb-4">
          Style directions stabilise much deeper than bias or factual directions
        </div>
        <LayerStability />
      </div>

      {/* mechanism explanation */}
      <div className="rounded-xl border border-border bg-card p-6">
        <div className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-4">
          How the steering intervention works
        </div>
        <div className="grid grid-cols-3 gap-6">
          <div>
            <div className="text-xs font-mono text-primary mb-2">1 — Learn direction</div>
            <div className="text-xs text-muted-foreground leading-relaxed">
              A linear probe is trained on residual stream activations extracted from
              positive and negative examples of each concept. The probe weight vector
              defines a direction in activation space. A contrast direction is computed
              separately as mean(positive activations) − mean(negative activations).
            </div>
          </div>
          <div>
            <div className="text-xs font-mono text-primary mb-2">2 — Intervene at inference</div>
            <div className="bg-black/40 rounded-lg p-3 font-mono text-sm text-center mb-2">
              h&apos; = h + α · v̂
            </div>
            <div className="text-xs text-muted-foreground leading-relaxed">
              A forward hook adds α times the unit contrast vector to the residual stream
              at layers 23–26 simultaneously. No weights are modified. The sign of α
              controls direction: positive suppresses, negative amplifies.
            </div>
          </div>
          <div>
            <div className="text-xs font-mono text-primary mb-2">3 — Measure effect</div>
            <div className="text-xs text-muted-foreground leading-relaxed">
              Style concepts: free-generation completions are compared to baseline.
              Bias concepts: CrowS-Pairs log-probability scores measure how often
              the model favors stereotyped over counterstereotyped sentences.
              α sweeps from 0 to 60; coherence collapses above ~20 for style concepts.
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}
