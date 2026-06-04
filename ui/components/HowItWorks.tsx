"use client";

const STYLE_CONCEPTS  = ["Hemingway", "Legal", "Scientific", "Shakespeare", "Austen"];
const BIAS_CONCEPTS   = ["Age / Competence", "Gender / Profession", "Gender / Emotion", "Race / Crime", "Nationality"];
const FACTUAL_CONCEPTS = ["Capital Cities", "Country Language", "Element Symbols", "Historical Dates", "Inventors"];

const CATEGORIES = [
  { label: "Stylistic", color: "#a78bfa", desc: "Writing style and register", items: STYLE_CONCEPTS },
  { label: "Bias",      color: "#f87171", desc: "Social stereotypes", items: BIAS_CONCEPTS },
  { label: "Factual",   color: "#fbbf24", desc: "World knowledge", items: FACTUAL_CONCEPTS },
];

const LAYERS = [1, 2, 3, 4, 5, "·", "·", "·", 23, 24, 25, 26, "·", "·", 28];

function ResidualStreamDiagram() {
  return (
    <div className="rounded-2xl border border-border bg-card p-6 overflow-x-auto">
      <div className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-5">
        Residual stream · Llama 3.2-3B · 28 transformer layers
      </div>

      {/* top row: layer boxes */}
      <div className="relative min-w-[680px]">

        {/* residual stream rail */}
        <div className="absolute left-[52px] right-[52px] top-[38px] h-px bg-white/10" />

        {/* layer row */}
        <div className="flex items-center justify-between relative">

          {/* input token */}
          <div className="flex flex-col items-center gap-1.5 z-10">
            <div className="w-10 h-10 rounded-lg border border-white/15 bg-background flex items-center justify-center">
              <span className="text-[9px] font-mono text-muted-foreground/60 text-center leading-tight">token<br/>emb.</span>
            </div>
            <span className="text-[9px] font-mono text-muted-foreground/40">input</span>
          </div>

          {/* arrow right */}
          <div className="flex-1 h-px bg-white/10 relative">
            <span className="absolute right-0 -top-1.5 text-muted-foreground/30 text-xs">›</span>
          </div>

          {LAYERS.map((l, i) => (
            <div key={i} className="flex flex-col items-center gap-1.5 z-10">
              {typeof l === "number" ? (
                <>
                  <div
                    className="w-9 h-9 rounded-lg border flex items-center justify-center"
                    style={{
                      borderColor: (l >= 23 && l <= 26) ? "rgba(167,139,250,0.5)" : "rgba(255,255,255,0.08)",
                      background:  (l >= 23 && l <= 26) ? "rgba(167,139,250,0.08)" : "rgb(14,14,18)",
                    }}
                  >
                    <span className="text-[9px] font-mono" style={{ color: (l >= 23 && l <= 26) ? "#a78bfa" : "rgba(160,160,175,0.5)" }}>
                      L{l}
                    </span>
                  </div>
                  <span className="text-[9px] font-mono text-muted-foreground/30">layer {l}</span>
                </>
              ) : (
                <>
                  <div className="w-9 h-9 flex items-end justify-center pb-2">
                    <span className="text-muted-foreground/20 text-lg font-bold tracking-widest">·</span>
                  </div>
                  <span className="text-[9px] font-mono text-muted-foreground/20">···</span>
                </>
              )}
            </div>
          ))}

          {/* arrow right */}
          <div className="flex-1 h-px bg-white/10 relative">
            <span className="absolute right-0 -top-1.5 text-muted-foreground/30 text-xs">›</span>
          </div>

          {/* output */}
          <div className="flex flex-col items-center gap-1.5 z-10">
            <div className="w-10 h-10 rounded-lg border border-white/15 bg-background flex items-center justify-center">
              <span className="text-[9px] font-mono text-muted-foreground/60 text-center leading-tight">next<br/>token</span>
            </div>
            <span className="text-[9px] font-mono text-muted-foreground/40">output</span>
          </div>
        </div>

        {/* intervention annotation below layers 23–26 */}
        <div className="mt-3 flex justify-center">
          <div className="inline-flex flex-col items-center gap-1">
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-violet-400/30 bg-violet-400/05">
              <span className="font-mono text-[11px]" style={{ color: "#a78bfa" }}>h&apos; = h + α · v̂</span>
              <span className="text-[10px] text-muted-foreground/50 ml-1">injected at layers 23–26</span>
            </div>
            <p className="text-[10px] font-mono text-muted-foreground/40 text-center">
              v̂ = unit contrast direction &nbsp;·&nbsp; α controls strength &nbsp;·&nbsp; no weights modified
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function ConceptCategories() {
  return (
    <div className="grid grid-cols-3 gap-4">
      {CATEGORIES.map(cat => (
        <div key={cat.label} className="rounded-xl border border-border bg-card p-5">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-2 h-2 rounded-full" style={{ background: cat.color }} />
            <span className="text-sm font-semibold font-mono" style={{ color: cat.color }}>{cat.label}</span>
          </div>
          <p className="text-xs text-muted-foreground mb-3">{cat.desc}</p>
          <ul className="space-y-1">
            {cat.items.map(item => (
              <li key={item} className="text-xs font-mono text-muted-foreground/60 flex items-center gap-1.5">
                <span style={{ color: cat.color + "60" }}>—</span>
                {item}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

export default function HowItWorks() {
  return (
    <div className="pt-10 space-y-8">
      {/* header */}
      <div>
        <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-muted-foreground mb-3">
          <span className="w-4 h-px bg-primary/60" />
          How It Works
        </div>
        <h2 className="text-3xl font-bold tracking-tight mb-2">
          Finding where concepts live inside a model
        </h2>
        <p className="text-sm text-muted-foreground max-w-2xl leading-relaxed">
          Language models process text as a sequence of vectors flowing through transformer layers.
          At each layer, the residual stream carries a hidden state that encodes everything the model
          knows about the token so far. We train linear probes on those hidden states to find the
          <em className="text-foreground/80"> direction</em> in that high-dimensional space that
          corresponds to a specific concept — then add or subtract that direction at inference time.
        </p>
      </div>

      {/* residual stream diagram */}
      <ResidualStreamDiagram />

      {/* concept categories */}
      <div>
        <div className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-4">
          21 concept directions across three categories
        </div>
        <ConceptCategories />
      </div>

      {/* bridge to the graph */}
      <div className="rounded-xl border border-border bg-card p-6">
        <div className="text-sm font-semibold mb-2">
          Do these 21 directions have any structure relative to each other?
        </div>
        <p className="text-xs text-muted-foreground leading-relaxed max-w-2xl">
          Once you have a direction for every concept, you can measure the cosine similarity between
          each pair. If concepts that share register — legal and scientific writing, for instance —
          end up geometrically close, the model&apos;s representation space is doing something
          interpretable. The network graph below maps exactly that. The answer is yes.
        </p>
        <div className="mt-3 flex items-center gap-2 text-xs font-mono text-muted-foreground/50">
          <span className="w-4 h-px bg-primary/40" />
          Scroll down to see the full geometry
        </div>
      </div>
    </div>
  );
}
