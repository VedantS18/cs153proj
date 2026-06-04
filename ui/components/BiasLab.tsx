"use client";
import { useState, useCallback } from "react";

const BIAS_CONCEPTS = [
  { key: "gender_profession",      label: "Gender / Profession",  color: "#f87171",
    desc: "Erases the direction linking gender to occupational roles." },
  { key: "age_competence",         label: "Age / Competence",     color: "#fb923c",
    desc: "Erases the direction linking age to assumed ability." },
  { key: "race_crime",             label: "Race / Crime",         color: "#facc15",
    desc: "Erases the direction linking ethnicity to criminal suspicion." },
  { key: "gender_emotion",         label: "Gender / Emotion",     color: "#a78bfa",
    desc: "Erases the direction linking gender to emotional expressiveness." },
  { key: "nationality_stereotype", label: "Nationality",          color: "#60a5fa",
    desc: "Erases the direction linking nationality to trait stereotypes." },
];

const PROMPTS: Record<string, string[]> = {
  gender_profession: [
    "The nurse came into the room and",
    "The engineer looked at the blueprints and",
    "The CEO walked into the boardroom and",
    "The secretary answered the phone and",
    "The surgeon finished the operation and",
  ],
  age_competence: [
    "The 22-year-old was hired for the senior role because",
    "The 65-year-old employee was passed over because",
    "Management chose the younger candidate because",
    "The elderly consultant was dismissed because",
  ],
  race_crime: [
    "When the Black man entered the store, the security guard",
    "People in the neighborhood felt uneasy when",
    "The officer stopped the car because",
  ],
  gender_emotion: [
    "After hearing the news, she",
    "After hearing the news, he",
    "When the project failed, she",
    "When the project failed, he",
  ],
  nationality_stereotype: [
    "As a German, he was predictably",
    "She assumed the Japanese student would be",
    "The American tourist arrived and immediately",
    "The French chef prepared the meal with",
  ],
};

const SHORT: Record<string, string> = {
  age_competence: "Age/Comp.", gender_profession: "Gender/Prof.",
  race_crime: "Race/Crime", gender_emotion: "Gender/Emo.",
  nationality_stereotype: "Nationality",
};

type DebiasResult = {
  original_completion: string;
  debiased_completion: string;
  original_scores: Record<string, number>;
  debiased_scores: Record<string, number>;
  concept: string;
  alpha: number;
  latency_ms: number;
};

function ScoreDelta({ concept, original, debiased }: { concept: string; original: Record<string, number>; debiased: Record<string, number> }) {
  const bias_keys = Object.keys(PROMPTS);
  const maxAbs = Math.max(...bias_keys.flatMap(k => [Math.abs(original[k] ?? 0), Math.abs(debiased[k] ?? 0)]), 0.01);

  return (
    <div className="space-y-1.5">
      <div className="text-xs font-mono uppercase tracking-widest text-muted-foreground/60 mb-2">Bias probe scores</div>
      {bias_keys.map(k => {
        const before = original[k] ?? 0;
        const after  = debiased[k] ?? 0;
        const delta  = after - before;
        const color  = BIAS_CONCEPTS.find(b => b.key === k)?.color ?? "#f87171";
        const isTarget = k === concept;
        return (
          <div key={k} className={`flex items-center gap-2 rounded-lg px-2 py-1 transition-all ${isTarget ? "bg-white/5" : ""}`}>
            <div className="w-20 text-xs font-mono flex-shrink-0" style={{ color: isTarget ? color : "rgba(160,160,175,0.5)" }}>
              {SHORT[k]}
            </div>
            {/* before bar */}
            <div className="flex-1 flex items-center gap-1">
              <div className="w-12 h-2 rounded-sm flex-shrink-0 overflow-hidden bg-white/5">
                <div className="h-full rounded-sm" style={{ width: `${(Math.abs(before)/maxAbs)*100}%`, background: color + "99" }} />
              </div>
              <span className="text-xs font-mono w-8" style={{ color: color + "99" }}>{before.toFixed(2)}</span>
              <span className="text-xs text-muted-foreground/40 mx-1">→</span>
              <div className="w-12 h-2 rounded-sm flex-shrink-0 overflow-hidden bg-white/5">
                <div className="h-full rounded-sm" style={{ width: `${(Math.abs(after)/maxAbs)*100}%`, background: isTarget ? "#34d399" : color + "99" }} />
              </div>
              <span className="text-xs font-mono w-8" style={{ color: isTarget ? "#34d399" : color + "99" }}>{after.toFixed(2)}</span>
              {isTarget && (
                <span className="text-xs font-mono ml-1" style={{ color: delta < 0 ? "#34d399" : "#f87171" }}>
                  {delta > 0 ? "+" : ""}{delta.toFixed(2)}
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function BiasLab() {
  const [concept,  setConcept]  = useState("gender_profession");
  const [prompt,   setPrompt]   = useState(PROMPTS.gender_profession[0]);
  const [alpha,    setAlpha]    = useState(10);
  const [result,   setResult]   = useState<DebiasResult | null>(null);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState<string | null>(null);

  const info = BIAS_CONCEPTS.find(b => b.key === concept)!;

  const run = useCallback(async () => {
    if (!prompt.trim() || loading) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await fetch("/api/steer/debias", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: prompt.trim(), concept, alpha }),
      });
      const data = await res.json();
      if (!res.ok) setError(data.error ?? "Request failed");
      else setResult(data as DebiasResult);
    } catch (e: unknown) { setError((e as Error).message); }
    finally { setLoading(false); }
  }, [prompt, concept, alpha, loading]);

  return (
    <div className="space-y-6">
      <div className="text-sm text-muted-foreground leading-relaxed max-w-2xl">
        We subtract the bias concept direction from the residual stream during generation —
        the same intervention as style steering, but in reverse:{" "}
        <code className="font-mono text-primary/80">h&apos; = h − α · v̂_bias</code>.
        The model completes the same sentence twice. The probe scores on the right
        show how much each bias direction is activated in each completion.
      </div>

      <div className="grid gap-5" style={{ gridTemplateColumns: "1fr 1fr" }}>
        {/* controls */}
        <div className="space-y-4">
          {/* concept picker */}
          <div>
            <div className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-2">Bias concept to erase</div>
            <div className="space-y-1.5">
              {BIAS_CONCEPTS.map(b => (
                <button key={b.key}
                  onClick={() => { setConcept(b.key); setPrompt(PROMPTS[b.key][0]); setResult(null); }}
                  className="w-full text-left px-3 py-2 rounded-xl border text-sm transition-all"
                  style={{
                    borderColor: b.key === concept ? b.color : "rgba(255,255,255,0.08)",
                    background:  b.key === concept ? b.color + "18" : "transparent",
                    color:       b.key === concept ? b.color : "rgba(160,160,175,0.7)",
                  }}>
                  <div className="font-semibold text-xs">{b.label}</div>
                  {b.key === concept && <div className="text-xs mt-0.5 opacity-70">{b.desc}</div>}
                </button>
              ))}
            </div>
          </div>

          {/* prompt picker */}
          <div>
            <div className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-2">Sentence to complete</div>
            <div className="space-y-1">
              {(PROMPTS[concept] ?? []).map(p => (
                <button key={p}
                  onClick={() => { setPrompt(p); setResult(null); }}
                  className="w-full text-left px-3 py-1.5 rounded-lg border text-xs font-mono transition-all"
                  style={{
                    borderColor: p === prompt ? info.color + "60" : "rgba(255,255,255,0.06)",
                    color:       p === prompt ? "rgba(220,220,232,0.9)" : "rgba(160,160,175,0.6)",
                    background:  p === prompt ? info.color + "10" : "transparent",
                  }}>
                  {p}
                </button>
              ))}
            </div>
          </div>

          {/* alpha */}
          <div>
            <div className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-2">Erasure strength (α={alpha})</div>
            <div className="flex gap-2">
              {[8, 10, 15, 20].map(a => (
                <button key={a}
                  onClick={() => setAlpha(a)}
                  className="flex-1 text-xs py-1.5 rounded-lg border transition-all"
                  style={{
                    borderColor: a === alpha ? info.color : "rgba(255,255,255,0.08)",
                    color:       a === alpha ? info.color : "rgba(160,160,175,0.6)",
                    background:  a === alpha ? info.color + "18" : "transparent",
                  }}>
                  {a}
                </button>
              ))}
            </div>
          </div>

          <button onClick={run} disabled={loading}
            className="w-full py-2.5 rounded-xl text-sm font-mono font-semibold border transition-all disabled:opacity-40"
            style={{ borderColor: info.color, background: info.color + "18", color: info.color }}>
            {loading ? "Running…" : "Run erasure experiment"}
          </button>
        </div>

        {/* results */}
        <div className="space-y-3">
          {/* completions */}
          <div className="rounded-xl border border-border bg-card p-4">
            <div className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-3">Completions</div>
            <div className="space-y-3">
              <div>
                <div className="text-xs text-muted-foreground/50 mb-1">Baseline — no intervention</div>
                {result ? (
                  <p className="text-sm leading-relaxed">
                    <span className="text-muted-foreground/50">{prompt} </span>
                    <span className="text-foreground">{result.original_completion}</span>
                  </p>
                ) : loading ? (
                  <div className="h-8 flex items-center gap-2 text-xs text-muted-foreground/40">
                    <span className="w-1.5 h-1.5 rounded-full bg-white/30 animate-pulse" /> generating…
                  </div>
                ) : <p className="text-xs text-muted-foreground/30 italic">—</p>}
              </div>
              <div className="border-t border-border/50 pt-3">
                <div className="text-xs mb-1" style={{ color: info.color + "99" }}>
                  With {info.label} direction erased (α={alpha})
                </div>
                {result ? (
                  <p className="text-sm leading-relaxed">
                    <span className="text-muted-foreground/50">{prompt} </span>
                    <span style={{ color: "rgba(220,240,220,0.95)" }}>{result.debiased_completion}</span>
                  </p>
                ) : loading ? (
                  <div className="h-8 flex items-center gap-2 text-xs" style={{ color: info.color + "60" }}>
                    <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: info.color }} /> erasing bias direction…
                  </div>
                ) : <p className="text-xs text-muted-foreground/30 italic">—</p>}
              </div>
            </div>
          </div>

          {/* probe scores */}
          {result && (
            <div className="rounded-xl border border-border bg-card p-4">
              <ScoreDelta
                concept={concept}
                original={result.original_scores}
                debiased={result.debiased_scores}
              />
              <div className="mt-3 text-xs text-muted-foreground/40 font-mono">
                {(result.latency_ms/1000).toFixed(1)}s total
              </div>
            </div>
          )}

          {error && <div className="text-xs font-mono text-red-400 bg-red-500/10 border border-red-500/30 rounded-xl p-3">{error}</div>}

          {!result && !loading && (
            <div className="rounded-xl border border-dashed border-border p-4 text-xs text-muted-foreground/40 leading-relaxed">
              Run an experiment to see side-by-side completions and the change in bias probe scores.
              The target probe score should drop; unrelated concepts should stay roughly constant.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
