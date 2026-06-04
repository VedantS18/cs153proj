"use client";
import { useState, useCallback } from "react";

const ALL_CONCEPTS = [
  { key: "hemingway",          label: "Hemingway",       color: "oklch(0.72 0.18 25)"  },
  { key: "shakespeare",        label: "Shakespeare",      color: "oklch(0.72 0.18 290)" },
  { key: "fitzgerald",         label: "Fitzgerald",       color: "oklch(0.74 0.17 265)" },
  { key: "austen",             label: "Austen",           color: "oklch(0.74 0.16 160)" },
  { key: "dickens",            label: "Dickens",          color: "oklch(0.68 0.15 50)"  },
  { key: "woolf",              label: "Woolf",            color: "oklch(0.72 0.17 220)" },
  { key: "jk_rowling",         label: "J.K. Rowling",     color: "oklch(0.72 0.20 310)" },
  { key: "cormac_mccarthy",    label: "McCarthy",         color: "oklch(0.65 0.15 35)"  },
  { key: "legal_text",         label: "Legal",            color: "oklch(0.72 0.15 200)" },
  { key: "scientific_writing", label: "Scientific",       color: "oklch(0.72 0.18 145)" },
  { key: "news_wire",          label: "News Wire",        color: "oklch(0.72 0.15 60)"  },
] as const;

// Pre-load interesting pairs with expected interaction
const PRESETS = [
  { a: "hemingway",          b: "legal_text",         label: "Hemingway × Legal",      note: "opposing directions (cos −0.31) — they fight each other" },
  { a: "legal_text",         b: "scientific_writing", label: "Legal × Scientific",      note: "similar directions (cos +0.58) — they reinforce each other" },
  { a: "shakespeare",        b: "news_wire",          label: "Shakespeare × News Wire", note: "opposing directions (cos −0.38)" },
  { a: "hemingway",          b: "shakespeare",        label: "Hemingway × Shakespeare", note: "sparse vs ornate — clear tension" },
  { a: "fitzgerald",         label: "Fitzgerald solo", note: "lyrical, Jazz Age" },
] as const;

const SUGGESTIONS = [
  "The old man had been fishing for three days without catching anything.",
  "She walked into the room and sat down at the table.",
  "The committee voted on the proposed changes to the policy.",
];

type Result = { steered: string; concepts: string[]; alphas: number[]; latency_ms: number };

export default function StyleBlend() {
  const [text,    setText]    = useState(SUGGESTIONS[0]);
  const [conceptA, setConceptA] = useState("hemingway");
  const [conceptB, setConceptB] = useState("legal_text");
  const [alphaA,  setAlphaA]  = useState(3);
  const [alphaB,  setAlphaB]  = useState(3);
  const [result,  setResult]  = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  const infoA = ALL_CONCEPTS.find(c => c.key === conceptA)!;
  const infoB = ALL_CONCEPTS.find(c => c.key === conceptB)!;

  const run = useCallback(async () => {
    if (!text.trim() || loading) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await fetch("/api/steer/blend", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text.trim(),
          concepts: [conceptA, conceptB],
          alphas:   [alphaA,   alphaB],
        }),
      });
      const data = await res.json();
      if (!res.ok) setError(data.error ?? "Request failed");
      else setResult(data as Result);
    } catch (e: unknown) { setError((e as Error).message); }
    finally { setLoading(false); }
  }, [text, conceptA, conceptB, alphaA, alphaB, loading]);

  return (
    <div className="space-y-5">
      <div className="text-xs text-muted-foreground leading-relaxed max-w-2xl">
        Two concept vectors are added to the residual stream simultaneously:
        {" "}<code className="font-mono text-primary/80">h&apos; = h + α₁·v̂_A + α₂·v̂_B</code>.
        When the directions oppose each other (e.g. Hemingway vs Legal, cos = −0.31) they
        fight — the result is incoherent or lands somewhere in between. When they align
        (Legal + Scientific, cos = +0.58) they reinforce. This is concept arithmetic in activation space.
      </div>

      {/* presets */}
      <div className="flex flex-wrap gap-2">
        {PRESETS.map(p => (
          <button key={p.label}
            onClick={() => {
              if ("a" in p && "b" in p) { setConceptA(p.a); setConceptB(p.b); }
            }}
            className="text-xs px-3 py-1 rounded-full border border-border bg-card hover:bg-primary/10 text-muted-foreground hover:text-foreground transition-all"
            title={p.note}
          >
            {p.label}
          </button>
        ))}
      </div>

      <div className="grid gap-4" style={{ gridTemplateColumns: "1fr 1fr" }}>
        {/* left: controls */}
        <div className="space-y-4">
          <textarea
            value={text}
            onChange={e => setText(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run(); }}
            rows={3}
            className="w-full rounded-xl border border-border bg-card/60 px-4 py-3 text-sm font-mono resize-none focus:outline-none focus:ring-1 focus:ring-primary/50"
          />
          <div className="flex flex-wrap gap-1.5">
            {SUGGESTIONS.map(s => (
              <button key={s} onClick={() => setText(s)}
                className="text-xs px-2 py-0.5 rounded-full border border-border bg-card text-muted-foreground/70">
                {s.length > 40 ? s.slice(0, 40) + "…" : s}
              </button>
            ))}
          </div>

          {/* two concept pickers */}
          {([["A", conceptA, setConceptA, alphaA, setAlphaA, infoA],
             ["B", conceptB, setConceptB, alphaB, setAlphaB, infoB]] as const).map(
            ([label, selected, setSelected, alpha, setAlpha, info]) => (
              <div key={label} className="rounded-xl border p-3 space-y-2"
                style={{ borderColor: (info as typeof infoA).color + "40" }}>
                <div className="text-xs font-mono" style={{ color: (info as typeof infoA).color }}>
                  Vector {label}: <strong>{(info as typeof infoA).label}</strong> · α={alpha}
                </div>
                <div className="flex flex-wrap gap-1">
                  {ALL_CONCEPTS.map(c => (
                    <button key={c.key}
                      onClick={() => (setSelected as (v: string) => void)(c.key)}
                      className="text-xs px-2 py-0.5 rounded border transition-all"
                      style={{
                        borderColor: c.key === selected ? c.color : "rgba(255,255,255,0.08)",
                        background:  c.key === selected ? c.color + "22" : "transparent",
                        color:       c.key === selected ? c.color : "rgba(160,160,175,0.6)",
                      }}
                    >{c.label}</button>
                  ))}
                </div>
                <div className="flex gap-2 items-center">
                  <span className="text-xs text-muted-foreground/50 font-mono w-4">α</span>
                  {[3, 5, 8, 12].map(a => (
                    <button key={a}
                      onClick={() => (setAlpha as (v: number) => void)(a)}
                      className="text-xs px-2 py-0.5 rounded border transition-all"
                      style={{
                        borderColor: a === alpha ? (info as typeof infoA).color : "rgba(255,255,255,0.08)",
                        color:       a === alpha ? (info as typeof infoA).color : "rgba(160,160,175,0.5)",
                      }}
                    >{a}</button>
                  ))}
                </div>
              </div>
            )
          )}

          <button onClick={run} disabled={!text.trim() || loading}
            className="w-full py-2.5 rounded-xl text-sm font-mono font-semibold border transition-all disabled:opacity-40"
            style={{ borderColor: infoA.color, background: infoA.color + "18", color: infoA.color }}>
            {loading ? "Generating…" : `Blend ${infoA.label} + ${infoB.label}  (⌘↵)`}
          </button>
        </div>

        {/* right: output */}
        <div className="space-y-3">
          <div className="rounded-xl border border-border bg-card p-4" style={{ minHeight: 90 }}>
            <div className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-2">Your text</div>
            <p className="text-sm leading-relaxed text-muted-foreground">{text || "—"}</p>
          </div>

          <div className="rounded-xl border p-4" style={{
            minHeight: 120,
            borderColor: infoA.color + "40",
            background: `linear-gradient(135deg, ${infoA.color}08, ${infoB.color}08)`,
          }}>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs font-mono" style={{ color: infoA.color }}>{infoA.label}</span>
              <span className="text-xs text-muted-foreground/40">+</span>
              <span className="text-xs font-mono" style={{ color: infoB.color }}>{infoB.label}</span>
              {result && <span className="ml-auto text-xs font-mono text-muted-foreground/40">{(result.latency_ms/1000).toFixed(1)}s</span>}
            </div>
            {result ? (
              <p className="text-sm leading-relaxed">
                <span className="text-muted-foreground/50">{text.trim()} </span>
                <span>{result.steered}</span>
              </p>
            ) : loading ? (
              <div className="flex items-center gap-2 text-xs text-muted-foreground/50">
                <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: infoA.color }} />
                Summing concept vectors and generating…
              </div>
            ) : (
              <p className="text-xs text-muted-foreground/30 italic">Output appears here</p>
            )}
          </div>

          {error && <div className="text-xs font-mono text-red-400 bg-red-500/10 border border-red-500/30 rounded-xl p-3">{error}</div>}

          <div className="rounded-xl border border-border bg-card/40 p-3 text-xs text-muted-foreground/60 leading-relaxed">
            The geometry predicts the result: when |cos| is high between two directions, they reinforce.
            When it is near zero or negative, they interfere. The network graph in the Explain tab shows exactly which pairs to expect conflict from.
          </div>
        </div>
      </div>
    </div>
  );
}
