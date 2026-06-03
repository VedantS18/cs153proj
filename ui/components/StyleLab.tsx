"use client";

import { useEffect, useState } from "react";

const STYLES = [
  { key: "hemingway",          label: "Hemingway",        color: "oklch(0.72 0.18 25)",   desc: "Sparse. Declarative. Iceberg." },
  { key: "shakespeare",        label: "Shakespeare",       color: "oklch(0.72 0.18 290)",  desc: "Iambic. Ornate. Archaic." },
  { key: "jk_rowling",         label: "J.K. Rowling",      color: "oklch(0.72 0.20 310)",  desc: "Whimsical. Invented detail. Magical." },
  { key: "cormac_mccarthy",    label: "Cormac McCarthy",   color: "oklch(0.65 0.15 35)",   desc: "Bleak. Spare. No punctuation." },
  { key: "legal_text",         label: "Legal",             color: "oklch(0.72 0.15 200)",  desc: "Formal. Precise. Conditional." },
  { key: "scientific_writing", label: "Scientific",        color: "oklch(0.72 0.18 145)",  desc: "Passive. Hedged. Evidence-driven." },
  { key: "news_wire",          label: "News Wire",         color: "oklch(0.72 0.15 60)",   desc: "Inverted pyramid. Direct." },
] as const;

type StyleKey = typeof STYLES[number]["key"];

const ALPHA_LABELS: Record<string, string> = {
  "0.0":  "None (α=0)",
  "5.0":  "Subtle (α=5)",
  "10.0": "Moderate (α=10)",
  "20.0": "Strong (α=20) ⚡",
  "40.0": "Collapse (α=40) ☠",
};

const DANGER_ALPHAS  = new Set(["20.0", "40.0"]);
const GIBBERISH_ALPHAS = new Set(["40.0"]);

type CompletionEntry = { prompt: string; completion: string; seed?: number };
type CompletionData  = Record<string, Record<string, CompletionEntry[]>>;

export default function StyleLab() {
  const [data,          setData]          = useState<CompletionData | null>(null);
  const [selectedStyle, setSelectedStyle] = useState<StyleKey>("hemingway");
  const [selectedAlpha, setSelectedAlpha] = useState("5.0");
  const [selectedIdx,   setSelectedIdx]   = useState(0);

  useEffect(() => {
    fetch("/data/style_inject_completions.json")
      .then(r => r.json()).then(setData).catch(() => null);
  }, []);

  const styleInfo  = STYLES.find(s => s.key === selectedStyle)!;
  const alphaKeys  = data ? Object.keys(data).sort((a, b) => Number(a) - Number(b)) : [];
  const examples: CompletionEntry[] = data?.[selectedAlpha]?.[selectedStyle] ?? [];
  const baseline: CompletionEntry[] = data?.["0.0"]?.[selectedStyle]         ?? [];

  // Deduplicate prompts (3 seeds per prompt)
  const prompts = Array.from(new Set(baseline.map(e => e.prompt)));
  const currentExample      = examples[selectedIdx]      ?? examples[0];
  const baselineForPrompt   = baseline.find(e => e.prompt === currentExample?.prompt) ?? baseline[0];

  const isGibberish = GIBBERISH_ALPHAS.has(selectedAlpha);
  const isDanger    = DANGER_ALPHAS.has(selectedAlpha);

  return (
    <div className="pt-10">
      {/* section header */}
      <div className="mb-8">
        <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-muted-foreground mb-3">
          <span className="w-4 h-px bg-primary/60" />
          Style Lab
        </div>
        <h2 className="text-3xl font-bold tracking-tight mb-2">Steer any text toward a style</h2>
        <p className="text-muted-foreground text-sm max-w-xl">
          A learned style vector is added to the model&apos;s residual stream at inference time.
          No fine-tuning. No system prompt. Pick a style, pick a strength, watch it happen.
        </p>
      </div>

      {/* style picker */}
      <div className="flex gap-2 flex-wrap mb-6">
        {STYLES.map(s => {
          // Only show styles that have data loaded
          const hasData = !data || data[selectedAlpha]?.[s.key]?.length > 0
                        || data["0.0"]?.[s.key]?.length > 0;
          if (!hasData && data) return null;
          return (
            <button
              key={s.key}
              onClick={() => { setSelectedStyle(s.key); setSelectedIdx(0); }}
              disabled={!!(data && !data["0.0"]?.[s.key]?.length)}
              className={`group relative px-4 py-2.5 rounded-xl border text-sm font-medium transition-all duration-200 ${
                selectedStyle === s.key
                  ? "border-transparent text-background"
                  : "border-border text-muted-foreground hover:text-foreground hover:border-border/60 disabled:opacity-30 disabled:cursor-not-allowed"
              }`}
              style={selectedStyle === s.key ? { background: s.color } : {}}
            >
              <span>{s.label}</span>
              <span className={`block text-xs font-normal mt-0.5 transition-opacity ${
                selectedStyle === s.key ? "opacity-75" : "opacity-0 group-hover:opacity-50"
              }`}>{s.desc}</span>
            </button>
          );
        })}
      </div>

      {/* strength selector */}
      <div className="flex items-center gap-2 flex-wrap mb-5">
        <span className="text-xs text-muted-foreground font-mono">Strength:</span>
        {alphaKeys.filter(k => ALPHA_LABELS[k]).map(key => (
          <button
            key={key}
            onClick={() => setSelectedAlpha(key)}
            className={`px-3 py-1 rounded-lg text-xs font-mono border transition-all ${
              selectedAlpha === key
                ? isDanger
                  ? "border-orange-500/60 text-orange-300 bg-orange-500/10"
                  : "border-primary/60 text-foreground bg-primary/10"
                : DANGER_ALPHAS.has(key)
                  ? "border-orange-500/25 text-orange-500/60 hover:text-orange-400"
                  : "border-border text-muted-foreground hover:text-foreground"
            }`}
          >
            {ALPHA_LABELS[key]}
          </button>
        ))}
      </div>

      {/* danger banner */}
      {isDanger && (
        <div className={`rounded-xl border p-3 mb-5 text-xs leading-relaxed ${
          isGibberish
            ? "border-red-500/30 bg-red-500/5 text-red-400"
            : "border-orange-500/30 bg-orange-500/5 text-orange-400"
        }`}>
          {isGibberish
            ? "☠  The model is unraveling. Style vectors have cos(probe, contrast) ≈ 1.0 — they are load-bearing in the residual stream. Push too hard and coherence collapses. This is evidence the vectors are real, not just correlational noise."
            : "⚡  Approaching the coherence cliff. Style is starting to overwrite meaning — one notch further and the text breaks down."}
        </div>
      )}

      {!data && (
        <div className="rounded-2xl border border-border bg-card p-16 text-center">
          <div className="w-2 h-2 rounded-full bg-primary/60 mx-auto mb-4 pulse-dot" />
          <p className="text-muted-foreground text-sm">Loading completions…</p>
        </div>
      )}

      {data && examples.length === 0 && (
        <div className="rounded-2xl border border-border bg-card p-16 text-center">
          <p className="text-muted-foreground text-sm">No data for this style yet — job may still be running.</p>
        </div>
      )}

      {data && examples.length > 0 && (
        <>
          {/* prompt picker */}
          <div className="flex items-center gap-2 flex-wrap mb-5">
            <span className="text-xs text-muted-foreground font-mono">Prompt:</span>
            {prompts.map((p, i) => (
              <button
                key={i}
                onClick={() => setSelectedIdx(i)}
                className={`px-3 py-1 rounded-lg text-xs border transition-all max-w-[220px] truncate ${
                  selectedIdx === i
                    ? "border-primary/60 text-foreground bg-primary/10"
                    : "border-border text-muted-foreground hover:text-foreground"
                }`}
              >
                &ldquo;{p}&rdquo;
              </button>
            ))}
          </div>

          {/* before / after */}
          <div className="grid grid-cols-2 gap-4">
            <div className="rounded-2xl border border-border bg-card p-6">
              <div className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-4">
                Baseline — no steering
              </div>
              <p className="text-xs font-mono text-muted-foreground mb-3">
                &ldquo;{baselineForPrompt?.prompt}&rdquo;
              </p>
              <p className="text-sm leading-relaxed">{baselineForPrompt?.completion ?? "—"}</p>
            </div>

            <div
              className="rounded-2xl border bg-card p-6"
              style={{
                borderColor: isGibberish ? "rgb(239 68 68 / 40%)"
                           : isDanger    ? "rgb(249 115 22 / 40%)"
                           : `${styleInfo.color}40`,
              }}
            >
              <div className="flex items-center gap-2 mb-4">
                <span className="text-xs font-mono uppercase tracking-widest"
                      style={{ color: isGibberish ? "#f87171" : isDanger ? "#fb923c" : styleInfo.color }}>
                  {styleInfo.label} · {ALPHA_LABELS[selectedAlpha] ?? `α=${selectedAlpha}`}
                </span>
                <span className="ml-auto text-xs px-2 py-0.5 rounded-full font-mono"
                      style={{
                        background: isGibberish ? "rgb(239 68 68 / 15%)"
                                  : isDanger    ? "rgb(249 115 22 / 15%)"
                                  : `${styleInfo.color}20`,
                        color: isGibberish ? "#f87171" : isDanger ? "#fb923c" : styleInfo.color,
                      }}>
                  {isGibberish ? "collapsed" : isDanger ? "unstable" : "steered"}
                </span>
              </div>
              <p className="text-xs font-mono text-muted-foreground mb-3">
                &ldquo;{currentExample?.prompt}&rdquo;
              </p>
              <p className={`text-sm leading-relaxed ${isGibberish ? "text-red-400/80 font-mono" : ""}`}>
                {currentExample?.completion ?? "—"}
              </p>
            </div>
          </div>

          <p className="text-xs text-muted-foreground mt-3 font-mono text-center">
            Layers 23–26 steered simultaneously · Llama 3.2 3B · no fine-tuning
          </p>
        </>
      )}
    </div>
  );
}
