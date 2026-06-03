"use client";

import { useEffect, useState } from "react";

const STYLES = [
  { key: "hemingway",         label: "Hemingway",   color: "oklch(0.72 0.18 25)",   desc: "Sparse. Declarative. Iceberg." },
  { key: "shakespeare",       label: "Shakespeare", color: "oklch(0.72 0.18 290)",  desc: "Iambic. Ornate. Archaic." },
  { key: "legal_text",        label: "Legal",       color: "oklch(0.72 0.15 200)",  desc: "Formal. Precise. Conditional." },
  { key: "scientific_writing",label: "Scientific",  color: "oklch(0.72 0.18 145)",  desc: "Passive. Hedged. Evidence-driven." },
  { key: "news_wire",         label: "News Wire",   color: "oklch(0.72 0.15 60)",   desc: "Inverted pyramid. Direct." },
] as const;

type StyleKey = typeof STYLES[number]["key"];

type CompletionEntry = { prompt: string; completion: string; seed: number };
type CompletionData = Record<string, Record<StyleKey, CompletionEntry[]>>;

const ALPHA_LABELS: Record<string, string> = {
  "0.0": "Baseline",
  "5.0": "Subtle (α=5)",
  "10.0": "Moderate (α=10)",
  "20.0": "Strong (α=20)",
  "40.0": "Intense (α=40)",
  "60.0": "Maximum (α=60)",
};

export default function StyleLab() {
  const [data, setData] = useState<CompletionData | null>(null);
  const [selectedStyle, setSelectedStyle] = useState<StyleKey>("hemingway");
  const [selectedAlpha, setSelectedAlpha] = useState("10.0");
  const [selectedIdx, setSelectedIdx] = useState(0);

  useEffect(() => {
    fetch("/data/style_inject_completions.json")
      .then(r => r.json())
      .then(setData)
      .catch(() => null);
  }, []);

  const styleInfo = STYLES.find(s => s.key === selectedStyle)!;
  const alphaKeys = data ? Object.keys(data).sort((a, b) => Number(a) - Number(b)) : [];
  const examples: CompletionEntry[] = data?.[selectedAlpha]?.[selectedStyle] ?? [];
  const baseline: CompletionEntry[] = data?.["0.0"]?.[selectedStyle] ?? [];

  const currentExample = examples.find((e, i) => i === selectedIdx) ?? examples[0];
  const baselineForPrompt = baseline.find(e => e.prompt === currentExample?.prompt) ?? baseline[0];

  const prompts = Array.from(new Set(baseline.map(e => e.prompt)));

  return (
    <section className="py-24 px-6" id="lab">
      <div className="max-w-6xl mx-auto">
        {/* header */}
        <div className="mb-12 fade-up fade-up-1">
          <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-muted-foreground mb-4">
            <span className="w-4 h-px bg-primary/60" />
            Style Lab
          </div>
          <h2 className="text-4xl font-bold tracking-tight mb-3">
            Steer any text toward a style
          </h2>
          <p className="text-muted-foreground max-w-xl">
            We add a learned <span className="text-foreground font-medium">style vector</span> directly
            to the model&apos;s residual stream at inference time — no fine-tuning, no prompting.
            The same neutral sentence transforms based on which direction we push the hidden states.
          </p>
        </div>

        {/* style picker */}
        <div className="flex gap-3 flex-wrap mb-8 fade-up fade-up-2">
          {STYLES.map(s => (
            <button
              key={s.key}
              onClick={() => { setSelectedStyle(s.key); setSelectedIdx(0); }}
              className={`group relative px-5 py-3 rounded-xl border text-sm font-medium transition-all duration-200 ${
                selectedStyle === s.key
                  ? "border-transparent text-background"
                  : "border-border text-muted-foreground hover:text-foreground hover:border-border/60"
              }`}
              style={selectedStyle === s.key ? { background: s.color } : {}}
            >
              <span>{s.label}</span>
              <span className={`block text-xs font-normal mt-0.5 transition-opacity ${
                selectedStyle === s.key ? "opacity-80" : "opacity-0 group-hover:opacity-60"
              }`}>
                {s.desc}
              </span>
            </button>
          ))}
        </div>

        {/* strength selector */}
        <div className="flex gap-2 flex-wrap mb-10 fade-up fade-up-2">
          <span className="text-xs text-muted-foreground self-center mr-1 font-mono">Steering strength:</span>
          {alphaKeys.map(key => (
            <button
              key={key}
              onClick={() => setSelectedAlpha(key)}
              className={`px-3 py-1 rounded-lg text-xs font-mono border transition-all ${
                selectedAlpha === key
                  ? "border-primary/60 text-foreground bg-primary/10"
                  : "border-border text-muted-foreground hover:text-foreground"
              }`}
            >
              {ALPHA_LABELS[key] ?? `α=${key}`}
            </button>
          ))}
        </div>

        {!data && (
          <div className="rounded-2xl border border-border bg-card p-12 text-center fade-up fade-up-3">
            <div className="w-2 h-2 rounded-full bg-primary/60 mx-auto mb-4 pulse-dot" />
            <p className="text-muted-foreground text-sm">
              Style injection results loading… (job running on cluster)
            </p>
          </div>
        )}

        {data && examples.length === 0 && (
          <div className="rounded-2xl border border-border bg-card p-12 text-center fade-up fade-up-3">
            <p className="text-muted-foreground text-sm">No data for this combination yet.</p>
          </div>
        )}

        {data && examples.length > 0 && (
          <div className="fade-up fade-up-3">
            {/* prompt picker */}
            <div className="flex gap-2 flex-wrap mb-6">
              <span className="text-xs text-muted-foreground self-center mr-1 font-mono">Prompt:</span>
              {prompts.map((p, i) => (
                <button
                  key={i}
                  onClick={() => setSelectedIdx(i)}
                  className={`px-3 py-1 rounded-lg text-xs border transition-all max-w-xs truncate ${
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
                <div className="flex items-center gap-2 mb-4">
                  <span className="text-xs font-mono uppercase tracking-widest text-muted-foreground">Baseline (α=0)</span>
                </div>
                <p className="text-sm font-mono text-muted-foreground mb-3 leading-relaxed">
                  &ldquo;{baselineForPrompt?.prompt}&rdquo;
                </p>
                <p className="text-base leading-relaxed">
                  {baselineForPrompt?.completion ?? "—"}
                </p>
              </div>

              <div
                className="rounded-2xl border bg-card p-6 glow-border"
                style={{ borderColor: `${styleInfo.color}40` }}
              >
                <div className="flex items-center gap-2 mb-4">
                  <span
                    className="text-xs font-mono uppercase tracking-widest"
                    style={{ color: styleInfo.color }}
                  >
                    {styleInfo.label} · {ALPHA_LABELS[selectedAlpha] ?? `α=${selectedAlpha}`}
                  </span>
                  <span
                    className="ml-auto text-xs px-2 py-0.5 rounded-full font-mono"
                    style={{ background: `${styleInfo.color}20`, color: styleInfo.color }}
                  >
                    steered
                  </span>
                </div>
                <p className="text-sm font-mono text-muted-foreground mb-3 leading-relaxed">
                  &ldquo;{currentExample?.prompt}&rdquo;
                </p>
                <p className="text-base leading-relaxed">
                  {currentExample?.completion ?? "—"}
                </p>
              </div>
            </div>

            <p className="text-xs text-muted-foreground mt-4 text-center font-mono">
              Layers 23–26 steered simultaneously · Llama 3.2 3B · no fine-tuning
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
