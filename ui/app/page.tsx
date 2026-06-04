"use client";

import { useState, useEffect } from "react";
import StyleInput from "@/components/StyleInput";
import StyleBlend from "@/components/StyleBlend";
import ExplainTab from "@/components/ExplainTab";
import ConceptScanner from "@/components/ConceptScanner";
import MeasureTab from "@/components/MeasureTab";
import BiasLab from "@/components/BiasLab";
import BiasCompletions from "@/components/BiasCompletions";
import HowItWorks from "@/components/HowItWorks";

const TABS = [
  { key: "how-it-works", label: "How It Works", sub: "Architecture & concepts" },
  { key: "discover",     label: "Discover",     sub: "Style steering" },
  { key: "explain",      label: "Explain",      sub: "The geometry" },
  { key: "bias",         label: "Bias",         sub: "Erasure experiments" },
  { key: "measure",      label: "Measure",      sub: "Benchmark results" },
] as const;
type Tab = typeof TABS[number]["key"];

const FINDINGS = [
  { value: "17×",      label: "more geometric structure",      sub: "style directions vs random vectors" },
  { value: "−24 pp",   label: "stereotype rate reduction",     sub: "age/competence · CrowS-Pairs · α=20" },
  { value: "+0.58",    label: "cosine similarity",             sub: "legal ↔ scientific (top aligned pair)" },
  { value: "2 models", label: "same geometric structure",      sub: "Llama 3B and Mistral 7B independently" },
];

function Section({ eyebrow, title, desc, children }: {
  eyebrow: string; title: string; desc?: string; children: React.ReactNode;
}) {
  return (
    <div className="pt-10 space-y-6">
      <div>
        <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-muted-foreground mb-3">
          <span className="w-4 h-px bg-primary/60" />
          {eyebrow}
        </div>
        <h2 className="text-3xl font-bold tracking-tight mb-2">{title}</h2>
        {desc && <p className="text-sm text-muted-foreground max-w-2xl leading-relaxed">{desc}</p>}
      </div>
      {children}
    </div>
  );
}

function LiveBadge() {
  const [state, setState] = useState<"checking" | "online" | "offline">("checking");
  useEffect(() => {
    const check = () =>
      fetch("/api/steer").then(r => setState(r.ok ? "online" : "offline")).catch(() => setState("offline"));
    check();
    const id = setInterval(check, 10000);
    return () => clearInterval(id);
  }, []);
  if (state === "checking") return null;
  return (
    <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-mono"
      style={{
        borderColor: state === "online" ? "rgba(52,211,153,0.4)" : "rgba(248,113,113,0.3)",
        background:  state === "online" ? "rgba(52,211,153,0.08)" : "rgba(248,113,113,0.06)",
        color:       state === "online" ? "rgb(52,211,153)" : "rgb(248,113,113)",
      }}>
      <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${state === "online" ? "bg-emerald-400 animate-pulse" : "bg-red-400"}`} />
      {state === "online" ? "LIVE inference" : "server offline"}
    </div>
  );
}

export default function Home() {
  const [tab, setTab] = useState<Tab>("discover");

  return (
    <main className="min-h-screen">
      {/* nav */}
      <nav className="fixed top-0 inset-x-0 z-50 border-b border-border/60 backdrop-blur-md bg-background/80">
        <div className="max-w-5xl mx-auto px-6 h-14 flex items-center justify-between">
          <span className="font-mono text-sm font-semibold tracking-tight">
            <span className="gradient-text">Neural</span>Style
          </span>
          <div className="flex items-center gap-1.5">
            {TABS.map(t => (
              <button key={t.key} onClick={() => setTab(t.key)}
                className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  tab === t.key ? "bg-primary/15 text-foreground" : "text-muted-foreground hover:text-foreground"
                }`}>
                {t.label}
              </button>
            ))}
            <div className="ml-3">
              <LiveBadge />
            </div>
          </div>
        </div>
      </nav>

      {/* hero */}
      <section className="hero-bg pt-32 pb-12 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-muted-foreground mb-5 border border-border px-3 py-1.5 rounded-full fade-up fade-up-1">
            <span className="w-1.5 h-1.5 rounded-full bg-primary pulse-dot" />
            CS 153 · Stanford · 2026
          </div>
          <h1 className="text-5xl font-bold tracking-tight leading-[1.1] mb-4 fade-up fade-up-2">
            Steer what a language model<br />
            <span className="gradient-text">thinks about</span>
          </h1>
          <p className="text-base text-muted-foreground max-w-2xl mb-3 leading-relaxed fade-up fade-up-3">
            We add a learned direction vector to a model&apos;s residual stream at inference time —
            no fine-tuning, no system prompt — and measure how concepts like writing style,
            demographic bias, and factual knowledge respond.
          </p>
          <p className="text-sm text-muted-foreground max-w-2xl mb-8 leading-relaxed fade-up fade-up-3">
            The central finding: stylistic concept directions occupy a geometrically structured
            subspace (17× more structured than random vectors), while bias directions are near-random.
            This geometric property predicts which interventions are controllable and which are not.
          </p>

          {/* key findings bar */}
          <div className="grid grid-cols-4 gap-3 mb-8 fade-up fade-up-3">
            {FINDINGS.map(f => (
              <div key={f.value} className="rounded-xl border border-border bg-card/60 px-4 py-3">
                <div className="text-2xl font-bold tracking-tight gradient-text mb-1">{f.value}</div>
                <div className="text-xs font-semibold text-foreground/80 leading-tight mb-0.5">{f.label}</div>
                <div className="text-xs text-muted-foreground/60 leading-tight">{f.sub}</div>
              </div>
            ))}
          </div>

          <div className="flex gap-3 fade-up fade-up-3">
            {TABS.map(t => (
              <button key={t.key} onClick={() => setTab(t.key)}
                className={`group flex flex-col items-start px-5 py-3 rounded-xl border text-left transition-all ${
                  tab === t.key
                    ? "border-primary/40 bg-primary/8 text-foreground"
                    : "border-border text-muted-foreground hover:text-foreground hover:border-border/60"
                }`}>
                <span className="text-sm font-semibold">{t.label}</span>
                <span className="text-xs text-muted-foreground mt-0.5">{t.sub}</span>
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* tab content */}
      <div className="max-w-5xl mx-auto px-6 pb-24">

        {tab === "how-it-works" && <HowItWorks />}

        {tab === "discover" && (
          <>
            <Section eyebrow="Style steering" title="Pick a style. See it in action."
              desc="A learned direction vector is added to the residual stream at inference time. Pick a style — the live demo and precomputed reference examples below both update. No fine-tuning. No system prompt.">
              <div className="rounded-2xl border border-border bg-card p-6">
                <StyleInput />
              </div>
            </Section>

            <Section eyebrow="Vector arithmetic" title="Blend two concept directions"
              desc="h′ = h + α₁·v̂_A + α₂·v̂_B. Opposing directions fight; aligned directions reinforce. The cosine similarity between concept directions predicts the output before you run it.">
              <div className="rounded-2xl border border-border bg-card p-6">
                <StyleBlend />
              </div>
            </Section>
          </>
        )}

        {tab === "explain" && (
          <>
            <ExplainTab />
            <Section eyebrow="Concept scanner" title="What does the model see in your text?"
              desc="A forward pass computes the cosine similarity between the last-token hidden state and each of the 21 probe directions. The bars show how strongly each concept is activated.">
              <div className="rounded-2xl border border-border bg-card p-6">
                <ConceptScanner />
              </div>
            </Section>
          </>
        )}

        {tab === "bias" && (
          <>
            <Section eyebrow="Before and after" title="What the model actually says"
              desc="The same prompt, completed twice — once normally, once with the bias direction subtracted from the residual stream. An independent LLM judge labels each completion as stereotyped or counter-stereotyped.">
              <div className="rounded-2xl border border-border bg-card p-6">
                <BiasCompletions />
              </div>
            </Section>

            <Section eyebrow="Live experiment" title="Measure the log-probability gap"
              desc="The same protocol as CrowS-Pairs: compare the model's log-probability on a stereotyped sentence vs its counter-stereotype before and after erasure. Run live on the current model.">
              <div className="rounded-2xl border border-border bg-card p-6">
                <BiasLab />
              </div>
            </Section>

            <MeasureTab />
          </>
        )}

        {tab === "measure" && <MeasureTab />}

      </div>

      <footer className="border-t border-border py-6 px-6">
        <div className="max-w-5xl mx-auto flex items-center justify-between text-xs text-muted-foreground font-mono">
          <span>NeuralStyle · CS 153 · Stanford 2026</span>
          <span>RepE-style contrast direction steering · Llama 3.2-3B · layers 23–26</span>
        </div>
      </footer>
    </main>
  );
}
