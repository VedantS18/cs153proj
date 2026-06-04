"use client";

import { useState } from "react";
import StyleLab from "@/components/StyleLab";
import StyleInput from "@/components/StyleInput";
import StyleBlend from "@/components/StyleBlend";
import ExplainTab from "@/components/ExplainTab";
import ConceptScanner from "@/components/ConceptScanner";
import MeasureTab from "@/components/MeasureTab";
import BiasLab from "@/components/BiasLab";

const TABS = [
  { key: "discover", label: "Discover", sub: "Style steering" },
  { key: "explain",  label: "Explain",  sub: "The geometry" },
  { key: "bias",     label: "Bias",     sub: "Erasure experiments" },
  { key: "measure",  label: "Measure",  sub: "Benchmark results" },
] as const;
type Tab = typeof TABS[number]["key"];

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
          <div className="flex items-center gap-1">
            {TABS.map(t => (
              <button key={t.key} onClick={() => setTab(t.key)}
                className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  tab === t.key ? "bg-primary/15 text-foreground" : "text-muted-foreground hover:text-foreground"
                }`}>
                {t.label}
              </button>
            ))}
            <span className="ml-4 px-2 py-0.5 rounded border border-border text-xs text-muted-foreground/60 font-mono">
              Llama 3.2 3B
            </span>
          </div>
        </div>
      </nav>

      {/* hero */}
      <section className="hero-bg pt-32 pb-16 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-muted-foreground mb-5 border border-border px-3 py-1.5 rounded-full fade-up fade-up-1">
            <span className="w-1.5 h-1.5 rounded-full bg-primary pulse-dot" />
            CS 153 · Stanford · 2026
          </div>
          <h1 className="text-5xl font-bold tracking-tight leading-[1.1] mb-4 fade-up fade-up-2">
            Steer what a language model<br />
            <span className="gradient-text">thinks about</span>
          </h1>
          <p className="text-base text-muted-foreground max-w-2xl mb-4 leading-relaxed fade-up fade-up-3">
            We add a learned direction vector to a model&apos;s residual stream at inference time —
            no fine-tuning, no system prompt — and measure how concepts like writing style,
            demographic bias, and factual knowledge respond.
          </p>
          <p className="text-sm text-muted-foreground max-w-2xl mb-8 leading-relaxed fade-up fade-up-3">
            The central finding: stylistic concept directions occupy a geometrically structured
            subspace (17× more structured than random vectors), while bias directions are near-random.
            This geometric property predicts which interventions are controllable and which are not.
          </p>
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

        {/* ── DISCOVER ── */}
        {tab === "discover" && (
          <>
            <Section eyebrow="Try it live" title="Type anything. Pick a style."
              desc="The model continues your text with a steering vector active in its residual stream. No fine-tuning. No system prompt. Just vector addition.">
              <div className="rounded-2xl border border-border bg-card p-6">
                <StyleInput />
              </div>
            </Section>

            <Section eyebrow="Vector arithmetic" title="Blend two concept directions"
              desc="Add two vectors simultaneously — h′ = h + α₁·v̂_A + α₂·v̂_B. When the directions oppose each other in concept space they fight; when they align they reinforce. The geometry predicts the result.">
              <div className="rounded-2xl border border-border bg-card p-6">
                <StyleBlend />
              </div>
            </Section>

            <StyleLab />
          </>
        )}

        {/* ── EXPLAIN ── */}
        {tab === "explain" && (
          <>
            <ExplainTab />

            <Section eyebrow="Concept scanner" title="What does the model see in your text?"
              desc="A forward pass computes the dot product of the last-token hidden state with each of the 21 probe directions. The bars show how strongly each concept is activated — positive means the representation is aligned with the concept, negative means it points away.">
              <div className="rounded-2xl border border-border bg-card p-6">
                <ConceptScanner />
              </div>
            </Section>
          </>
        )}

        {/* ── BIAS ── */}
        {tab === "bias" && (
          <>
            <Section eyebrow="Bias erasure" title="Erase a bias direction from model completions"
              desc="We subtract the bias concept direction from the residual stream during generation. The same sentence is completed twice — once normally, once with the bias direction removed. Probe scores on the right measure how much each bias direction is activated in each output.">
              <div className="rounded-2xl border border-border bg-card p-6">
                <BiasLab />
              </div>
            </Section>

            <MeasureTab />
          </>
        )}

        {/* ── MEASURE ── */}
        {tab === "measure" && <MeasureTab />}

      </div>

      <footer className="border-t border-border py-6 px-6">
        <div className="max-w-5xl mx-auto flex items-center justify-between text-xs text-muted-foreground font-mono">
          <span>NeuralStyle · CS 153 · Stanford 2026</span>
          <span>RepE-style contrast direction steering · layers 23–26</span>
        </div>
      </footer>
    </main>
  );
}
