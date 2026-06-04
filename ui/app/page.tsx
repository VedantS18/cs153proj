"use client";

import { useState } from "react";
import StyleLab from "@/components/StyleLab";
import StyleInput from "@/components/StyleInput";
import ExplainTab from "@/components/ExplainTab";
import MeasureTab from "@/components/MeasureTab";

const TABS = [
  { key: "discover", label: "Discover", sub: "Style steering demo" },
  { key: "explain",  label: "Explain",  sub: "The geometry" },
  { key: "measure",  label: "Measure",  sub: "Benchmark results" },
] as const;
type Tab = typeof TABS[number]["key"];

function LiveSteer() {
  return (
    <div className="pt-10">
      <div className="mb-8">
        <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-muted-foreground mb-3">
          <span className="w-4 h-px bg-primary/60" />
          Try it live
        </div>
        <h2 className="text-3xl font-bold tracking-tight mb-2">Type anything. Pick a style.</h2>
        <p className="text-muted-foreground text-sm max-w-xl">
          The model runs twice on your input — once with no intervention, once with a steering
          vector added to its residual stream. The difference is the style direction at work.
        </p>
      </div>
      <div className="rounded-2xl border border-border bg-card p-6">
        <StyleInput />
      </div>
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
              <button
                key={t.key}
                onClick={() => setTab(t.key)}
                className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  tab === t.key
                    ? "bg-primary/15 text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
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

          {/* tab strip */}
          <div className="flex gap-3 fade-up fade-up-3">
            {TABS.map(t => (
              <button
                key={t.key}
                onClick={() => setTab(t.key)}
                className={`group flex flex-col items-start px-5 py-3 rounded-xl border text-left transition-all ${
                  tab === t.key
                    ? "border-primary/40 bg-primary/8 text-foreground"
                    : "border-border text-muted-foreground hover:text-foreground hover:border-border/60"
                }`}
              >
                <span className="text-sm font-semibold">{t.label}</span>
                <span className="text-xs text-muted-foreground mt-0.5">{t.sub}</span>
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* tab content */}
      <div className="max-w-5xl mx-auto px-6 pb-24">
        {tab === "discover" && (
          <>
            <LiveSteer />
            <StyleLab />
          </>
        )}
        {tab === "explain"  && <ExplainTab />}
        {tab === "measure"  && <MeasureTab />}
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
