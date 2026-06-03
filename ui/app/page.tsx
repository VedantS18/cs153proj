"use client";

import StyleLab from "@/components/StyleLab";
import StyleCompass from "@/components/StyleCompass";
import ConceptAtlas from "@/components/ConceptAtlas";
import ResultsSection from "@/components/ResultsSection";
import ProbeSpace3D from "@/components/ProbeSpace3D";

export default function Home() {
  return (
    <main className="min-h-screen">
      {/* nav */}
      <nav className="fixed top-0 inset-x-0 z-50 border-b border-border/60 backdrop-blur-md bg-background/80">
        <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
          <span className="font-mono text-sm font-semibold tracking-tight">
            <span className="gradient-text">Neural</span>Style
          </span>
          <div className="flex items-center gap-6 text-xs text-muted-foreground font-mono">
            <a href="#lab" className="hover:text-foreground transition-colors">Style Lab</a>
            <a href="#probe-space" className="hover:text-foreground transition-colors">Vector Space</a>
            <a href="#compass" className="hover:text-foreground transition-colors">Compass</a>
            <a href="#atlas" className="hover:text-foreground transition-colors">Atlas</a>
            <a href="#results" className="hover:text-foreground transition-colors">Results</a>
            <span className="px-2 py-0.5 rounded border border-border text-muted-foreground/60">
              Llama 3.2 3B
            </span>
          </div>
        </div>
      </nav>

      {/* hero */}
      <section className="hero-bg pt-32 pb-24 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="fade-up fade-up-1">
            <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-muted-foreground mb-6 border border-border px-3 py-1.5 rounded-full">
              <span className="w-1.5 h-1.5 rounded-full bg-primary pulse-dot" />
              CS 153 · Stanford · 2026
            </div>
          </div>

          <h1 className="text-6xl font-bold tracking-tight leading-[1.1] mb-6 fade-up fade-up-2">
            Steer what a language model<br />
            <span className="gradient-text">thinks about</span>
          </h1>

          <p className="text-lg text-muted-foreground max-w-2xl mb-10 leading-relaxed fade-up fade-up-3">
            We add a single learned vector to a model&apos;s hidden states at inference time —
            no fine-tuning, no prompts — and watch it write like Hemingway, shed stereotypes,
            or forget factual knowledge on demand.
            Then we explain <em>why</em> some interventions work and others don&apos;t.
          </p>

          <div className="flex gap-4 fade-up fade-up-3">
            <a
              href="#lab"
              className="px-6 py-3 rounded-xl font-medium text-sm text-primary-foreground transition-all hover:opacity-90 active:scale-95"
              style={{ background: "oklch(0.72 0.20 270)" }}
            >
              Try Style Lab →
            </a>
            <a
              href="#results"
              className="px-6 py-3 rounded-xl font-medium text-sm border border-border text-muted-foreground hover:text-foreground hover:border-border/60 transition-all"
            >
              View Results
            </a>
          </div>

          {/* quick stats */}
          <div className="mt-16 grid grid-cols-3 gap-px bg-border rounded-2xl overflow-hidden fade-up fade-up-3">
            {[
              { value: "15", label: "concepts studied", sub: "factual · bias · style" },
              { value: "−24pp", label: "bias reduction", sub: "age / competence on CrowS-Pairs" },
              { value: "cos ≈ 0.16", label: "probe–contrast gap", sub: "explains why erasure failed" },
            ].map(({ value, label, sub }) => (
              <div key={label} className="bg-card px-8 py-6">
                <div className="text-3xl font-bold gradient-text mb-1">{value}</div>
                <div className="text-sm font-medium">{label}</div>
                <div className="text-xs text-muted-foreground mt-0.5">{sub}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <StyleLab />
      <ProbeSpace3D />
      <StyleCompass />
      <ConceptAtlas />
      <ResultsSection />

      {/* footer */}
      <footer className="border-t border-border py-8 px-6">
        <div className="max-w-6xl mx-auto flex items-center justify-between text-xs text-muted-foreground font-mono">
          <span>NeuralStyle · CS 153 · Stanford 2026</span>
          <span>Llama 3.2 3B · RepE-style contrast direction steering</span>
        </div>
      </footer>
    </main>
  );
}
