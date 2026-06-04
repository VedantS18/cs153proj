"use client";
import { useEffect, useState } from "react";

type Example = {
  prompt: string;
  baseline: string;
  steered: string;
  baseline_label: string;
  steered_label: string;
};
type ConceptData = {
  label: string;
  color: string;
  crows_baseline: number;
  judge_rates: Record<string, number>;
  insight: string;
  examples: Example[];
};
type BiasData = Record<string, ConceptData>;

const ALPHA_ORDER = ["1.0", "5.0", "10.0", "15.0", "20.0"];

export default function BiasCompletions() {
  const [data,    setData]    = useState<BiasData | null>(null);
  const [concept, setConcept] = useState("age_competence");

  useEffect(() => {
    fetch("/data/bias_completions_ui.json").then(r => r.json()).then(setData).catch(() => null);
  }, []);

  if (!data) return (
    <div className="h-32 flex items-center justify-center text-xs text-muted-foreground/40 font-mono">
      Loading…
    </div>
  );

  const d = data[concept];

  return (
    <div className="space-y-6">
      {/* concept tabs */}
      <div className="flex flex-wrap gap-2">
        {Object.entries(data).map(([key, val]) => (
          <button key={key} onClick={() => setConcept(key)}
            className="px-3 py-1.5 rounded-lg border text-xs transition-all"
            style={{
              borderColor: key === concept ? val.color : "rgba(255,255,255,0.08)",
              background:  key === concept ? val.color + "18" : "transparent",
              color:       key === concept ? val.color : "rgba(160,160,175,0.6)",
            }}>
            {val.label}
          </button>
        ))}
      </div>

      {/* rates bar */}
      <div className="rounded-xl border border-border bg-card p-4">
        <div className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-3">
          Stereotype rate as erasure strength increases
        </div>
        <div className="flex items-end gap-3">
          {/* baseline */}
          <div className="flex flex-col items-center gap-1">
            <div className="text-xs font-mono" style={{ color: d.color }}>
              {(d.crows_baseline * 100).toFixed(0)}%
            </div>
            <div className="w-12 rounded-t-sm" style={{
              height: `${d.crows_baseline * 80}px`,
              background: d.color + "80",
            }} />
            <div className="text-xs text-muted-foreground/50 font-mono">baseline</div>
          </div>
          <div className="text-muted-foreground/30 text-xs pb-5">→</div>
          {ALPHA_ORDER.filter(a => a in d.judge_rates).map(alpha => {
            const rate = d.judge_rates[alpha];
            return (
              <div key={alpha} className="flex flex-col items-center gap-1">
                <div className="text-xs font-mono" style={{ color: rate === 0 ? "#34d399" : d.color }}>
                  {rate === 0 ? "0%" : `${(rate * 100).toFixed(0)}%`}
                </div>
                <div className="w-12 rounded-t-sm transition-all" style={{
                  height: rate === 0 ? "4px" : `${rate * 80}px`,
                  background: rate === 0 ? "#34d39966" : d.color + "60",
                }} />
                <div className="text-xs text-muted-foreground/50 font-mono">α={alpha}</div>
              </div>
            );
          })}
        </div>
        <p className="text-xs text-muted-foreground/60 mt-3 leading-relaxed">{d.insight}</p>
      </div>

      {/* before / after completions */}
      <div className="space-y-4">
        <div className="text-xs font-mono uppercase tracking-widest text-muted-foreground">
          Actual model completions — before and after
        </div>
        {d.examples.map((ex, i) => (
          <div key={i} className="space-y-1">
            <div className="text-xs font-mono text-muted-foreground/50 px-1">
              prompt: &ldquo;{ex.prompt}&rdquo;
            </div>
            <div className="grid gap-2" style={{ gridTemplateColumns: "1fr 1fr" }}>
              {/* baseline */}
              <div className="rounded-xl border border-border bg-card p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-mono text-muted-foreground/50">No erasure</span>
                  <span className="text-xs font-mono px-2 py-0.5 rounded-full"
                    style={{ background: d.color + "22", color: d.color }}>
                    {ex.baseline_label}
                  </span>
                </div>
                <p className="text-sm leading-relaxed">
                  <span className="text-muted-foreground/50">{ex.prompt} </span>
                  <span>{ex.baseline}</span>
                </p>
              </div>
              {/* steered */}
              <div className="rounded-xl border p-4"
                style={{ borderColor: "#34d39940", background: "#34d39908" }}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-mono" style={{ color: "#34d39999" }}>
                    Bias direction erased
                  </span>
                  <span className="text-xs font-mono px-2 py-0.5 rounded-full"
                    style={{ background: "#34d39922", color: "#34d399" }}>
                    {ex.steered_label}
                  </span>
                </div>
                <p className="text-sm leading-relaxed">
                  <span className="text-muted-foreground/50">{ex.prompt} </span>
                  <span>{ex.steered}</span>
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="text-xs text-muted-foreground/40 leading-relaxed">
        Completions at baseline are drawn from the unsteered model. Steered completions use the best-performing
        erasure strength for each concept. Labels assigned by an independent LLM judge.
      </div>
    </div>
  );
}
