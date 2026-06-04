"use client";

import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from "recharts";

type ConceptResult = { crows_stereo_rate: number | null; mmlu: number };
type SweepData = Record<string, Record<string, ConceptResult>>;

type JudgeAlpha = { stereotype_rate: number; examples: { label: string }[] };
type JudgeConcept = {
  label: string; color: string;
  crows_baseline: number;
  judge_rates: Record<string, number>;
};
type JudgeData = Record<string, JudgeConcept>;

const BIAS_CONCEPTS = [
  "age_competence", "gender_profession", "race_crime", "gender_emotion", "nationality_stereotype",
];
const CONCEPT_COLOR: Record<string, string> = {
  age_competence:         "oklch(0.72 0.18 25)",
  gender_profession:      "oklch(0.72 0.15 60)",
  race_crime:             "oklch(0.72 0.18 290)",
  gender_emotion:         "oklch(0.72 0.20 320)",
  nationality_stereotype: "oklch(0.72 0.15 200)",
};
const CONCEPT_LABEL: Record<string, string> = {
  age_competence:         "Age / Competence",
  gender_profession:      "Gender / Profession",
  race_crime:             "Race / Crime",
  gender_emotion:         "Gender / Emotion",
  nationality_stereotype: "Nationality",
};

export default function MeasureTab() {
  const [sweep,   setSweep]   = useState<SweepData | null>(null);
  const [amplify, setAmplify] = useState<SweepData | null>(null);
  const [judge,   setJudge]   = useState<JudgeData | null>(null);

  useEffect(() => {
    fetch("/data/repe_sweep.json").then(r => r.json()).then(setSweep);
    fetch("/data/bias_amplify_sweep.json").then(r => r.json()).then(setAmplify).catch(() => null);
    fetch("/data/bias_completions_ui.json").then(r => r.json()).then(setJudge).catch(() => null);
  }, []);

  const alphaKeys = sweep ? Object.keys(sweep).sort((a,b) => Number(a) - Number(b)) : [];

  const chartData = alphaKeys.map(key => {
    const point: Record<string, number> = { alpha: Number(key) };
    for (const c of BIAS_CONCEPTS) {
      const v = sweep![key]?.[c]?.crows_stereo_rate;
      if (v != null) point[c] = v;
    }
    return point;
  });

  const stats = sweep
    ? BIAS_CONCEPTS.map(c => {
        const baseline = sweep[alphaKeys[0]]?.[c]?.crows_stereo_rate;
        const best     = Math.min(...alphaKeys.map(k => sweep![k]?.[c]?.crows_stereo_rate ?? 1));
        return { c, baseline, best, delta: baseline != null ? (baseline - best) * 100 : null };
      }).filter(s => s.baseline != null)
    : [];
  const headline = stats.find(s => s.c === "age_competence");

  // Bidirectional data
  const bidir = (() => {
    if (!sweep || !amplify) return null;
    const C = "age_competence";
    const ampKeys = Object.keys(amplify).sort((a,b) => Number(a) - Number(b));
    const combined = [
      ...ampKeys.filter(k => Number(k) > 0).reverse().map(k => ({
        alpha: -Number(k),
        rate: amplify[k]?.[C]?.crows_stereo_rate ?? null,
      })),
      { alpha: 0, rate: sweep["0.0"]?.[C]?.crows_stereo_rate ?? null },
      ...alphaKeys.filter(k => Number(k) > 0).map(k => ({
        alpha: Number(k),
        rate: sweep[k]?.[C]?.crows_stereo_rate ?? null,
      })),
    ].filter(p => p.rate != null);
    const maxAmp   = Math.max(...combined.filter(p => p.alpha < 0).map(p => p.rate as number));
    const minDebias = Math.min(...combined.filter(p => p.alpha > 0).map(p => p.rate as number));
    return { combined, swing: ((maxAmp - minDebias) * 100).toFixed(0), maxAmp, minDebias };
  })();

  return (
    <div className="pt-10 space-y-8">
      {/* header */}
      <div>
        <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-muted-foreground mb-3">
          <span className="w-4 h-px bg-primary/60" />
          Benchmark Results
        </div>
        <h2 className="text-3xl font-bold tracking-tight mb-2">Bias reduction, measured</h2>
        <p className="text-sm text-muted-foreground max-w-xl">
          CrowS-Pairs is an external benchmark of 300+ real sentence pairs. For each pair the model
          assigns higher log-probability to one sentence — we measure how often it picks the
          stereotyped one. 50% is chance. Lower is less biased.
        </p>
      </div>

      {/* headline */}
      {headline && (
        <div className="rounded-2xl border bg-card p-6 flex items-center gap-8"
             style={{ borderColor: "oklch(0.72 0.18 25 / 30%)" }}>
          <div>
            <div className="text-6xl font-bold leading-none gradient-text">
              −{headline.delta?.toFixed(0)}pp
            </div>
            <div className="text-xs text-muted-foreground mt-2">Age / Competence · Llama 3.2 3B</div>
          </div>
          <div className="h-14 w-px bg-border" />
          <div>
            <div className="text-xs text-muted-foreground">Baseline stereotype rate</div>
            <div className="text-2xl font-bold">{((headline.baseline ?? 0)*100).toFixed(0)}%</div>
          </div>
          <div className="text-xl text-muted-foreground">→</div>
          <div>
            <div className="text-xs text-muted-foreground">After steering</div>
            <div className="text-2xl font-bold" style={{ color: "oklch(0.72 0.18 145)" }}>
              {((headline.best ?? 0)*100).toFixed(0)}%
            </div>
            <div className="text-xs text-muted-foreground">near chance (50%)</div>
          </div>
          <div className="ml-auto text-right flex-shrink-0">
            <div className="text-xs font-mono text-muted-foreground">method</div>
            <div className="text-sm font-medium mt-1">Contrast direction steering</div>
            <div className="text-xs text-muted-foreground font-mono">layers 23–26 · α=20</div>
          </div>
        </div>
      )}

      {/* CrowS line chart */}
      {sweep && (
        <div className="rounded-2xl border border-border bg-card p-6">
          <p className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-5">
            Stereotype rate vs. steering strength (all 5 bias concepts)
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={chartData} margin={{ top:4, right:24, bottom:16, left:0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="oklch(1 0 0 / 6%)" />
              <XAxis dataKey="alpha"
                     label={{ value: "Steering strength (α)", position:"insideBottom", offset:-8, fill:"oklch(0.45 0 0)", fontSize:11 }}
                     tick={{ fill:"oklch(0.55 0 0)", fontSize:11 }} />
              <YAxis domain={[0.4, 0.8]} tickFormatter={v=>`${(v*100).toFixed(0)}%`}
                     tick={{ fill:"oklch(0.55 0 0)", fontSize:11 }} />
              <Tooltip
                contentStyle={{ background:"oklch(0.14 0 0)", border:"1px solid oklch(1 0 0 / 8%)", borderRadius:8, fontSize:12 }}
                formatter={(v) => typeof v === "number" ? `${(v*100).toFixed(1)}%` : v}
              />
              <Legend formatter={k => CONCEPT_LABEL[k] ?? k} wrapperStyle={{ fontSize:11, paddingTop:10 }} />
              <ReferenceLine y={0.5} stroke="oklch(0.45 0 0)" strokeDasharray="6 3"
                             label={{ value:"50% = chance", position:"right", fill:"oklch(0.40 0 0)", fontSize:10 }} />
              {BIAS_CONCEPTS.map(c => (
                <Line key={c} type="monotone" dataKey={c} stroke={CONCEPT_COLOR[c]} strokeWidth={2.5}
                      dot={{ r:4, fill:CONCEPT_COLOR[c] }} name={c} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* LLM-as-judge section */}
      {judge && (
        <div className="rounded-2xl border border-border bg-card p-6 space-y-5">
          <div>
            <p className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-1">
              LLM-as-judge evaluation · free-generation completions
            </p>
            <p className="text-sm text-muted-foreground max-w-2xl">
              CrowS-Pairs measures log-probability on fixed sentences. This measures actual generation:
              the model completes 18 prompts per concept, and an independent Claude judge labels each
              completion as stereotyped or counter-stereotyped. Four of five concepts reach 0% stereotype
              rate at moderate alpha.
            </p>
          </div>
          <div className="grid gap-4">
            {Object.entries(judge).map(([key, c]) => {
              const alphas = Object.keys(c.judge_rates).sort((a,b) => Number(a) - Number(b));
              const best = Math.min(...Object.values(c.judge_rates));
              const baseline = c.crows_baseline;
              return (
                <div key={key} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono font-semibold" style={{ color: c.color }}>{c.label}</span>
                    <span className="text-xs font-mono text-muted-foreground/50">
                      CrowS baseline {(baseline*100).toFixed(0)}% → judge best {(best*100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="flex items-end gap-2 h-14">
                    {/* baseline bar */}
                    <div className="flex flex-col items-center gap-1 w-12 flex-shrink-0">
                      <div className="w-full rounded-sm flex-1 flex items-end">
                        <div className="w-full rounded-sm"
                          style={{ height: `${baseline * 100}%`, background: "rgba(255,255,255,0.08)" }} />
                      </div>
                      <span className="text-[9px] font-mono text-muted-foreground/40">base</span>
                    </div>
                    {/* alpha bars */}
                    {alphas.map(a => {
                      const rate = c.judge_rates[a];
                      return (
                        <div key={a} className="flex flex-col items-center gap-1 flex-1">
                          <div className="w-full rounded-sm flex-1 flex items-end">
                            <div className="w-full rounded-sm transition-all"
                              style={{
                                height: rate === 0 ? "3px" : `${Math.max(rate * 100, 3)}%`,
                                background: rate === 0 ? c.color + "33" : c.color,
                                opacity: rate === 0 ? 0.4 : 0.85,
                              }} />
                          </div>
                          <span className="text-[9px] font-mono text-muted-foreground/40">α={a}</span>
                        </div>
                      );
                    })}
                  </div>
                  {/* rate labels */}
                  <div className="flex items-center gap-2 text-[9px] font-mono">
                    <span className="w-12 text-center text-muted-foreground/40">{(baseline*100).toFixed(0)}%</span>
                    {alphas.map(a => (
                      <span key={a} className="flex-1 text-center"
                        style={{ color: c.judge_rates[a] === 0 ? c.color : "rgba(160,160,175,0.5)" }}>
                        {(c.judge_rates[a]*100).toFixed(0)}%
                        {c.judge_rates[a] === 0 && " ✓"}
                      </span>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
          <p className="text-xs text-muted-foreground/50 font-mono">
            Judge: Claude Haiku · 18 prompts per concept · each rated independently · race_crime shows partial reduction only
          </p>
        </div>
      )}

      {/* Bidirectional */}
      {bidir && (
        <div className="rounded-2xl border bg-card p-6"
             style={{ borderColor:"oklch(0.72 0.18 25 / 25%)" }}>
          <div className="flex items-start justify-between mb-4">
            <div>
              <p className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-1">
                Bidirectional control · Age / Competence
              </p>
              <p className="text-sm text-muted-foreground max-w-md">
                Same vector, sign flipped. Negative α amplifies bias; positive suppresses it.
              </p>
            </div>
            <div className="text-right flex-shrink-0 ml-6">
              <div className="text-3xl font-bold gradient-text">{bidir.swing}pp</div>
              <div className="text-xs text-muted-foreground">controllable range</div>
              <div className="text-xs font-mono text-muted-foreground mt-0.5">
                {(bidir.maxAmp*100).toFixed(0)}% → {(bidir.minDebias*100).toFixed(0)}%
              </div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={bidir.combined} margin={{ top:4, right:24, bottom:20, left:0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="oklch(1 0 0 / 6%)" />
              <XAxis dataKey="alpha"
                     label={{ value:"← amplify  ·  α  ·  suppress →", position:"insideBottom", offset:-10, fill:"oklch(0.40 0 0)", fontSize:11 }}
                     tick={{ fill:"oklch(0.55 0 0)", fontSize:11 }} />
              <YAxis domain={[0.35, 0.9]} tickFormatter={v=>`${(v*100).toFixed(0)}%`}
                     tick={{ fill:"oklch(0.55 0 0)", fontSize:11 }} />
              <Tooltip
                contentStyle={{ background:"oklch(0.14 0 0)", border:"1px solid oklch(1 0 0 / 8%)", borderRadius:8, fontSize:12 }}
                formatter={(v) => typeof v === "number" ? `${(v*100).toFixed(1)}%` : v}
              />
              <ReferenceLine y={0.5} stroke="oklch(0.45 0 0)" strokeDasharray="6 3"
                             label={{ value:"50% = chance", position:"right", fill:"oklch(0.40 0 0)", fontSize:10 }} />
              <ReferenceLine x={0} stroke="oklch(0.35 0 0)" strokeWidth={1.5} />
              <Line type="monotone" dataKey="rate" strokeWidth={2.5}
                    stroke={CONCEPT_COLOR["age_competence"]}
                    dot={{ r:4, fill:CONCEPT_COLOR["age_competence"] }}
                    name="stereotype rate" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="rounded-xl border border-border/50 bg-card/50 p-4">
        <p className="text-xs text-muted-foreground leading-relaxed">
          <span className="text-foreground font-medium">Why age_competence but not others?</span>{" "}
          The 5 bias concepts show different debiasing curves because their contrast directions
          are not equally well-aligned with the benchmark. For age_competence, the mean-difference
          direction happens to be strongly aligned with what CrowS-Pairs measures (71% baseline → 47%).
          For other concepts the direction is noisier, reflecting our core finding:
          bias does not live in a coherent geometric subspace the way style does.
        </p>
      </div>
    </div>
  );
}
