"use client";

import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from "recharts";

type ConceptResult = { crows_stereo_rate: number | null; mmlu: number };
type SweepData = Record<string, Record<string, ConceptResult>>;

const BIAS_CONCEPTS = ["age_competence","gender_profession","race_crime","gender_emotion","nationality_stereotype"];
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

export default function ResultsSection() {
  const [sweep, setSweep] = useState<SweepData | null>(null);
  const [amplify, setAmplify] = useState<SweepData | null>(null);

  useEffect(() => {
    fetch("/data/repe_sweep.json").then(r => r.json()).then(setSweep);
    fetch("/data/bias_amplify_sweep.json").then(r => r.json()).then(setAmplify).catch(() => null);
  }, []);

  const alphaKeys = sweep ? Object.keys(sweep).sort((a, b) => Number(a) - Number(b)) : [];

  const chartData = alphaKeys.map(key => {
    const point: Record<string, number> = { alpha: Number(key) };
    for (const c of BIAS_CONCEPTS) {
      const v = sweep![key]?.[c]?.crows_stereo_rate;
      if (v != null) point[c] = v;
    }
    return point;
  });

  const stats = sweep ? BIAS_CONCEPTS.map(c => {
    const baseline = sweep[alphaKeys[0]]?.[c]?.crows_stereo_rate;
    const best = Math.min(...alphaKeys.map(k => sweep![k]?.[c]?.crows_stereo_rate ?? 1));
    return { c, baseline, best, delta: baseline != null ? (baseline - best) * 100 : null };
  }).filter(s => s.baseline != null) : [];

  const headline = stats.find(s => s.c === "age_competence");

  return (
    <section className="py-24 px-6 border-t border-border" id="results">
      <div className="max-w-6xl mx-auto">
        <div className="mb-12">
          <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-muted-foreground mb-4">
            <span className="w-4 h-px bg-primary/60" />
            Results
          </div>
          <h2 className="text-4xl font-bold tracking-tight mb-3">
            Bias reduction, measured
          </h2>
          <p className="text-muted-foreground max-w-xl">
            CrowS-Pairs is an external benchmark of 300+ real sentence pairs. For each pair
            the model assigns higher probability to one — we measure how often it picks the
            stereotyped one. 50% is random chance. Lower is less biased.
          </p>
        </div>

        {/* headline stat */}
        {headline && (
          <div className="rounded-2xl border bg-card p-8 mb-8 flex items-center gap-8"
               style={{ borderColor: "oklch(0.72 0.18 25 / 30%)" }}>
            <div>
              <div className="text-7xl font-bold leading-none gradient-text">
                −{headline.delta?.toFixed(0)}pp
              </div>
              <div className="text-sm text-muted-foreground mt-2">
                Age / Competence bias · Llama 3.2 3B
              </div>
            </div>
            <div className="h-16 w-px bg-border" />
            <div className="space-y-1">
              <div className="text-sm text-muted-foreground">Baseline</div>
              <div className="text-2xl font-bold">{((headline.baseline ?? 0) * 100).toFixed(0)}%</div>
              <div className="text-xs text-muted-foreground">stereotype rate</div>
            </div>
            <div className="text-2xl text-muted-foreground">→</div>
            <div className="space-y-1">
              <div className="text-sm text-muted-foreground">Steered</div>
              <div className="text-2xl font-bold" style={{ color: "oklch(0.72 0.18 145)" }}>
                {((headline.best ?? 0) * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-muted-foreground">near chance (50%)</div>
            </div>
            <div className="ml-auto text-right">
              <div className="text-xs font-mono text-muted-foreground">method</div>
              <div className="text-sm font-medium mt-1">Contrast direction steering</div>
              <div className="text-xs text-muted-foreground">layers 23–26 · α=20</div>
            </div>
          </div>
        )}

        {/* stat cards */}
        {stats.length > 0 && (
          <div className="grid grid-cols-5 gap-3 mb-8">
            {stats.map(({ c, baseline, delta }) => (
              <div key={c} className="rounded-xl border border-border bg-card p-4">
                <div className="text-xs font-mono mb-2 truncate"
                     style={{ color: CONCEPT_COLOR[c] }}>
                  {CONCEPT_LABEL[c]}
                </div>
                <div className="text-lg font-bold">{((baseline ?? 0) * 100).toFixed(0)}%</div>
                <div className="text-xs text-muted-foreground">baseline</div>
                <div className="text-sm font-semibold mt-1" style={{ color: "oklch(0.72 0.18 145)" }}>
                  −{delta?.toFixed(1)}pp
                </div>
              </div>
            ))}
          </div>
        )}

        {/* line chart */}
        {sweep && (
          <div className="rounded-2xl border border-border bg-card p-6">
            <p className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-6">
              CrowS-Pairs stereotype rate vs. steering strength (α)
            </p>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData} margin={{ top: 4, right: 24, bottom: 16, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="oklch(1 0 0 / 6%)" />
                <XAxis dataKey="alpha"
                       label={{ value: "Steering strength (α)", position: "insideBottom", offset: -8, fill: "oklch(0.55 0 0)", fontSize: 11 }}
                       tick={{ fill: "oklch(0.55 0 0)", fontSize: 11 }} />
                <YAxis domain={[0.4, 0.75]} tickFormatter={v => `${(v * 100).toFixed(0)}%`}
                       tick={{ fill: "oklch(0.55 0 0)", fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: "oklch(0.14 0 0)", border: "1px solid oklch(1 0 0 / 8%)", borderRadius: 8, fontSize: 12 }}
                  formatter={(v) => typeof v === "number" ? `${(v * 100).toFixed(1)}%` : v}
                />
                <Legend
                  formatter={key => CONCEPT_LABEL[key] ?? key}
                  wrapperStyle={{ fontSize: 11, paddingTop: 12 }}
                />
                <ReferenceLine y={0.5} stroke="oklch(0.55 0 0)" strokeDasharray="6 3"
                               label={{ value: "50% = chance", position: "right", fill: "oklch(0.45 0 0)", fontSize: 10 }} />
                {BIAS_CONCEPTS.map(c => (
                  <Line key={c} type="monotone" dataKey={c}
                        stroke={CONCEPT_COLOR[c]} strokeWidth={2.5}
                        dot={{ r: 4, fill: CONCEPT_COLOR[c] }}
                        name={c} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Bidirectional chart — shown once amplify data lands */}
        {sweep && amplify && (() => {
          const focusConcept = "age_competence";
          const debias = alphaKeys.map(k => ({
            alpha: Number(k),
            rate: sweep[k]?.[focusConcept]?.crows_stereo_rate ?? null,
          })).filter(p => p.rate != null);
          const ampKeys = Object.keys(amplify).sort((a, b) => Number(a) - Number(b));
          const amp = ampKeys
            .filter(k => Number(k) > 0)
            .map(k => ({
              alpha: -Number(k),
              rate: amplify[k]?.[focusConcept]?.crows_stereo_rate ?? null,
            }))
            .filter(p => p.rate != null)
            .reverse();
          const combined = [...amp, ...debias.filter(p => p.alpha > 0)];
          const baseline = debias.find(p => p.alpha === 0)?.rate ?? 0;

          return (
            <div className="rounded-2xl border border-border bg-card p-6 mt-6">
              <p className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-1">
                Bidirectional control — Age / Competence
              </p>
              <p className="text-sm text-muted-foreground mb-6">
                The same steering vector, applied in two directions. Left of zero: bias amplified. Right: bias reduced.
                This is the same model, same weights — just the sign of α changes.
              </p>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={combined} margin={{ top: 4, right: 24, bottom: 16, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="oklch(1 0 0 / 6%)" />
                  <XAxis dataKey="alpha"
                         label={{ value: "← amplify  ·  α  ·  debias →", position: "insideBottom", offset: -8, fill: "oklch(0.55 0 0)", fontSize: 11 }}
                         tick={{ fill: "oklch(0.55 0 0)", fontSize: 11 }} />
                  <YAxis domain={[0.35, 0.85]} tickFormatter={v => `${(v * 100).toFixed(0)}%`}
                         tick={{ fill: "oklch(0.55 0 0)", fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ background: "oklch(0.14 0 0)", border: "1px solid oklch(1 0 0 / 8%)", borderRadius: 8, fontSize: 12 }}
                    formatter={(v) => typeof v === "number" ? `${(v * 100).toFixed(1)}%` : v}
                  />
                  <ReferenceLine y={0.5} stroke="oklch(0.55 0 0)" strokeDasharray="6 3"
                                 label={{ value: "50% = chance", position: "right", fill: "oklch(0.45 0 0)", fontSize: 10 }} />
                  <ReferenceLine y={baseline} stroke="oklch(0.55 0 0)" strokeDasharray="3 3"
                                 label={{ value: "baseline", position: "insideTopRight", fill: "oklch(0.45 0 0)", fontSize: 10 }} />
                  <ReferenceLine x={0} stroke="oklch(0.40 0 0)" strokeWidth={1} />
                  <Line type="monotone" dataKey="rate"
                        stroke={CONCEPT_COLOR[focusConcept]} strokeWidth={2.5}
                        dot={{ r: 4, fill: CONCEPT_COLOR[focusConcept] }}
                        name="stereotype rate" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          );
        })()}
      </div>
    </section>
  );
}
