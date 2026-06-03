"use client";

import { useEffect, useState } from "react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ResponsiveContainer, Tooltip,
} from "recharts";

const STYLE_COLOR: Record<string, string> = {
  hemingway:         "oklch(0.72 0.18 25)",
  shakespeare:       "oklch(0.72 0.18 290)",
  legal_text:        "oklch(0.72 0.15 200)",
  scientific_writing:"oklch(0.72 0.18 145)",
  news_wire:         "oklch(0.72 0.15 60)",
};

const CONCEPT_LABEL: Record<string, string> = {
  hemingway:          "Hemingway",
  shakespeare:        "Shakespeare",
  legal_text:         "Legal",
  scientific_writing: "Scientific",
  news_wire:          "News Wire",
  gender_profession:  "Gender / Profession",
  gender_emotion:     "Gender / Emotion",
  age_competence:     "Age / Competence",
  race_crime:         "Race / Crime",
  nationality_stereotype: "Nationality",
  capital_cities:     "Capital Cities",
  element_symbols:    "Chemistry",
  inventor_invention: "Inventors",
  country_language:   "Languages",
  historical_dates:   "History",
};

type ContrastData = Record<string, { cos_probe_contrast: number; peak_layer: number }>;

const CATEGORIES = {
  stylistic: ["hemingway","shakespeare","legal_text","scientific_writing","news_wire"],
  bias:      ["gender_profession","gender_emotion","age_competence","race_crime","nationality_stereotype"],
  factual:   ["capital_cities","element_symbols","inventor_invention","country_language","historical_dates"],
};

const CAT_COLOR = {
  stylistic: "oklch(0.72 0.18 145)",
  bias:      "oklch(0.72 0.18 25)",
  factual:   "oklch(0.72 0.20 270)",
};

const CAT_LABEL = { stylistic: "Writing Style", bias: "Social Bias", factual: "Factual Knowledge" };

export default function StyleCompass() {
  const [data, setData] = useState<ContrastData | null>(null);

  useEffect(() => {
    fetch("/data/contrast_directions.json").then(r => r.json()).then(setData);
  }, []);

  if (!data) return null;

  // Radar data: for each concept category, show avg cos alignment
  const radarData = Object.entries(CATEGORIES).map(([cat, keys]) => {
    const avg = keys.reduce((s, k) => s + (data[k]?.cos_probe_contrast ?? 0), 0) / keys.length;
    return {
      category: CAT_LABEL[cat as keyof typeof CAT_LABEL],
      alignment: parseFloat((avg * 100).toFixed(1)),
      color: CAT_COLOR[cat as keyof typeof CAT_COLOR],
    };
  });

  // Sorted bar list
  const allConcepts = Object.entries(data)
    .map(([k, v]) => ({ key: k, cos: v.cos_probe_contrast }))
    .sort((a, b) => b.cos - a.cos);

  const catOf = (k: string) =>
    CATEGORIES.stylistic.includes(k) ? "stylistic" :
    CATEGORIES.bias.includes(k) ? "bias" : "factual";

  return (
    <section className="py-24 px-6 border-t border-border" id="compass">
      <div className="max-w-6xl mx-auto">
        <div className="mb-12">
          <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-muted-foreground mb-4">
            <span className="w-4 h-px bg-primary/60" />
            Style Compass
          </div>
          <h2 className="text-4xl font-bold tracking-tight mb-3">
            Why erasure fails — and steering works
          </h2>
          <p className="text-muted-foreground max-w-xl">
            We measure the angle between two directions for each concept: the{" "}
            <span className="text-foreground font-medium">probe direction</span>{" "}
            (what a linear classifier learns to detect) and the{" "}
            <span className="text-foreground font-medium">contrast direction</span>{" "}
            (mean difference between concept-present and concept-absent activations).
            When these align, interventions work. When they don&apos;t, the model ignores them.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-8 items-start">
          {/* bar chart — all concepts */}
          <div>
            <p className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-4">
              cos(probe, contrast) per concept
            </p>
            <div className="space-y-2">
              {allConcepts.map(({ key, cos }) => {
                const cat = catOf(key);
                const color = CAT_COLOR[cat as keyof typeof CAT_COLOR];
                return (
                  <div key={key} className="flex items-center gap-3">
                    <span className="text-xs text-muted-foreground w-36 truncate font-mono">
                      {CONCEPT_LABEL[key] ?? key}
                    </span>
                    <div className="flex-1 h-5 rounded bg-muted/30 overflow-hidden">
                      <div
                        className="h-full rounded transition-all duration-700"
                        style={{ width: `${cos * 100}%`, background: color }}
                      />
                    </div>
                    <span className="text-xs font-mono w-10 text-right"
                          style={{ color }}>
                      {cos.toFixed(2)}
                    </span>
                  </div>
                );
              })}
            </div>
            <div className="flex gap-4 mt-5">
              {Object.entries(CAT_LABEL).map(([cat, label]) => (
                <div key={cat} className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ background: CAT_COLOR[cat as keyof typeof CAT_COLOR] }} />
                  <span className="text-xs text-muted-foreground">{label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* insight cards */}
          <div className="space-y-4">
            <div className="rounded-2xl border border-border bg-card p-6">
              <div className="text-3xl font-bold mb-1" style={{ color: CAT_COLOR.stylistic }}>
                cos ≈ 1.0
              </div>
              <div className="text-sm font-medium mb-2">Writing style concepts</div>
              <p className="text-sm text-muted-foreground">
                The probe direction and contrast direction are nearly identical for Hemingway,
                Shakespeare, and legal text. Both interventions work — erasure and steering —
                because they&apos;re pointing at the same thing.
              </p>
            </div>

            <div className="rounded-2xl border border-border bg-card p-6">
              <div className="text-3xl font-bold mb-1" style={{ color: CAT_COLOR.bias }}>
                cos ≈ 0.16
              </div>
              <div className="text-sm font-medium mb-2">Social bias concepts</div>
              <p className="text-sm text-muted-foreground">
                For gender and race bias, the probe direction is nearly orthogonal to the
                contrast direction. A classifier can detect the concept, but erasing along
                that axis does nothing to behavior. Steering with the contrast direction does.
              </p>
            </div>

            <div className="rounded-2xl border border-border bg-card p-6">
              <div className="text-3xl font-bold mb-1" style={{ color: CAT_COLOR.factual }}>
                cos ≈ 0.40
              </div>
              <div className="text-sm font-medium mb-2">Factual knowledge concepts</div>
              <p className="text-sm text-muted-foreground">
                Factual concepts fall in between. Partial alignment means interventions
                have some effect but the model&apos;s factual recall is distributed enough
                to resist full suppression.
              </p>
            </div>
          </div>
        </div>

        {/* radar chart */}
        <div className="mt-12 rounded-2xl border border-border bg-card p-8">
          <p className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-6 text-center">
            Average direction alignment by concept category
          </p>
          <ResponsiveContainer width="100%" height={280}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="oklch(1 0 0 / 8%)" />
              <PolarAngleAxis dataKey="category" tick={{ fill: "oklch(0.55 0 0)", fontSize: 12 }} />
              <Radar
                name="Alignment"
                dataKey="alignment"
                stroke="oklch(0.72 0.20 270)"
                fill="oklch(0.72 0.20 270)"
                fillOpacity={0.15}
                strokeWidth={2}
              />
              <Tooltip
                contentStyle={{ background: "oklch(0.14 0 0)", border: "1px solid oklch(1 0 0 / 8%)", borderRadius: 8 }}
                formatter={(v) => [typeof v === "number" ? `${v}%` : v, "Alignment"]}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  );
}
