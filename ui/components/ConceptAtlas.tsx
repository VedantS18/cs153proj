"use client";

import { useEffect, useRef, useState } from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Label,
} from "recharts";

// Concept ordering: stylistic together, bias together, factual together
const CONCEPTS = [
  // stylistic — formal/technical cluster
  "scientific_writing", "legal_text", "news_wire",
  // stylistic — literary cluster
  "hemingway", "shakespeare",
  // bias
  "gender_profession", "gender_emotion", "age_competence",
  "race_crime", "nationality_stereotype",
  // factual
  "capital_cities", "element_symbols", "inventor_invention",
  "country_language", "historical_dates",
];

const CONCEPT_LABEL: Record<string, string> = {
  scientific_writing:    "Scientific",
  legal_text:            "Legal",
  news_wire:             "News Wire",
  hemingway:             "Hemingway",
  shakespeare:           "Shakespeare",
  gender_profession:     "Gender / Job",
  gender_emotion:        "Gender / Emotion",
  age_competence:        "Age / Competence",
  race_crime:            "Race / Crime",
  nationality_stereotype:"Nationality",
  capital_cities:        "Capitals",
  element_symbols:       "Chemistry",
  inventor_invention:    "Inventors",
  country_language:      "Languages",
  historical_dates:      "History",
};

const CATEGORY: Record<string, "stylistic" | "bias" | "factual"> = {
  scientific_writing: "stylistic", legal_text: "stylistic", news_wire: "stylistic",
  hemingway: "stylistic", shakespeare: "stylistic",
  gender_profession: "bias", gender_emotion: "bias", age_competence: "bias",
  race_crime: "bias", nationality_stereotype: "bias",
  capital_cities: "factual", element_symbols: "factual", inventor_invention: "factual",
  country_language: "factual", historical_dates: "factual",
};

const CAT_COLOR = {
  stylistic: "#a78bfa",  // violet
  bias:      "#f87171",  // red
  factual:   "#60a5fa",  // blue
};

type OverlapData = {
  cosine_similarity: Record<string, Record<string, number>>;
};

type GeomStats = { mean: number; std: number; max: number; min: number };
type TopPair = { c1: string; c2: string; cos_3b: number; cos_1b: number | null; cat: string };
type GeomComparison = {
  null_stats: GeomStats;
  stats_3b: { style: GeomStats; bias: GeomStats; factual: GeomStats };
  stats_1b: { style: GeomStats; bias: GeomStats; factual: GeomStats };
  top_pairs: TopPair[];
};

const PAIR_LABEL: Record<string, string> = {
  scientific_writing: "Scientific", legal_text: "Legal", news_wire: "News",
  hemingway: "Hemingway", shakespeare: "Shakespeare",
};
function pairLabel(c: string) { return PAIR_LABEL[c] ?? CONCEPT_LABEL[c] ?? c; }

// Map a cosine value in [-1,1] to a color.
// Negative → blue tones, zero → dark gray, positive → amber/red tones
function cosToColor(v: number): string {
  if (v >= 0.98) return "oklch(0.14 0 0)"; // diagonal — black
  const abs = Math.abs(v);
  if (abs < 0.05) return "oklch(0.18 0 0)"; // ~zero — near background
  if (v > 0) {
    // positive: amber → red
    const t = Math.min(abs / 0.65, 1);
    const l = 0.65 - t * 0.25;
    const c = 0.10 + t * 0.20;
    return `oklch(${l.toFixed(2)} ${c.toFixed(2)} 30)`;
  } else {
    // negative: slate → blue
    const t = Math.min(abs / 0.45, 1);
    const l = 0.55 - t * 0.15;
    const c = 0.08 + t * 0.18;
    return `oklch(${l.toFixed(2)} ${c.toFixed(2)} 250)`;
  }
}

const ANNOTATIONS = [
  { r1: 0, r2: 2, c1: 0, c2: 2, label: "Formal prose cluster", color: "#a78bfa" },
  { r1: 3, r2: 4, c1: 3, c2: 4, label: "Literary cluster", color: "#a78bfa" },
];

export default function ConceptAtlas() {
  const [data, setData] = useState<OverlapData | null>(null);
  const [geom, setGeom] = useState<GeomComparison | null>(null);
  const [hovered, setHovered] = useState<{ r: number; c: number; v: number } | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    fetch("/data/subspace_overlap.json").then(r => r.json()).then(setData);
    fetch("/data/geometry_comparison.json").then(r => r.json()).then(setGeom);
  }, []);

  if (!data) return null;

  const cos = data.cosine_similarity;
  const n = CONCEPTS.length;
  const cellSize = 42;
  const labelWidth = 96;
  const totalW = labelWidth + n * cellSize;
  const totalH = labelWidth + n * cellSize;

  const selectedRow = selected ? CONCEPTS.indexOf(selected) : -1;

  return (
    <section className="py-24 px-6 border-t border-border" id="atlas">
      <div className="max-w-6xl mx-auto">
        <div className="mb-12">
          <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-muted-foreground mb-4">
            <span className="w-4 h-px bg-primary/60" />
            Concept Atlas
          </div>
          <h2 className="text-4xl font-bold tracking-tight mb-3">
            How concepts relate inside the model
          </h2>
          <p className="text-muted-foreground max-w-2xl">
            Each cell shows the cosine similarity between two concept directions in
            the model&apos;s residual stream. <span className="text-amber-400 font-medium">Warm = correlated</span>{" "}
            (steering one also steers the other).{" "}
            <span className="text-blue-400 font-medium">Cool = anti-correlated</span>{" "}
            (steering one opposes the other).
            Bias and factual concepts are orthogonal islands — nearly zero correlation with everything.
            Style has structure: formal prose and literary writing occupy opposite ends of a shared axis.
          </p>
        </div>

        <div className="flex gap-8 items-start flex-wrap">
          {/* Heatmap */}
          <div className="overflow-x-auto">
            <svg
              width={totalW + 8}
              height={totalH + 8}
              style={{ display: "block" }}
            >
              {/* Column labels (rotated) */}
              {CONCEPTS.map((c, i) => (
                <g
                  key={`col-${c}`}
                  transform={`translate(${labelWidth + i * cellSize + cellSize / 2}, ${labelWidth - 6})`}
                  style={{ cursor: "pointer" }}
                  onClick={() => setSelected(selected === c ? null : c)}
                >
                  <text
                    transform="rotate(-45)"
                    textAnchor="start"
                    fontSize={10}
                    fill={selected === c ? CAT_COLOR[CATEGORY[c]] : CATEGORY[c] === "stylistic" ? "#a78bfa99" : CATEGORY[c] === "bias" ? "#f8717199" : "#60a5fa99"}
                    fontWeight={selected === c ? "600" : "400"}
                    fontFamily="var(--font-geist-mono)"
                  >
                    {CONCEPT_LABEL[c]}
                  </text>
                </g>
              ))}

              {/* Row labels + cells */}
              {CONCEPTS.map((rowC, r) => (
                <g key={`row-${rowC}`}>
                  {/* Row label */}
                  <text
                    x={labelWidth - 8}
                    y={labelWidth + r * cellSize + cellSize / 2 + 4}
                    textAnchor="end"
                    fontSize={10}
                    fill={selected === rowC ? CAT_COLOR[CATEGORY[rowC]] : CATEGORY[rowC] === "stylistic" ? "#a78bfa99" : CATEGORY[rowC] === "bias" ? "#f8717199" : "#60a5fa99"}
                    fontWeight={selected === rowC ? "600" : "400"}
                    fontFamily="var(--font-geist-mono)"
                    style={{ cursor: "pointer" }}
                    onClick={() => setSelected(selected === rowC ? null : rowC)}
                  >
                    {CONCEPT_LABEL[rowC]}
                  </text>

                  {/* Cells */}
                  {CONCEPTS.map((colC, c) => {
                    const v = cos[rowC]?.[colC] ?? 0;
                    const isDiag = r === c;
                    const isHighlighted = selected ? (selected === rowC || selected === colC) : true;
                    const isHovered = hovered?.r === r && hovered?.c === c;

                    return (
                      <g
                        key={`cell-${r}-${c}`}
                        onMouseEnter={() => !isDiag && setHovered({ r, c, v })}
                        onMouseLeave={() => setHovered(null)}
                        style={{ cursor: isDiag ? "default" : "crosshair" }}
                      >
                        <rect
                          x={labelWidth + c * cellSize + 1}
                          y={labelWidth + r * cellSize + 1}
                          width={cellSize - 2}
                          height={cellSize - 2}
                          rx={3}
                          fill={isDiag ? "oklch(0.22 0 0)" : cosToColor(v)}
                          opacity={isHighlighted ? 1 : 0.25}
                          stroke={isHovered ? "white" : "transparent"}
                          strokeWidth={1.5}
                        />
                        {!isDiag && Math.abs(v) >= 0.12 && (
                          <text
                            x={labelWidth + c * cellSize + cellSize / 2}
                            y={labelWidth + r * cellSize + cellSize / 2 + 4}
                            textAnchor="middle"
                            fontSize={8.5}
                            fontFamily="var(--font-geist-mono)"
                            fill="white"
                            opacity={isHighlighted ? 0.9 : 0.2}
                          >
                            {v > 0 ? "+" : ""}{v.toFixed(2)}
                          </text>
                        )}
                      </g>
                    );
                  })}
                </g>
              ))}

              {/* Category bracket lines */}
              {[
                { start: 0, end: 4, label: "Style", color: "#a78bfa" },
                { start: 5, end: 9, label: "Bias", color: "#f87171" },
                { start: 10, end: 14, label: "Factual", color: "#60a5fa" },
              ].map(({ start, end, label, color }) => (
                <g key={label}>
                  <line
                    x1={labelWidth + start * cellSize}
                    y1={labelWidth + (end + 1) * cellSize + 3}
                    x2={labelWidth + (end + 1) * cellSize}
                    y2={labelWidth + (end + 1) * cellSize + 3}
                    stroke={color}
                    strokeWidth={1.5}
                    opacity={0.4}
                  />
                  <line
                    x1={labelWidth + (end + 1) * cellSize + 3}
                    y1={labelWidth + start * cellSize}
                    x2={labelWidth + (end + 1) * cellSize + 3}
                    y2={labelWidth + (end + 1) * cellSize}
                    stroke={color}
                    strokeWidth={1.5}
                    opacity={0.4}
                  />
                </g>
              ))}
            </svg>

            {/* Tooltip */}
            {hovered && !CONCEPTS[hovered.r] === !CONCEPTS[hovered.c] && (
              <div className="mt-2 text-xs font-mono text-muted-foreground">
                {CONCEPT_LABEL[CONCEPTS[hovered.r]]} ↔ {CONCEPT_LABEL[CONCEPTS[hovered.c]]}: {hovered.v > 0 ? "+" : ""}{hovered.v.toFixed(4)}
              </div>
            )}
            {hovered && (
              <div className="mt-2 text-xs font-mono text-muted-foreground">
                {CONCEPT_LABEL[CONCEPTS[hovered.r]]} ↔ {CONCEPT_LABEL[CONCEPTS[hovered.c]]}: <span className={hovered.v > 0.05 ? "text-amber-400" : hovered.v < -0.05 ? "text-blue-400" : ""}>{hovered.v > 0 ? "+" : ""}{hovered.v.toFixed(4)}</span>
              </div>
            )}
          </div>

          {/* Insight panel */}
          <div className="flex-1 min-w-64 space-y-4">
            {/* Color legend */}
            <div className="rounded-xl border border-border bg-card p-4">
              <p className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-3">Legend</p>
              <div className="flex items-center gap-2 mb-2">
                <div className="w-24 h-3 rounded" style={{ background: "linear-gradient(to right, oklch(0.40 0.26 250), oklch(0.18 0 0), oklch(0.60 0.30 30))" }} />
                <span className="text-xs text-muted-foreground font-mono">−0.6 → 0 → +0.6</span>
              </div>
              <div className="flex justify-between text-xs font-mono">
                <span className="text-blue-400">anti-correlated</span>
                <span className="text-amber-400">correlated</span>
              </div>
            </div>

            {/* Key findings */}
            <div className="rounded-xl border border-border bg-card p-4 space-y-3">
              <p className="text-xs font-mono uppercase tracking-widest text-muted-foreground">Key findings</p>

              <div>
                <div className="flex items-baseline gap-1.5 mb-0.5">
                  <span className="font-mono text-sm font-bold text-amber-400">+0.58</span>
                  <span className="text-xs text-muted-foreground">Legal ↔ Scientific</span>
                </div>
                <p className="text-xs text-muted-foreground">Formal writing styles share a common direction. Steering one also steers the other.</p>
              </div>

              <div>
                <div className="flex items-baseline gap-1.5 mb-0.5">
                  <span className="font-mono text-sm font-bold text-blue-400">−0.38</span>
                  <span className="text-xs text-muted-foreground">Shakespeare ↔ Scientific</span>
                </div>
                <p className="text-xs text-muted-foreground">Literary and formal writing are geometrically opposed. Injecting one suppresses the other.</p>
              </div>

              <div>
                <div className="flex items-baseline gap-1.5 mb-0.5">
                  <span className="font-mono text-sm font-bold" style={{ color: "oklch(0.65 0.05 0)" }}>≈ 0.0</span>
                  <span className="text-xs text-muted-foreground">All bias concepts</span>
                </div>
                <p className="text-xs text-muted-foreground">Bias concepts are orthogonal to each other and to everything else — independent subspaces.</p>
              </div>
            </div>

            {/* Category legend */}
            <div className="rounded-xl border border-border bg-card p-4">
              <p className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-3">Categories</p>
              {[
                { cat: "stylistic", label: "Writing Style", color: "#a78bfa" },
                { cat: "bias", label: "Social Bias", color: "#f87171" },
                { cat: "factual", label: "Factual Knowledge", color: "#60a5fa" },
              ].map(({ label, color }) => (
                <div key={label} className="flex items-center gap-2 mb-1.5">
                  <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: color }} />
                  <span className="text-xs text-muted-foreground">{label}</span>
                </div>
              ))}
              <p className="text-xs text-muted-foreground mt-2">Click a concept to highlight its row and column.</p>
            </div>
          </div>
        </div>

        {/* ── Panel 1: Null model comparison ─────────────────────────────── */}
        {geom && (() => {
          const cats = [
            { key: "style",   label: "Style",   color: "#a78bfa", std3: geom.stats_3b.style.std,   std1: geom.stats_1b.style.std },
            { key: "factual", label: "Factual", color: "#60a5fa", std3: geom.stats_3b.factual.std, std1: geom.stats_1b.factual.std },
            { key: "bias",    label: "Bias",    color: "#f87171", std3: geom.stats_3b.bias.std,    std1: geom.stats_1b.bias.std },
          ];
          const randStd = geom.null_stats.std;
          const maxStd = geom.stats_3b.style.std;

          return (
            <div className="mt-12 grid grid-cols-2 gap-6">
              {/* bar chart panel */}
              <div className="rounded-2xl border border-border bg-card p-6">
                <p className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-1">
                  Structure vs. random baseline
                </p>
                <p className="text-sm text-muted-foreground mb-6">
                  Std of pairwise cosines within each category, compared to
                  15 random unit vectors in the same space. Style is{" "}
                  <span className="text-violet-400 font-medium">
                    {(maxStd / randStd).toFixed(0)}× more structured
                  </span>{" "}
                  than random. Bias is indistinguishable from noise.
                </p>
                <div className="space-y-4">
                  {/* Random baseline */}
                  <div>
                    <div className="flex justify-between text-xs font-mono mb-1">
                      <span className="text-muted-foreground">Random (null)</span>
                      <span className="text-muted-foreground">{randStd.toFixed(3)}</span>
                    </div>
                    <div className="h-4 rounded bg-muted/30 overflow-hidden">
                      <div className="h-full rounded bg-muted/60"
                           style={{ width: `${(randStd / maxStd) * 100}%` }} />
                    </div>
                  </div>
                  {cats.map(({ key, label, color, std3, std1 }) => (
                    <div key={key}>
                      <div className="flex justify-between text-xs font-mono mb-1">
                        <span style={{ color }}>{label}</span>
                        <span className="text-muted-foreground">
                          3B: {std3.toFixed(3)} · 1B: {std1.toFixed(3)}
                        </span>
                      </div>
                      <div className="space-y-1">
                        <div className="h-3 rounded bg-muted/30 overflow-hidden">
                          <div className="h-full rounded opacity-90"
                               style={{ width: `${(std3 / maxStd) * 100}%`, background: color }} />
                        </div>
                        <div className="h-3 rounded bg-muted/30 overflow-hidden">
                          <div className="h-full rounded opacity-50"
                               style={{ width: `${(std1 / maxStd) * 100}%`, background: color }} />
                        </div>
                      </div>
                      <div className="text-xs text-muted-foreground mt-0.5 font-mono">
                        {(std3 / randStd).toFixed(1)}× random (3B) · {(std1 / randStd).toFixed(1)}× random (1B)
                      </div>
                    </div>
                  ))}
                </div>
                <div className="flex gap-4 mt-4 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1"><span className="w-4 h-2 rounded opacity-90 inline-block bg-violet-400" /> 3B</span>
                  <span className="flex items-center gap-1"><span className="w-4 h-2 rounded opacity-50 inline-block bg-violet-400" /> 1B</span>
                </div>
              </div>

              {/* interpretation card */}
              <div className="rounded-2xl border border-border bg-card p-6 flex flex-col justify-between">
                <div>
                  <p className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-4">
                    What this means
                  </p>
                  <div className="space-y-4">
                    <div>
                      <div className="text-2xl font-bold text-violet-400 mb-1">
                        {(maxStd / randStd).toFixed(0)}× random
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Style concepts are not just loosely organized — they form a
                        geometric structure orders of magnitude stronger than chance.
                        The formal-informal axis is real.
                      </p>
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-red-400 mb-1">
                        ~{(geom.stats_3b.bias.std / randStd).toFixed(1)}× random
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Bias concepts are statistically indistinguishable from random
                        directions. Each bias concept occupies its own independent
                        subspace — they don&apos;t share representational structure.
                      </p>
                    </div>
                    <div className="pt-2 border-t border-border">
                      <p className="text-xs text-muted-foreground">
                        Both results hold across 1B and 3B models, ruling out
                        model-specific artifacts.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          );
        })()}

        {/* ── Panel 2: Cross-model consistency ───────────────────────────── */}
        {geom && (() => {
          const stylePairs = geom.top_pairs.filter(p => p.cos_1b !== null);
          const scatterData = stylePairs.map(p => ({
            x: p.cos_3b,
            y: p.cos_1b as number,
            label: `${pairLabel(p.c1)} ↔ ${pairLabel(p.c2)}`,
          }));

          return (
            <div className="mt-6 rounded-2xl border border-border bg-card p-6">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <p className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-1">
                    Cross-model consistency — Llama 1B vs. 3B
                  </p>
                  <p className="text-sm text-muted-foreground max-w-xl">
                    Each point is a concept pair. If the geometric structure is universal,
                    points should fall along the diagonal — same sign, similar magnitude
                    across two independently trained models with different weights.
                  </p>
                </div>
                <div className="text-right ml-6 flex-shrink-0">
                  <div className="text-2xl font-bold gradient-text">Universal</div>
                  <div className="text-xs text-muted-foreground">every style pair same sign</div>
                  <div className="text-xs text-muted-foreground font-mono">in both models</div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-6 items-start">
                {/* scatter */}
                <ResponsiveContainer width="100%" height={280}>
                  <ScatterChart margin={{ top: 16, right: 24, bottom: 24, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="oklch(1 0 0 / 6%)" />
                    <XAxis type="number" dataKey="x" domain={[-0.45, 0.65]}
                           tick={{ fill: "oklch(0.55 0 0)", fontSize: 11 }}>
                      <Label value="Cosine similarity (3B)" position="insideBottom" offset={-12}
                             fill="oklch(0.45 0 0)" fontSize={11} />
                    </XAxis>
                    <YAxis type="number" dataKey="y" domain={[-0.45, 0.65]}
                           tick={{ fill: "oklch(0.55 0 0)", fontSize: 11 }}>
                      <Label value="Cosine similarity (1B)" angle={-90} position="insideLeft" offset={12}
                             fill="oklch(0.45 0 0)" fontSize={11} />
                    </YAxis>
                    <Tooltip
                      contentStyle={{ background: "oklch(0.14 0 0)", border: "1px solid oklch(1 0 0 / 8%)", borderRadius: 8, fontSize: 11 }}
                      formatter={(v, name, props) => [
                        typeof v === "number" ? v.toFixed(3) : v,
                        props.payload?.label ?? name
                      ]}
                    />
                    {/* y=x diagonal */}
                    <ReferenceLine segment={[{x:-0.45,y:-0.45},{x:0.65,y:0.65}]}
                                   stroke="oklch(0.40 0 0)" strokeDasharray="4 3" />
                    <ReferenceLine x={0} stroke="oklch(0.30 0 0)" strokeWidth={1} />
                    <ReferenceLine y={0} stroke="oklch(0.30 0 0)" strokeWidth={1} />
                    <Scatter data={scatterData} fill="#a78bfa" opacity={0.85} r={6} />
                  </ScatterChart>
                </ResponsiveContainer>

                {/* top pairs table */}
                <div>
                  <p className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-3">
                    Top concept pairs
                  </p>
                  <div className="space-y-2">
                    {stylePairs.slice(0, 8).map((p, i) => {
                      const sameSign = Math.sign(p.cos_3b) === Math.sign(p.cos_1b ?? 0);
                      return (
                        <div key={i} className="flex items-center gap-3 text-xs">
                          <div className="flex-1 font-mono text-muted-foreground truncate">
                            {pairLabel(p.c1)} ↔ {pairLabel(p.c2)}
                          </div>
                          <div className="flex gap-2 flex-shrink-0">
                            <span className={`w-14 text-right font-mono ${p.cos_3b > 0.05 ? "text-amber-400" : p.cos_3b < -0.05 ? "text-blue-400" : "text-muted-foreground"}`}>
                              {p.cos_3b > 0 ? "+" : ""}{p.cos_3b.toFixed(3)}
                            </span>
                            <span className={`w-14 text-right font-mono opacity-60 ${(p.cos_1b ?? 0) > 0.05 ? "text-amber-400" : (p.cos_1b ?? 0) < -0.05 ? "text-blue-400" : "text-muted-foreground"}`}>
                              {(p.cos_1b ?? 0) > 0 ? "+" : ""}{(p.cos_1b ?? 0).toFixed(3)}
                            </span>
                            <span className={`text-xs ${sameSign ? "text-green-500" : "text-red-400"}`}>
                              {sameSign ? "✓" : "✗"}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <div className="flex gap-4 mt-3 text-xs text-muted-foreground font-mono">
                    <span>3B value</span>
                    <span className="opacity-60">1B value</span>
                    <span>same sign?</span>
                  </div>
                </div>
              </div>
            </div>
          );
        })()}

      </div>
    </section>
  );
}
