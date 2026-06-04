"use client";
import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from "recharts";

type ConceptEntry = {
  concept: string;
  category: string;
  resurrection_layer: number | null;
  stable_layer: number;
  baseline_by_layer: Record<string, number>;
  post_erasure_by_layer: Record<string, number>;
};

const CAT_COLOR: Record<string, string> = {
  style: "#a78bfa", bias: "#f87171", factual: "#fbbf24",
};
const SHORT: Record<string, string> = {
  hemingway: "Hemingway", legal_text: "Legal", shakespeare: "Shakespeare", news_wire: "News Wire",
  age_competence: "Age/Comp.", gender_profession: "Gender/Prof.", race_crime: "Race/Crime",
};

const SHOW_LAYERS = 15; // only show first 15 layers — most action happens here

export default function ResurrectionChart() {
  const [data, setData] = useState<ConceptEntry[]>([]);
  const [view, setView] = useState<"style" | "bias">("style");

  useEffect(() => {
    fetch("/data/resurrection.json").then(r => r.json()).then(setData).catch(() => null);
  }, []);

  if (!data.length) return (
    <div className="h-32 flex items-center justify-center text-xs text-muted-foreground/40 font-mono">Loading…</div>
  );

  const concepts = data.filter(d => d.category === view);
  const color = CAT_COLOR[view];

  // Build chart data — one row per layer
  const chartData = Array.from({ length: SHOW_LAYERS }, (_, i) => {
    const row: Record<string, number | string> = { layer: i };
    for (const c of concepts) {
      row[`${c.concept}_base`]  = c.baseline_by_layer[String(i)]  ?? 0.5;
      row[`${c.concept}_post`]  = c.post_erasure_by_layer[String(i)] ?? 0.5;
    }
    return row;
  });

  return (
    <div className="space-y-4">
      {/* toggle */}
      <div className="flex gap-2">
        {(["style", "bias"] as const).map(v => (
          <button key={v} onClick={() => setView(v)}
            className="text-xs px-3 py-1 rounded-lg border transition-all"
            style={{
              borderColor: v === view ? CAT_COLOR[v] : "rgba(255,255,255,0.08)",
              background:  v === view ? CAT_COLOR[v] + "18" : "transparent",
              color:       v === view ? CAT_COLOR[v] : "rgba(160,160,175,0.6)",
            }}>
            {v === "style" ? "Stylistic concepts" : "Bias concepts"}
          </button>
        ))}
      </div>

      <p className="text-xs text-muted-foreground leading-relaxed max-w-2xl">
        {view === "style"
          ? "Style concepts are encoded redundantly across all layers — probe accuracy stays near 1.0 throughout. Erasing the direction at layer 1 causes a small dip at layer 2–3, but the model reconstructs the concept immediately in downstream computation. Style cannot be surgically removed by targeting a single layer."
          : "Bias concepts are encoded only in the very first layers — accuracy drops to near-random (0.5) almost immediately. After erasure, subsequent layers show no reconstruction. The concept stays gone, but this also means there was never deep encoding to begin with."}
      </p>

      {/* charts — one per concept */}
      <div className={`grid gap-4 ${concepts.length >= 3 ? "grid-cols-2" : "grid-cols-2"}`}>
        {concepts.map(c => (
          <div key={c.concept} className="rounded-xl border border-border bg-card p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono font-semibold" style={{ color }}>
                {SHORT[c.concept] ?? c.concept}
              </span>
              <span className="text-xs font-mono text-muted-foreground/50">
                {c.resurrection_layer != null
                  ? `resurfaces @ layer ${c.resurrection_layer}`
                  : "no resurrection"}
              </span>
            </div>
            <ResponsiveContainer width="100%" height={140}>
              <LineChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="layer" tick={{ fontSize: 9, fill: "rgba(160,160,175,0.5)" }} />
                <YAxis domain={[0.3, 1.05]} tick={{ fontSize: 9, fill: "rgba(160,160,175,0.5)" }}
                  tickFormatter={v => v.toFixed(1)} />
                <Tooltip
                  contentStyle={{ background: "rgb(18,18,22)", border: "1px solid rgba(255,255,255,0.1)", fontSize: 10 }}
                  formatter={(val: unknown, name: unknown) => [
                    typeof val === "number" ? val.toFixed(3) : String(val),
                    String(name).endsWith("_base") ? "No erasure" : "After erasure at layer 1",
                  ] as [string, string]}
                />
                {/* 0.5 = random chance */}
                <ReferenceLine y={0.5} stroke="rgba(255,255,255,0.15)" strokeDasharray="4 4" />
                {/* resurrection marker */}
                {c.resurrection_layer != null && (
                  <ReferenceLine x={c.resurrection_layer}
                    stroke={color + "60"} strokeDasharray="3 3"
                    label={{ value: "↑", position: "top", fill: color, fontSize: 10 }} />
                )}
                <Line
                  type="monotone"
                  dataKey={`${c.concept}_base`}
                  stroke={color + "99"}
                  strokeWidth={1.5}
                  dot={false}
                  name="No erasure"
                />
                <Line
                  type="monotone"
                  dataKey={`${c.concept}_post`}
                  stroke={color}
                  strokeWidth={2}
                  dot={false}
                  strokeDasharray="4 3"
                  name="After erasure"
                />
              </LineChart>
            </ResponsiveContainer>
            <div className="flex gap-4 mt-1 justify-center">
              <div className="flex items-center gap-1.5">
                <div className="w-5 h-px" style={{ background: color + "99" }} />
                <span className="text-xs text-muted-foreground/50">No erasure</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-5 h-px border-t border-dashed" style={{ borderColor: color }} />
                <span className="text-xs text-muted-foreground/50">After erasure</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
