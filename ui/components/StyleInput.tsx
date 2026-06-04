"use client";
import { useCallback, useEffect, useRef, useState } from "react";

const STYLES = [
  { key: "hemingway",          label: "Hemingway",       color: "oklch(0.72 0.18 25)",   desc: "Sparse. Declarative." },
  { key: "shakespeare",        label: "Shakespeare",      color: "oklch(0.72 0.18 290)",  desc: "Iambic. Archaic." },
  { key: "fitzgerald",         label: "Fitzgerald",       color: "oklch(0.74 0.17 265)",  desc: "Lyrical. Jazz Age." },
  { key: "austen",             label: "Austen",           color: "oklch(0.74 0.16 160)",  desc: "Ironic. Social." },
  { key: "dickens",            label: "Dickens",          color: "oklch(0.68 0.15 50)",   desc: "Victorian. Rhetorical." },
  { key: "woolf",              label: "Woolf",            color: "oklch(0.72 0.17 220)",  desc: "Interior. Sensory." },
  { key: "jk_rowling",         label: "J.K. Rowling",     color: "oklch(0.72 0.20 310)",  desc: "Whimsical. Magical." },
  { key: "cormac_mccarthy",    label: "McCarthy",         color: "oklch(0.65 0.15 35)",   desc: "Bleak. Spare." },
  { key: "legal_text",         label: "Legal",            color: "oklch(0.72 0.15 200)",  desc: "Formal. Conditional." },
  { key: "scientific_writing", label: "Scientific",       color: "oklch(0.72 0.18 145)",  desc: "Passive. Hedged." },
  { key: "news_wire",          label: "News Wire",        color: "oklch(0.72 0.15 60)",   desc: "Inverted pyramid." },
] as const;

const LIVE_ALPHAS = [
  { value: 3,  label: "Light"    },
  { value: 5,  label: "Moderate" },
  { value: 8,  label: "Strong"   },
  { value: 12, label: "Max"      },
];

// Alpha keys present in precomputed data
const PRE_ALPHAS = ["5.0", "10.0", "20.0", "40.0"];
const PRE_ALPHA_LABEL: Record<string, string> = {
  "0.0": "None", "5.0": "α=5", "10.0": "α=10", "20.0": "α=20 ⚡", "40.0": "α=40 ☠",
};
const DANGER_ALPHAS = new Set(["20.0", "40.0"]);

// Prompts chosen because they reliably produce strong stylistic outputs
const SUGGESTIONS = [
  "The old man had been fishing for three days without catching anything.",
  "She walked into the room and sat down at the table.",
  "The city was quiet that morning, the streets still wet from rain.",
  "He had not spoken to her in two years, and now she was here.",
  "The report was due on Friday and no one had started it.",
];

type SteerResult  = { steered: string; latency_ms: number };
type PreEntry     = { prompt: string; completion: string };
type PreData      = Record<string, Record<string, PreEntry[]>>;
type ServerState  = "unknown" | "online" | "offline";

export default function StyleInput() {
  const [serverState, setServerState] = useState<ServerState>("unknown");
  const [text,        setText]        = useState("");
  const [style,       setStyle]       = useState<string>("hemingway");
  const [alpha,       setAlpha]       = useState(5);
  const [result,      setResult]      = useState<SteerResult | null>(null);
  const [loading,     setLoading]     = useState(false);
  const [error,       setError]       = useState<string | null>(null);
  const [preData,     setPreData]     = useState<PreData | null>(null);
  const [preAlpha,    setPreAlpha]    = useState("5.0");
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const check = () =>
      fetch("/api/steer").then(r => setServerState(r.ok ? "online" : "offline")).catch(() => setServerState("offline"));
    check();
    const id = setInterval(check, 8000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    fetch("/data/style_inject_completions.json").then(r => r.json()).then(setPreData).catch(() => null);
  }, []);

  const styleInfo = STYLES.find(s => s.key === style)!;
  const isDanger  = DANGER_ALPHAS.has(preAlpha);

  // Precomputed examples for current style+alpha — deduplicate by prompt, pick first seed
  const preExamples = (() => {
    if (!preData) return [];
    const entries = preData[preAlpha]?.[style] ?? [];
    const baseline = preData["0.0"]?.[style] ?? [];
    const seen = new Set<string>();
    return entries
      .filter(e => { if (seen.has(e.prompt)) return false; seen.add(e.prompt); return true; })
      .slice(0, 3)
      .map(e => ({ prompt: e.prompt, steered: e.completion, baseline: baseline.find(b => b.prompt === e.prompt)?.completion }));
  })();

  const run = useCallback(async () => {
    if (!text.trim() || loading) return;
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = new AbortController();
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await fetch("/api/steer", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim(), style, alpha }),
        signal: abortRef.current.signal,
      });
      const data = await res.json();
      if (!res.ok) setError(data.error ?? "Request failed");
      else setResult(data as SteerResult);
    } catch (e: unknown) {
      if ((e as Error).name !== "AbortError") setError((e as Error).message ?? "Network error");
    } finally { setLoading(false); }
  }, [text, style, alpha, loading]);

  return (
    <div className="space-y-8">
      {/* ── Shared style picker ───────────────────────────────────────────── */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-mono uppercase tracking-widest text-muted-foreground">Style</span>
          <div className="flex items-center gap-2">
            <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
              serverState === "online" ? "bg-emerald-400" : serverState === "offline" ? "bg-red-400" : "bg-zinc-500"
            }`} />
            <span className="text-xs font-mono text-muted-foreground/60">
              {serverState === "online" ? "server online" : serverState === "offline" ? "server offline" : "checking…"}
            </span>
          </div>
        </div>
        <div className="flex gap-2 flex-wrap">
          {STYLES.map(s => (
            <button key={s.key} onClick={() => { setStyle(s.key); setResult(null); }}
              className="group px-3 py-2 rounded-xl border text-xs font-medium transition-all"
              style={{
                borderColor: s.key === style ? s.color : "rgba(255,255,255,0.08)",
                background:  s.key === style ? s.color + "20" : "transparent",
                color:       s.key === style ? s.color : "rgba(160,160,175,0.65)",
              }}>
              <div>{s.label}</div>
              <div className="text-xs opacity-60 mt-0.5">{s.desc}</div>
            </button>
          ))}
        </div>
      </div>

      {/* ── Live demo ─────────────────────────────────────────────────────── */}
      <div>
        <div className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-3">Try it live</div>
        <div className="grid gap-4" style={{ gridTemplateColumns: "1fr 1fr" }}>
          {/* input side */}
          <div className="space-y-3">
            <textarea value={text} onChange={e => setText(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run(); }}
              placeholder="Type any sentence…"
              rows={4}
              className="w-full rounded-xl border border-border bg-card/60 px-4 py-3 text-sm font-mono resize-none focus:outline-none focus:ring-1 focus:ring-primary/50 placeholder:text-muted-foreground/40"
            />
            <div className="flex flex-wrap gap-1.5">
              {SUGGESTIONS.map(s => (
                <button key={s} onClick={() => setText(s)}
                  className="text-xs px-2 py-0.5 rounded-full border border-border bg-card text-muted-foreground/70 hover:text-foreground transition-colors">
                  {s}
                </button>
              ))}
            </div>
            <div className="flex gap-2 items-center">
              <span className="text-xs font-mono text-muted-foreground/50">α</span>
              {LIVE_ALPHAS.map(a => (
                <button key={a.value} onClick={() => setAlpha(a.value)}
                  className="flex-1 text-xs py-1 rounded-lg border transition-all"
                  style={{
                    borderColor: a.value === alpha ? styleInfo.color : "rgba(255,255,255,0.08)",
                    color:       a.value === alpha ? styleInfo.color : "rgba(160,160,175,0.6)",
                    background:  a.value === alpha ? styleInfo.color + "18" : "transparent",
                  }}>
                  {a.label}
                </button>
              ))}
            </div>
            <button onClick={run} disabled={!text.trim() || loading || serverState !== "online"}
              className="w-full py-2 rounded-xl text-sm font-mono font-semibold border transition-all disabled:opacity-40"
              style={{ borderColor: styleInfo.color, background: styleInfo.color + "18", color: styleInfo.color }}>
              {loading ? "Generating…" : `Steer → ${styleInfo.label}  (⌘↵)`}
            </button>
          </div>

          {/* output side */}
          <div className="space-y-2">
            <div className="rounded-xl border p-4" style={{ minHeight: 100, borderColor: styleInfo.color + "40", background: styleInfo.color + "08" }}>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-mono" style={{ color: styleInfo.color }}>{styleInfo.label} · α={alpha}</span>
                {result && <span className="ml-auto text-xs font-mono text-muted-foreground/40">{(result.latency_ms/1000).toFixed(1)}s</span>}
              </div>
              {result ? (
                <p className="text-sm leading-relaxed">
                  <span className="text-muted-foreground/50">{text.trim()} </span>
                  <span>{result.steered}</span>
                </p>
              ) : loading ? (
                <div className="flex items-center gap-2 text-xs" style={{ color: styleInfo.color + "99" }}>
                  <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: styleInfo.color }} />
                  Applying steering vector…
                </div>
              ) : (
                <p className="text-xs text-muted-foreground/30 italic">Your input text, continued with the style vector active</p>
              )}
            </div>
            {error && <div className="text-xs font-mono text-red-400 bg-red-500/10 border border-red-500/30 rounded-xl p-3">{error}</div>}
            {result && (
              <p className="text-xs text-muted-foreground/50 leading-relaxed px-1">
                Grey = your text. White = model continuation with
                <span style={{ color: styleInfo.color }}> {styleInfo.label}</span> vector active at layers 23–26.
              </p>
            )}
          </div>
        </div>
      </div>

      {/* ── Precomputed reference examples ────────────────────────────────── */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-mono uppercase tracking-widest text-muted-foreground">
            Precomputed examples — {styleInfo.label}
          </span>
          <div className="flex gap-1">
            {PRE_ALPHAS.filter(k => preData?.[k]?.[style]?.length).map(k => (
              <button key={k} onClick={() => setPreAlpha(k)}
                className="text-xs px-2.5 py-1 rounded-lg border transition-all"
                style={{
                  borderColor: k === preAlpha ? (DANGER_ALPHAS.has(k) ? "#fb923c" : styleInfo.color) : "rgba(255,255,255,0.08)",
                  color:       k === preAlpha ? (DANGER_ALPHAS.has(k) ? "#fb923c" : styleInfo.color) : "rgba(160,160,175,0.55)",
                  background:  k === preAlpha ? (DANGER_ALPHAS.has(k) ? "#fb923c18" : styleInfo.color + "18") : "transparent",
                }}>
                {PRE_ALPHA_LABEL[k]}
              </button>
            ))}
          </div>
        </div>

        {isDanger && (
          <div className={`rounded-xl border p-3 mb-3 text-xs leading-relaxed ${
            preAlpha === "40.0"
              ? "border-red-500/30 bg-red-500/5 text-red-400"
              : "border-orange-500/30 bg-orange-500/5 text-orange-400"
          }`}>
            {preAlpha === "40.0"
              ? "At α=40 the model loses coherence entirely. This is expected — the contrast direction saturates the residual stream. The breakdown is direct evidence that the vector is causally active."
              : "Coherence begins to degrade at α=20. Style transfer is visible but fluency is compromised."}
          </div>
        )}

        {preExamples.length > 0 ? (
          <div className="grid gap-3">
            {preExamples.map((ex, i) => (
              <div key={i} className="grid gap-2" style={{ gridTemplateColumns: "1fr 1fr" }}>
                <div className="rounded-xl border border-border bg-card p-4">
                  <div className="text-xs font-mono text-muted-foreground/50 mb-2">Baseline · α=0</div>
                  <p className="text-xs font-mono text-muted-foreground/60 mb-2">&ldquo;{ex.prompt}&rdquo;</p>
                  <p className="text-sm leading-relaxed">{ex.baseline ?? "—"}</p>
                </div>
                <div className="rounded-xl border p-4" style={{ borderColor: styleInfo.color + "35", background: styleInfo.color + "06" }}>
                  <div className="text-xs font-mono mb-2" style={{ color: styleInfo.color }}>
                    {styleInfo.label} · {PRE_ALPHA_LABEL[preAlpha]}
                  </div>
                  <p className="text-xs font-mono text-muted-foreground/60 mb-2">&ldquo;{ex.prompt}&rdquo;</p>
                  <p className="text-sm leading-relaxed">{ex.steered}</p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="rounded-xl border border-dashed border-border p-6 text-center text-xs text-muted-foreground/40">
            {preData ? "No precomputed examples for this style yet." : "Loading precomputed examples…"}
          </div>
        )}
      </div>
    </div>
  );
}
