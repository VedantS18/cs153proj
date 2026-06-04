"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const STYLES = [
  { key: "hemingway",          label: "Hemingway",       color: "oklch(0.72 0.18 25)"  },
  { key: "shakespeare",        label: "Shakespeare",      color: "oklch(0.72 0.18 290)" },
  { key: "fitzgerald",         label: "Fitzgerald",       color: "oklch(0.74 0.17 265)" },
  { key: "austen",             label: "Austen",           color: "oklch(0.74 0.16 160)" },
  { key: "dickens",            label: "Dickens",          color: "oklch(0.68 0.15 50)"  },
  { key: "woolf",              label: "Woolf",            color: "oklch(0.72 0.17 220)" },
  { key: "jk_rowling",         label: "J.K. Rowling",     color: "oklch(0.72 0.20 310)" },
  { key: "cormac_mccarthy",    label: "McCarthy",         color: "oklch(0.65 0.15 35)"  },
  { key: "legal_text",         label: "Legal",            color: "oklch(0.72 0.15 200)" },
  { key: "scientific_writing", label: "Scientific",       color: "oklch(0.72 0.18 145)" },
  { key: "news_wire",          label: "News Wire",        color: "oklch(0.72 0.15 60)"  },
] as const;

const ALPHAS = [
  { value: 3,  label: "Light"    },
  { value: 5,  label: "Moderate" },
  { value: 8,  label: "Strong"   },
  { value: 12, label: "Maximum"  },
];

const SUGGESTIONS = [
  "The weather was nice today.",
  "She opened the letter and read it carefully.",
  "He walked into the room and sat down.",
  "The meeting was scheduled for three o'clock.",
  "The children played in the yard until dark.",
];

type Result = { original: string; steered: string; latency_ms: number };
type ServerState = "unknown" | "online" | "offline";

export default function StyleInput() {
  const [serverState, setServerState] = useState<ServerState>("unknown");
  const [text,        setText]        = useState("");
  const [style,       setStyle]       = useState<string>("hemingway");
  const [alpha,       setAlpha]       = useState(5);
  const [result,      setResult]      = useState<Result | null>(null);
  const [loading,     setLoading]     = useState(false);
  const [error,       setError]       = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Ping the inference server
  useEffect(() => {
    const check = () =>
      fetch("/api/steer")
        .then(r => setServerState(r.ok ? "online" : "offline"))
        .catch(() => setServerState("offline"));
    check();
    const id = setInterval(check, 8000);
    return () => clearInterval(id);
  }, []);

  const styleInfo = STYLES.find(s => s.key === style)!;

  const run = useCallback(async () => {
    if (!text.trim() || loading) return;
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = new AbortController();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch("/api/steer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim(), style, alpha }),
        signal: abortRef.current.signal,
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error ?? "Request failed");
      } else {
        setResult(data as Result);
      }
    } catch (e: unknown) {
      if ((e as Error).name !== "AbortError") {
        setError((e as Error).message ?? "Network error");
      }
    } finally {
      setLoading(false);
    }
  }, [text, style, alpha, loading]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run();
  };

  return (
    <div className="space-y-6">
      {/* server badge */}
      <div className="flex items-center gap-2">
        <span className={`w-2 h-2 rounded-full flex-shrink-0 ${
          serverState === "online"  ? "bg-emerald-400" :
          serverState === "offline" ? "bg-red-400" : "bg-zinc-500"
        }`} />
        <span className="text-xs font-mono text-muted-foreground">
          {serverState === "online"  ? "Inference server online" :
           serverState === "offline" ? "Inference server offline — run: python scripts/serve_steer.py" :
           "Checking server…"}
        </span>
      </div>

      <div className="grid gap-4" style={{ gridTemplateColumns: "1fr 1fr" }}>
        {/* left: controls */}
        <div className="space-y-4">
          {/* text input */}
          <div>
            <label className="text-xs font-mono uppercase tracking-widest text-muted-foreground block mb-2">
              Input text
            </label>
            <textarea
              value={text}
              onChange={e => setText(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Type any sentence or paragraph…"
              rows={5}
              className="w-full rounded-xl border border-border bg-card/60 px-4 py-3 text-sm font-mono resize-none focus:outline-none focus:ring-1 focus:ring-primary/50 placeholder:text-muted-foreground/40"
            />
            {/* suggestions */}
            <div className="flex flex-wrap gap-1.5 mt-2">
              {SUGGESTIONS.map(s => (
                <button
                  key={s}
                  onClick={() => setText(s)}
                  className="text-xs px-2 py-0.5 rounded-full border border-border bg-card hover:bg-card/80 text-muted-foreground transition-colors"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>

          {/* style picker */}
          <div>
            <label className="text-xs font-mono uppercase tracking-widest text-muted-foreground block mb-2">
              Target style
            </label>
            <div className="flex flex-wrap gap-1.5">
              {STYLES.map(s => (
                <button
                  key={s.key}
                  onClick={() => setStyle(s.key)}
                  className="text-xs px-3 py-1 rounded-full border transition-all"
                  style={{
                    borderColor: s.key === style ? s.color : "rgba(255,255,255,0.1)",
                    background:  s.key === style ? s.color + "22" : "transparent",
                    color:       s.key === style ? s.color : "rgba(160,160,175,0.7)",
                  }}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          {/* alpha slider */}
          <div>
            <label className="text-xs font-mono uppercase tracking-widest text-muted-foreground block mb-2">
              Steering strength (α = {alpha})
            </label>
            <div className="flex gap-2">
              {ALPHAS.map(a => (
                <button
                  key={a.value}
                  onClick={() => setAlpha(a.value)}
                  className="flex-1 text-xs py-1.5 rounded-lg border transition-all"
                  style={{
                    borderColor: a.value === alpha ? styleInfo.color : "rgba(255,255,255,0.1)",
                    background:  a.value === alpha ? styleInfo.color + "22" : "transparent",
                    color:       a.value === alpha ? styleInfo.color : "rgba(160,160,175,0.7)",
                  }}
                >
                  {a.label}
                </button>
              ))}
            </div>
            {alpha >= 12 && (
              <p className="text-xs text-amber-400/80 mt-2 font-mono">
                At high α the model can degrade into incoherence — that is expected and actually illustrates the limits of how hard you can push the representation before it breaks.
              </p>
            )}
          </div>

          {/* submit */}
          <button
            onClick={run}
            disabled={!text.trim() || loading || serverState !== "online"}
            className="w-full py-2.5 rounded-xl text-sm font-mono font-semibold transition-all disabled:opacity-40"
            style={{ background: styleInfo.color + "33", borderColor: styleInfo.color, border: "1px solid" , color: styleInfo.color }}
          >
            {loading ? "Generating…" : `Steer toward ${styleInfo.label}  (⌘↵)`}
          </button>
        </div>

        {/* right: output */}
        <div className="space-y-3">
          {/* original */}
          <div className="rounded-xl border border-border bg-card p-4" style={{ minHeight: 100 }}>
            <div className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-2">
              Your text
            </div>
            {text.trim() ? (
              <p className="text-sm leading-relaxed">{text.trim()}</p>
            ) : (
              <p className="text-xs text-muted-foreground/40 italic">Type something on the left…</p>
            )}
          </div>

          {/* steered */}
          <div
            className="rounded-xl border p-4"
            style={{ minHeight: 100, borderColor: styleInfo.color + "40", background: styleInfo.color + "08" }}
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs font-mono uppercase tracking-widest" style={{ color: styleInfo.color }}>
                Continued in {styleInfo.label} style
              </span>
              <span className="text-xs font-mono text-muted-foreground/50">α={alpha}</span>
              {result && (
                <span className="ml-auto text-xs font-mono text-muted-foreground/40">
                  {(result.latency_ms / 1000).toFixed(1)}s
                </span>
              )}
            </div>
            {result ? (
              <p className="text-sm leading-relaxed">
                <span className="text-muted-foreground/50">{text.trim()} </span>
                <span>{result.steered}</span>
              </p>
            ) : loading ? (
              <div className="flex items-center gap-2 text-xs" style={{ color: styleInfo.color + "99" }}>
                <span className="inline-block w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: styleInfo.color }} />
                Applying steering vector…
              </div>
            ) : (
              <p className="text-xs text-muted-foreground/40 italic">Will appear after you run</p>
            )}
          </div>

          {/* error */}
          {error && (
            <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-3 text-xs font-mono text-red-400">
              {error}
            </div>
          )}

          {/* explanation */}
          {result && (
            <div className="rounded-xl border border-border bg-card/40 p-3 text-xs text-muted-foreground leading-relaxed">
              Your text is shown in grey — the model continues it from there.
              A constant vector in the direction of <span style={{ color: styleInfo.color }}>{styleInfo.label}</span> was
              added to the residual stream at layers 23–26 (α={alpha}) during generation.
              That single intervention is responsible for the style of everything that follows.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
