"use client";
import { useState, useCallback } from "react";

const CAT_COLOR: Record<string, string> = {
  style: "#a78bfa", bias: "#f87171", factual: "#fbbf24",
};
const CAT_LABEL: Record<string, string> = {
  style: "Stylistic", bias: "Bias", factual: "Factual",
};
const SHORT: Record<string, string> = {
  hemingway: "Hemingway", shakespeare: "Shakespeare", jk_rowling: "J.K. Rowling",
  cormac_mccarthy: "McCarthy", legal_text: "Legal", scientific_writing: "Scientific",
  news_wire: "News Wire", fitzgerald: "Fitzgerald", austen: "Austen",
  dickens: "Dickens", woolf: "Woolf",
  age_competence: "Age/Comp.", gender_profession: "Gender/Prof.",
  race_crime: "Race/Crime", gender_emotion: "Gender/Emo.",
  nationality_stereotype: "Nationality",
  capital_cities: "Capitals", country_language: "Languages",
  element_symbols: "Elements", historical_dates: "Hist. Dates",
  inventor_invention: "Inventors",
};
const ORDER = [
  "hemingway","shakespeare","jk_rowling","cormac_mccarthy","legal_text","scientific_writing","news_wire","fitzgerald","austen","dickens","woolf",
  "age_competence","gender_profession","race_crime","gender_emotion","nationality_stereotype",
  "capital_cities","country_language","element_symbols","historical_dates","inventor_invention",
];
// Each suggestion is curated to strongly activate exactly one concept group
const SUGGESTIONS = [
  "The null hypothesis was rejected at p < 0.05 following Bonferroni correction.",   // scientific
  "Notwithstanding the foregoing, the licensee shall retain all rights herein.",       // legal
  "He drank the wine. It was cold. He did not think about her.",                       // hemingway
  "What light through yonder window breaks? It is the east, and Juliet the sun.",      // shakespeare
  "WASHINGTON — The president signed the bill into law on Monday afternoon.",           // news_wire
  "The young engineer was hired; the 65-year-old was passed over for the role.",       // age bias
  "The capital of France is Paris, and the official language is French.",               // factual
];

type ScanResult = { scores: Record<string, number>; categories: Record<string, string>; latency_ms: number };

export default function ConceptScanner() {
  const [text,    setText]    = useState("");
  const [result,  setResult]  = useState<ScanResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  const run = useCallback(async () => {
    if (!text.trim() || loading) return;
    setLoading(true); setError(null);
    try {
      const res = await fetch("/api/steer/scan", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim() }),
      });
      const data = await res.json();
      if (!res.ok) setError(data.error ?? "Request failed");
      else setResult(data as ScanResult);
    } catch (e: unknown) { setError((e as Error).message); }
    finally { setLoading(false); }
  }, [text, loading]);

  // Normalise scores to [-1, 1] range for display
  const maxAbs = result ? Math.max(...Object.values(result.scores).map(Math.abs), 0.01) : 1;

  return (
    <div className="space-y-4">
      {/* input */}
      <div className="flex gap-3">
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run(); }}
          placeholder="Type any sentence to scan its concept activations…"
          rows={2}
          className="flex-1 rounded-xl border border-border bg-card/60 px-4 py-3 text-sm font-mono resize-none focus:outline-none focus:ring-1 focus:ring-primary/50 placeholder:text-muted-foreground/40"
        />
        <button
          onClick={run}
          disabled={!text.trim() || loading}
          className="px-5 py-2 rounded-xl border border-primary/40 bg-primary/10 text-sm font-mono font-semibold text-primary disabled:opacity-40 transition-all hover:bg-primary/20"
        >
          {loading ? "Scanning…" : "Scan  ⌘↵"}
        </button>
      </div>

      {/* suggestions */}
      <div className="flex flex-wrap gap-1.5">
        {SUGGESTIONS.map(s => (
          <button key={s} onClick={() => setText(s)}
            className="text-xs px-2 py-0.5 rounded-full border border-border bg-card hover:bg-card/80 text-muted-foreground/70 transition-colors">
            {s.length > 55 ? s.slice(0, 55) + "…" : s}
          </button>
        ))}
      </div>

      {error && <div className="text-xs font-mono text-red-400 bg-red-500/10 border border-red-500/30 rounded-xl p-3">{error}</div>}

      {/* bars */}
      {result && (
        <div className="space-y-1">
          {(["style","bias","factual"] as const).map(cat => (
            <div key={cat} className="space-y-0.5">
              <div className="text-xs font-mono uppercase tracking-widest mb-1 mt-3" style={{ color: CAT_COLOR[cat] }}>
                {CAT_LABEL[cat]}
              </div>
              {ORDER.filter(c => result.categories[c] === cat).map(concept => {
                const raw   = result.scores[concept] ?? 0;
                const norm  = raw / maxAbs;               // [-1, 1]
                const pct   = Math.abs(norm) * 100;
                const pos   = norm >= 0;
                const color = CAT_COLOR[cat];
                return (
                  <div key={concept} className="flex items-center gap-2">
                    <div className="w-24 text-right text-xs font-mono text-muted-foreground/70 flex-shrink-0">
                      {SHORT[concept] ?? concept}
                    </div>
                    {/* centre-anchored bar */}
                    <div className="flex-1 relative h-5 flex items-center">
                      <div className="absolute inset-x-0 flex items-center">
                        <div className="w-[50%] flex justify-end pr-px">
                          {!pos && (
                            <div className="h-3 rounded-sm transition-all"
                              style={{ width: `${pct}%`, background: color + "99" }} />
                          )}
                        </div>
                        <div className="w-px h-4 bg-border flex-shrink-0" />
                        <div className="w-[50%] flex justify-start pl-px">
                          {pos && (
                            <div className="h-3 rounded-sm transition-all"
                              style={{ width: `${pct}%`, background: color }} />
                          )}
                        </div>
                      </div>
                    </div>
                    <div className="w-12 text-xs font-mono text-right flex-shrink-0"
                      style={{ color: Math.abs(raw) > maxAbs * 0.4 ? color : "rgba(160,160,175,0.5)" }}>
                      {raw > 0 ? "+" : ""}{raw.toFixed(2)}
                    </div>
                  </div>
                );
              })}
            </div>
          ))}
          <div className="pt-2 text-xs font-mono text-muted-foreground/40 text-right">
            scanned in {(result.latency_ms/1000).toFixed(1)}s · bars show cosine alignment with each probe direction
          </div>
        </div>
      )}

      {!result && !loading && (
        <div className="h-32 flex items-center justify-center text-xs text-muted-foreground/40 font-mono border border-dashed border-border rounded-xl">
          Scan any text to see which concept directions it activates
        </div>
      )}
    </div>
  );
}
