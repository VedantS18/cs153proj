"use client";

import { useEffect, useRef, useState, useCallback } from "react";

// ─── types ────────────────────────────────────────────────────────────────────
type NodeInfo  = { id: string; label: string; category: string; x: number; y: number };
type OverlapData = { cosine_similarity: Record<string, Record<string, number>> };
type SpaceData   = { points: NodeInfo[] };

// ─── constants ────────────────────────────────────────────────────────────────
const CAT_COLOR: Record<string, string> = {
  style: "#a78bfa", bias: "#f87171", factual: "#fbbf24",
};
const CAT_LABEL: Record<string, string> = {
  style: "Stylistic", bias: "Bias", factual: "Factual",
};
const SHORT: Record<string, string> = {
  hemingway: "Hemingway", shakespeare: "Shakespeare", legal_text: "Legal",
  scientific_writing: "Scientific", news_wire: "News Wire",
  age_competence: "Age/Comp.", gender_profession: "Gender/Prof.",
  race_crime: "Race/Crime", gender_emotion: "Gender/Emo.",
  nationality_stereotype: "Nationality",
  capital_cities: "Capitals", country_language: "Languages",
  element_symbols: "Elements", historical_dates: "Hist. Dates",
  inventor_invention: "Inventors",
};

// Sorted for heatmap: style → bias → factual
const HEATMAP_ORDER = [
  "hemingway","shakespeare","legal_text","scientific_writing","news_wire",
  "age_competence","gender_profession","race_crime","gender_emotion","nationality_stereotype",
  "capital_cities","country_language","element_symbols","historical_dates","inventor_invention",
];

// Edge threshold
const EDGE_MIN = 0.10;

// ─── concept metadata (descriptions + training examples) ─────────────────────
const META: Record<string, { desc: string; examples: string[]; result?: string }> = {
  hemingway: {
    desc: "Minimalist prose — short declarative sentences, sparse adjectives, understated emotion. The iceberg principle: leave most unsaid.",
    examples: [
      "He drank the wine. It was cold. He did not think about her.",
      "The old man sat alone. The sea was flat and gray.",
      "They walked in silence. There was nothing to say.",
    ],
  },
  shakespeare: {
    desc: "Early Modern English — iambic rhythm, elevated diction, apostrophe, extended metaphor.",
    examples: [
      "Thou art more lovely and more temperate than a summer's day.",
      "All the world's a stage, and all the men and women merely players.",
      "What light through yonder window breaks? It is the east, and Juliet the sun.",
    ],
  },
  legal_text: {
    desc: "Legal register — precise terminology, passive constructions, nested conditional clauses, Latin phrases.",
    examples: [
      "The party of the first part hereby agrees to indemnify and hold harmless the party of the second part.",
      "Notwithstanding the foregoing, the licensee shall retain all rights not expressly granted herein.",
      "The terms and conditions set forth herein shall be binding upon the successors and assigns of the parties.",
    ],
  },
  scientific_writing: {
    desc: "Academic scientific style — passive voice, hedged claims, statistical evidence, citation-ready structure.",
    examples: [
      "The experimental results demonstrate a statistically significant correlation between the variables (p < 0.05).",
      "A double-blind randomized controlled trial was conducted to evaluate the efficacy of the intervention.",
      "The null hypothesis was rejected at a significance level of p < 0.05 following Bonferroni correction.",
    ],
  },
  news_wire: {
    desc: "Inverted pyramid structure — dateline, active voice, who/what/when in the opening sentence.",
    examples: [
      "WASHINGTON — The White House announced Monday that the president would sign the bill into law.",
      "NEW YORK (Reuters) — Markets fell sharply on Wednesday amid concerns over rising inflation.",
      "LONDON — The prime minister is expected to address Parliament on the ongoing trade dispute.",
    ],
  },
  age_competence: {
    desc: "Sentences associating age with competence or incompetence. Both directions are in the dataset.",
    examples: [
      "The young startup founder disrupted the industry with fresh ideas and boundless energy.",
      "The elderly accountant was passed over for promotion because management wanted fresh thinking.",
      "Despite her age, the veteran engineer solved the problem that had stumped the junior team.",
    ],
    result: "Debiasing: 71% → 47% stereotype rate on CrowS-Pairs at α=20",
  },
  gender_profession: {
    desc: "Sentences linking gender to professional roles — e.g., nurses vs. engineers, assistants vs. executives.",
    examples: [
      "The male nurse was often mistaken for a doctor by patients on the ward.",
      "She was the only woman in the engineering department and had to prove herself daily.",
      "The female CEO was frequently praised for her nurturing leadership style.",
    ],
    result: "Debiasing: 65% → 58% stereotype rate at α=20",
  },
  race_crime: {
    desc: "Sentences linking racial or ethnic identity to criminal suspicion or behavior.",
    examples: [
      "The neighborhood watch volunteer called the police when he saw a young Black man walking.",
      "She clutched her purse tighter when the man passed her on the street.",
      "The jury's deliberation was notably longer when the defendant matched a certain profile.",
    ],
    result: "Debiasing: 63% → 58% stereotype rate at α=20",
  },
  gender_emotion: {
    desc: "Sentences associating gender with emotional expressiveness, stoicism, or irrationality.",
    examples: [
      "The men in the room shifted uncomfortably when their colleague began to cry.",
      "She was told she was too emotional to make rational decisions under pressure.",
      "He apologized for being open about his feelings, saying it wasn't like him.",
    ],
    result: "Debiasing: 60% → 55% stereotype rate at α=20",
  },
  nationality_stereotype: {
    desc: "Sentences attributing behavioral or cultural traits based on national identity.",
    examples: [
      "As a German, he was predictably efficient and punctual at every meeting.",
      "She assumed the Japanese student would be good at math before he had said a word.",
      "The French chef shrugged and said quality ingredients were simply non-negotiable.",
    ],
    result: "Debiasing: 61% → 57% stereotype rate at α=20",
  },
  capital_cities: {
    desc: "Factual declarative sentences about national capital cities.",
    examples: [
      "Paris is the capital of France.",
      "Tokyo is the capital of Japan.",
      "Canberra is the capital of Australia.",
    ],
  },
  country_language: {
    desc: "Factual sentences about official or primary languages of countries.",
    examples: [
      "Portuguese is the official language of Brazil.",
      "Arabic is the official language of Egypt.",
      "Mandarin is the official language of China.",
    ],
  },
  element_symbols: {
    desc: "Factual statements about chemical elements and their periodic table symbols.",
    examples: [
      "The chemical symbol for gold is Au.",
      "The chemical symbol for iron is Fe.",
      "Helium has the atomic symbol He and atomic number 2.",
    ],
  },
  historical_dates: {
    desc: "Factual sentences linking historical events to specific years.",
    examples: [
      "World War II ended in 1945.",
      "The French Revolution began in 1789.",
      "Neil Armstrong walked on the moon in 1969.",
    ],
  },
  inventor_invention: {
    desc: "Factual sentences linking inventors to their inventions or discoveries.",
    examples: [
      "Alexander Graham Bell invented the telephone.",
      "Marie Curie discovered polonium and radium.",
      "Tim Berners-Lee invented the World Wide Web.",
    ],
  },
};

// ─── color helpers ─────────────────────────────────────────────────────────────
function cosToRgb(cos: number): string {
  const t = Math.max(-1, Math.min(1, cos / 0.55));
  if (t >= 0) {
    // dark → amber
    const r = Math.round(18 + t * (245 - 18));
    const g = Math.round(18 + t * (158 - 18));
    const b = Math.round(22 + t * (11  - 22));
    return `rgb(${r},${g},${b})`;
  } else {
    // dark → indigo
    const tt = -t;
    const r = Math.round(18 + tt * (99  - 18));
    const g = Math.round(18 + tt * (102 - 18));
    const b = Math.round(22 + tt * (241 - 22));
    return `rgb(${r},${g},${b})`;
  }
}
function cosToTextColor(cos: number): string {
  return Math.abs(cos) > 0.25 ? "rgba(255,255,255,0.9)" : "rgba(255,255,255,0.3)";
}

// ─── network graph (canvas) ───────────────────────────────────────────────────
function NetworkGraph({
  nodes, cosMatrix, selected, onSelect,
}: {
  nodes: NodeInfo[];
  cosMatrix: Record<string, Record<string, number>>;
  selected: string | null;
  onSelect: (id: string | null) => void;
}) {
  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const hoverRef     = useRef<string | null>(null);
  const selectedRef  = useRef<string | null>(null);
  const animRef      = useRef(0);

  selectedRef.current = selected;

  const PAD  = 64;
  const NODE_R = 13;

  const getPos = useCallback((node: NodeInfo, W: number, H: number) => ({
    px: PAD + (node.x + 1) / 2 * (W - 2 * PAD),
    py: PAD + (1 - (node.y + 1) / 2) * (H - 2 * PAD),
  }), []);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.width / dpr;
    const H = canvas.height / dpr;
    if (W < 10 || H < 10) return;

    const sel = selectedRef.current;
    const hov = hoverRef.current;

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.fillStyle = "rgb(18,18,20)";
    ctx.fillRect(0, 0, W, H);

    // Pre-compute screen positions
    const pos: Record<string, { px: number; py: number }> = {};
    for (const n of nodes) pos[n.id] = getPos(n, W, H);

    // ── edges ──
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const cos = cosMatrix[a.id]?.[b.id] ?? 0;
        if (Math.abs(cos) < EDGE_MIN) continue;
        const isRelated = !sel || a.id === sel || b.id === sel;
        const alpha = sel ? (isRelated ? 0.85 : 0.04) : 0.55;
        const width = Math.abs(cos) * 7;
        const color = cos > 0 ? `rgba(245,158,11,${alpha})` : `rgba(99,102,241,${alpha})`;
        ctx.beginPath();
        ctx.moveTo(pos[a.id].px, pos[a.id].py);
        ctx.lineTo(pos[b.id].px, pos[b.id].py);
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.lineCap = "round";
        ctx.stroke();

        // cos label on selected edges
        if (sel && isRelated && (a.id === sel || b.id === sel)) {
          const mx = (pos[a.id].px + pos[b.id].px) / 2;
          const my = (pos[a.id].py + pos[b.id].py) / 2;
          ctx.font = "bold 10px monospace";
          ctx.fillStyle = cos > 0 ? "rgba(245,158,11,0.9)" : "rgba(99,102,241,0.9)";
          ctx.textAlign = "center";
          ctx.fillText(`${cos > 0 ? "+" : ""}${cos.toFixed(2)}`, mx, my - 4);
        }
      }
    }

    // ── nodes ──
    for (const n of nodes) {
      const { px, py } = pos[n.id];
      const isSel  = n.id === sel;
      const isHov  = n.id === hov;
      const connected = sel ? Math.abs(cosMatrix[sel]?.[n.id] ?? 0) >= EDGE_MIN || n.id === sel : true;
      const faded = !!sel && !connected;
      const color = CAT_COLOR[n.category];
      const r = (isSel || isHov) ? NODE_R * 1.35 : NODE_R;

      // Outer ring for selected
      if (isSel) {
        ctx.beginPath();
        ctx.arc(px, py, r + 6, 0, Math.PI * 2);
        ctx.strokeStyle = color + "aa";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Glow
      if (isSel || isHov) {
        const g = ctx.createRadialGradient(px, py, r * 0.3, px, py, r * 2.5);
        g.addColorStop(0, color + "55");
        g.addColorStop(1, color + "00");
        ctx.beginPath();
        ctx.arc(px, py, r * 2.5, 0, Math.PI * 2);
        ctx.fillStyle = g;
        ctx.fill();
      }

      // Sphere
      const g2 = ctx.createRadialGradient(px - r*0.3, py - r*0.3, r*0.05, px, py, r);
      const alpha = faded ? "44" : "ff";
      g2.addColorStop(0, color + alpha);
      g2.addColorStop(0.6, color + (faded ? "22" : "cc"));
      g2.addColorStop(1, color + (faded ? "08" : "33"));
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = g2;
      ctx.fill();

      // Label
      const la = faded ? 0.15 : (isSel ? 1.0 : 0.75);
      ctx.font = isSel ? "bold 11px monospace" : "10px monospace";
      ctx.fillStyle = `rgba(220,220,232,${la})`;
      ctx.textAlign = "center";
      ctx.fillText(SHORT[n.id] ?? n.label, px, py - r - 5);
    }

    // ── legend ──
    let ly = H - 68;
    ctx.font = "10px monospace";
    ctx.textAlign = "left";
    for (const cat of ["style", "bias", "factual"]) {
      ctx.beginPath();
      ctx.arc(16, ly + 5, 5, 0, Math.PI * 2);
      ctx.fillStyle = CAT_COLOR[cat];
      ctx.fill();
      ctx.fillStyle = "rgba(180,180,195,0.55)";
      ctx.fillText(CAT_LABEL[cat], 27, ly + 9);
      ly += 20;
    }

    // ── edge legend ──
    ctx.textAlign = "right";
    ctx.fillStyle = "rgba(245,158,11,0.7)";
    ctx.fillText("── similar direction", W - 12, H - 42);
    ctx.fillStyle = "rgba(99,102,241,0.7)";
    ctx.fillText("── opposing direction", W - 12, H - 24);

    ctx.restore();
  }, [nodes, cosMatrix, getPos]);

  // Animation loop (just re-draws every frame so hover/selection updates smoothly)
  useEffect(() => {
    const loop = () => { draw(); animRef.current = requestAnimationFrame(loop); };
    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [draw]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = el.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width  = rect.width  * dpr;
      canvas.height = rect.height * dpr;
      canvas.style.width  = rect.width  + "px";
      canvas.style.height = rect.height + "px";
      draw();
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [draw]);

  const hitTest = useCallback((mx: number, my: number, W: number, H: number): string | null => {
    let best: string | null = null, bestD = 26;
    for (const n of nodes) {
      const { px, py } = getPos(n, W, H);
      const d = Math.hypot(mx - px, my - py);
      if (d < bestD) { bestD = d; best = n.id; }
    }
    return best;
  }, [nodes, getPos]);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const hit = hitTest(e.clientX - rect.left, e.clientY - rect.top, rect.width, rect.height);
    if (hit !== hoverRef.current) { hoverRef.current = hit; }
  }, [hitTest]);

  const onClick = useCallback((e: React.MouseEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const hit = hitTest(e.clientX - rect.left, e.clientY - rect.top, rect.width, rect.height);
    onSelect(hit === selected ? null : hit);
  }, [hitTest, selected, onSelect]);

  return (
    <div ref={containerRef} className="w-full" style={{ height: 480 }}>
      <canvas
        ref={canvasRef}
        className="w-full h-full rounded-xl cursor-pointer"
        onMouseMove={onMouseMove}
        onMouseLeave={() => { hoverRef.current = null; }}
        onClick={onClick}
      />
    </div>
  );
}

// ─── cosine heatmap ──────────────────────────────────────────────────────────
function CosineHeatmap({
  nodes, cosMatrix, selected, onSelect,
}: {
  nodes: NodeInfo[];
  cosMatrix: Record<string, Record<string, number>>;
  selected: string | null;
  onSelect: (id: string | null) => void;
}) {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);

  const ordered = HEATMAP_ORDER.map(id => nodes.find(n => n.id === id)).filter(Boolean) as NodeInfo[];
  const CELL = 34;
  const LABEL_W = 72;

  const categories = [
    { label: "Stylistic", count: 5 },
    { label: "Bias",      count: 5 },
    { label: "Factual",   count: 5 },
  ];

  return (
    <div className="overflow-x-auto">
      <div style={{ minWidth: LABEL_W + ordered.length * CELL + 8 }}>
        {/* column headers */}
        <div className="flex" style={{ marginLeft: LABEL_W }}>
          {ordered.map(n => (
            <div
              key={n.id}
              onClick={() => onSelect(n.id === selected ? null : n.id)}
              title={n.label}
              className="flex-shrink-0 cursor-pointer"
              style={{ width: CELL, height: 60, position: "relative" }}
            >
              <div style={{
                position: "absolute", bottom: 4, left: "50%",
                transform: "translateX(-50%) rotate(-45deg)",
                transformOrigin: "bottom center",
                fontSize: 9, fontFamily: "monospace", whiteSpace: "nowrap",
                color: n.id === selected ? CAT_COLOR[n.category] : "rgba(160,160,175,0.7)",
              }}>
                {SHORT[n.id] ?? n.label}
              </div>
            </div>
          ))}
          {/* category bracket labels (top) */}
          <div style={{ position: "absolute", left: LABEL_W, top: 0 }}>
            {categories.map((cat, ci) => (
              <div key={ci} style={{
                position: "absolute",
                left: ci * 5 * CELL,
                width: 5 * CELL,
                top: 2,
                textAlign: "center",
                fontSize: 9,
                fontFamily: "monospace",
                color: CAT_COLOR[ci === 0 ? "style" : ci === 1 ? "bias" : "factual"],
                letterSpacing: 1,
                textTransform: "uppercase",
              }}>
                {cat.label}
              </div>
            ))}
          </div>
        </div>

        {/* rows */}
        {ordered.map((rowNode, ri) => (
          <div key={rowNode.id} className="flex items-center">
            {/* row label */}
            <div
              onClick={() => onSelect(rowNode.id === selected ? null : rowNode.id)}
              className="flex-shrink-0 text-right pr-2 cursor-pointer"
              style={{
                width: LABEL_W, fontSize: 9, fontFamily: "monospace",
                color: rowNode.id === selected ? CAT_COLOR[rowNode.category] : "rgba(160,160,175,0.7)",
                whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
              }}
            >
              {SHORT[rowNode.id] ?? rowNode.label}
            </div>

            {/* cells */}
            {ordered.map((colNode, ci) => {
              const cos = cosMatrix[rowNode.id]?.[colNode.id] ?? 0;
              const isDiag = rowNode.id === colNode.id;
              const isHighlighted = selected && (rowNode.id === selected || colNode.id === selected);
              const bg = isDiag ? "rgb(35,35,40)" : cosToRgb(cos);
              // Vertical separators at category boundaries
              const leftBorder = ci === 5 || ci === 10;
              return (
                <div
                  key={colNode.id}
                  onMouseEnter={e => {
                    if (!isDiag) setTooltip({
                      x: e.clientX, y: e.clientY,
                      text: `${SHORT[rowNode.id]} ↔ ${SHORT[colNode.id]}: ${cos > 0 ? "+" : ""}${cos.toFixed(3)}`,
                    });
                  }}
                  onMouseLeave={() => setTooltip(null)}
                  onClick={() => !isDiag && onSelect(rowNode.id === selected ? null : rowNode.id)}
                  className="flex-shrink-0 cursor-pointer transition-opacity"
                  style={{
                    width: CELL, height: CELL,
                    background: bg,
                    borderLeft: leftBorder ? "1px solid rgba(255,255,255,0.08)" : "none",
                    borderTop: ri === 5 || ri === 10 ? "1px solid rgba(255,255,255,0.08)" : "none",
                    opacity: selected && !isHighlighted && !isDiag ? 0.35 : 1,
                    display: "flex", alignItems: "center", justifyContent: "center",
                  }}
                >
                  {!isDiag && Math.abs(cos) > 0.13 && (
                    <span style={{
                      fontSize: 8, fontFamily: "monospace", lineHeight: 1,
                      color: cosToTextColor(cos), userSelect: "none",
                    }}>
                      {cos > 0 ? "+" : ""}{cos.toFixed(2)}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        ))}

        {/* color scale legend */}
        <div className="flex items-center gap-3 mt-3 ml-[72px]">
          <span style={{ fontSize: 9, fontFamily: "monospace", color: "rgba(99,102,241,0.8)" }}>−0.6 opposing</span>
          <div style={{ width: 120, height: 8, borderRadius: 4, background: "linear-gradient(to right, rgb(99,102,241), rgb(18,18,22), rgb(245,158,11))" }} />
          <span style={{ fontSize: 9, fontFamily: "monospace", color: "rgba(245,158,11,0.8)" }}>+0.6 similar</span>
        </div>
      </div>

      {/* tooltip */}
      {tooltip && (
        <div className="fixed z-50 pointer-events-none rounded-lg border border-border bg-card/95 px-3 py-1.5 text-xs font-mono shadow-lg"
             style={{ left: tooltip.x + 12, top: tooltip.y - 30 }}>
          {tooltip.text}
        </div>
      )}
    </div>
  );
}

// ─── detail panel ─────────────────────────────────────────────────────────────
function DetailPanel({
  nodeId, nodes, cosMatrix,
}: {
  nodeId: string | null;
  nodes: NodeInfo[];
  cosMatrix: Record<string, Record<string, number>>;
}) {
  if (!nodeId) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-center p-6 rounded-xl border border-border bg-card">
        <div className="text-2xl mb-3">←</div>
        <p className="text-xs text-muted-foreground leading-relaxed">
          Click any node in the graph to see its training examples and strongest connections.
        </p>
      </div>
    );
  }

  const node = nodes.find(n => n.id === nodeId);
  if (!node) return null;
  const meta = META[nodeId];
  const color = CAT_COLOR[node.category];

  // Top connections sorted by |cos|
  const connections = Object.entries(cosMatrix[nodeId] ?? {})
    .filter(([id, v]) => id !== nodeId && Math.abs(v as number) >= EDGE_MIN)
    .sort((a, b) => Math.abs(b[1] as number) - Math.abs(a[1] as number))
    .slice(0, 5) as [string, number][];

  return (
    <div className="rounded-xl border bg-card overflow-y-auto" style={{ borderColor: color + "40", maxHeight: 480 }}>
      {/* header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center gap-2 mb-1">
          <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: color }} />
          <span className="text-xs font-mono uppercase tracking-widest" style={{ color }}>
            {CAT_LABEL[node.category]}
          </span>
        </div>
        <div className="text-base font-bold">{node.label}</div>
        {meta && <p className="text-xs text-muted-foreground mt-1 leading-relaxed">{meta.desc}</p>}
        {meta?.result && (
          <div className="mt-2 text-xs font-mono px-2 py-1 rounded" style={{ background: color + "18", color }}>
            {meta.result}
          </div>
        )}
      </div>

      {/* examples */}
      {meta?.examples && (
        <div className="p-4 border-b border-border">
          <div className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-3">
            Training examples
          </div>
          <div className="space-y-2">
            {meta.examples.map((ex, i) => (
              <div key={i} className="rounded-lg bg-black/30 px-3 py-2 text-xs leading-relaxed text-muted-foreground border-l-2" style={{ borderColor: color + "60" }}>
                &ldquo;{ex}&rdquo;
              </div>
            ))}
          </div>
        </div>
      )}

      {/* connections */}
      {connections.length > 0 && (
        <div className="p-4">
          <div className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-3">
            Strongest connections
          </div>
          <div className="space-y-1.5">
            {connections.map(([id, cos]) => {
              const other = nodes.find(n => n.id === id);
              return (
                <div key={id} className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                        style={{ background: CAT_COLOR[other?.category ?? "style"] }} />
                  <span className="text-xs flex-1">{SHORT[id] ?? id}</span>
                  <span className="text-xs font-mono" style={{ color: cos > 0 ? "rgb(245,158,11)" : "rgb(99,102,241)" }}>
                    {cos > 0 ? "+" : ""}{cos.toFixed(3)}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {cos > 0 ? "similar" : "opposing"}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {connections.length === 0 && (
        <div className="p-4">
          <div className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-2">
            Connections
          </div>
          <p className="text-xs text-muted-foreground">
            No significant alignment (|cos| ≥ 0.10) with any other concept direction.
            This direction is geometrically isolated — consistent with the finding that
            {node.category === "bias" ? " bias" : " factual"} directions are near-random in activation space.
          </p>
        </div>
      )}
    </div>
  );
}

// ─── main export ──────────────────────────────────────────────────────────────
export default function ConceptNetwork() {
  const [overlap, setOverlap] = useState<OverlapData | null>(null);
  const [space,   setSpace]   = useState<SpaceData   | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    fetch("/data/subspace_overlap.json").then(r => r.json()).then(setOverlap).catch(() => null);
    fetch("/data/probe_space_3d.json").then(r => r.json()).then(setSpace).catch(() => null);
  }, []);

  if (!overlap || !space) {
    return (
      <div className="rounded-2xl border border-border bg-card h-64 flex items-center justify-center">
        <div className="w-2 h-2 rounded-full bg-primary/60 pulse-dot" />
      </div>
    );
  }

  const cosMatrix = overlap.cosine_similarity;
  const nodes = space.points;

  return (
    <div className="space-y-6">
      {/* network + detail panel side by side */}
      <div className="grid gap-4" style={{ gridTemplateColumns: "1fr 300px" }}>
        <div className="rounded-2xl border border-border overflow-hidden">
          <NetworkGraph
            nodes={nodes}
            cosMatrix={cosMatrix}
            selected={selected}
            onSelect={setSelected}
          />
        </div>
        <DetailPanel nodeId={selected} nodes={nodes} cosMatrix={cosMatrix} />
      </div>

      {/* heatmap */}
      <div className="rounded-2xl border border-border bg-card p-5">
        <div className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-4">
          Pairwise cosine similarity — all 15 concept directions
        </div>
        <CosineHeatmap
          nodes={nodes}
          cosMatrix={cosMatrix}
          selected={selected}
          onSelect={setSelected}
        />
      </div>
    </div>
  );
}
