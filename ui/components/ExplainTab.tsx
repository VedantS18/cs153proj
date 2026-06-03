"use client";

import { useEffect, useRef, useState, useCallback } from "react";

type Point = { id: string; label: string; category: string; x: number; y: number; z: number };
type SpaceData = { points: Point[]; explained_variance_3d: number };

const CAT_COLOR: Record<string, string> = {
  style:   "#a78bfa",
  bias:    "#f87171",
  factual: "#fbbf24",
};
const CAT_LABEL: Record<string, string> = {
  style: "Stylistic", bias: "Bias", factual: "Factual",
};

function rotX(p: [number,number,number], a: number): [number,number,number] {
  const c = Math.cos(a), s = Math.sin(a);
  return [p[0], p[1]*c - p[2]*s, p[1]*s + p[2]*c];
}
function rotY(p: [number,number,number], a: number): [number,number,number] {
  const c = Math.cos(a), s = Math.sin(a);
  return [p[0]*c + p[2]*s, p[1], -p[0]*s + p[2]*c];
}

function Canvas3D({ data }: { data: SpaceData }) {
  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const rotRef      = useRef({ x: 0.3, y: -0.5 });
  const dragRef     = useRef<{ sx: number; sy: number; rx: number; ry: number } | null>(null);
  const zoomRef     = useRef(1.0);
  const animRef     = useRef(0);
  const autoRef     = useRef(true);
  const hoverRef    = useRef<string | null>(null);
  const [hovered, setHovered] = useState<Point | null>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width / window.devicePixelRatio;
    const H = canvas.height / window.devicePixelRatio;
    ctx.save();
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    ctx.clearRect(0, 0, W, H);

    const cx = W / 2, cy = H / 2;
    const scale = Math.min(W, H) * 0.33 * zoomRef.current;
    const fov = 3.5;

    const proj = data.points.map(pt => {
      let p: [number,number,number] = [pt.x, pt.y, pt.z];
      p = rotX(p, rotRef.current.x);
      p = rotY(p, rotRef.current.y);
      const z = p[2] + fov;
      return { ...pt, px: cx + (p[0]*scale)/z, py: cy - (p[1]*scale)/z, pz: p[2], depth: (p[2]+1.5)/3 };
    }).sort((a, b) => a.pz - b.pz);

    // Style edges
    ctx.save();
    const stylePoints = proj.filter(p => p.category === "style");
    for (let i = 0; i < stylePoints.length; i++) {
      for (let j = i+1; j < stylePoints.length; j++) {
        const a = stylePoints[i], b = stylePoints[j];
        const alpha = 0.06 + 0.05 * (a.depth + b.depth) / 2;
        ctx.beginPath();
        ctx.moveTo(a.px, a.py); ctx.lineTo(b.px, b.py);
        ctx.strokeStyle = CAT_COLOR.style + Math.round(alpha * 255).toString(16).padStart(2, "0");
        ctx.lineWidth = 0.7;
        ctx.stroke();
      }
    }
    ctx.restore();

    // Points
    for (const pt of proj) {
      const color = CAT_COLOR[pt.category];
      const isH = pt.id === hoverRef.current;
      const r = (isH ? 13 : 8) + pt.depth * 4;
      const alpha = 0.5 + pt.depth * 0.5;

      if (isH) {
        const g = ctx.createRadialGradient(pt.px, pt.py, r*0.3, pt.px, pt.py, r*2.8);
        g.addColorStop(0, color + "40"); g.addColorStop(1, color + "00");
        ctx.beginPath(); ctx.arc(pt.px, pt.py, r*2.8, 0, Math.PI*2); ctx.fillStyle = g; ctx.fill();
      }

      const g2 = ctx.createRadialGradient(pt.px - r*0.3, pt.py - r*0.3, r*0.05, pt.px, pt.py, r);
      g2.addColorStop(0, color + "ff");
      g2.addColorStop(0.6, color + Math.round(alpha * 190).toString(16).padStart(2,"0"));
      g2.addColorStop(1, color + "30");
      ctx.beginPath(); ctx.arc(pt.px, pt.py, r, 0, Math.PI*2); ctx.fillStyle = g2; ctx.fill();

      const la = isH ? 1.0 : 0.45 + pt.depth * 0.45;
      ctx.font = isH ? "bold 11px monospace" : "10px monospace";
      ctx.fillStyle = `rgba(255,255,255,${la})`;
      ctx.textAlign = "center";
      ctx.fillText(pt.label, pt.px, pt.py - r - 4);
    }

    // Legend
    let ly = 14;
    ctx.font = "10px monospace"; ctx.textAlign = "left";
    for (const cat of ["style","bias","factual"]) {
      ctx.beginPath(); ctx.arc(14, ly+5, 5, 0, Math.PI*2);
      ctx.fillStyle = CAT_COLOR[cat]; ctx.fill();
      ctx.fillStyle = "rgba(255,255,255,0.45)";
      ctx.fillText(CAT_LABEL[cat], 24, ly+9);
      ly += 18;
    }
    ctx.restore();
  }, [data]);

  useEffect(() => {
    let last = 0;
    const loop = (t: number) => {
      if (autoRef.current && !dragRef.current) {
        const dt = Math.min(t - last, 50);
        rotRef.current.y += 0.0025 * (dt / 16);
      }
      last = t;
      draw();
      animRef.current = requestAnimationFrame(loop);
    };
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
      canvas.width  = rect.width  * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      canvas.style.width  = rect.width  + "px";
      canvas.style.height = rect.height + "px";
      draw();
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [draw]);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    autoRef.current = false;
    dragRef.current = { sx: e.clientX, sy: e.clientY, rx: rotRef.current.x, ry: rotRef.current.y };
  }, []);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (dragRef.current) {
      rotRef.current = {
        x: dragRef.current.rx + (e.clientY - dragRef.current.sy) * 0.008,
        y: dragRef.current.ry + (e.clientX - dragRef.current.sx) * 0.008,
      };
      return;
    }
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const W = rect.width, H = rect.height;
    const cx = W/2, cy = H/2, fov = 3.5;
    const scale = Math.min(W,H) * 0.33 * zoomRef.current;
    let nearest: string | null = null, nearDist = 28;
    for (const pt of data.points) {
      let p: [number,number,number] = [pt.x, pt.y, pt.z];
      p = rotX(p, rotRef.current.x); p = rotY(p, rotRef.current.y);
      const z = p[2] + fov;
      const px = cx + (p[0]*scale)/z, py = cy - (p[1]*scale)/z;
      const d = Math.hypot(mx-px, my-py);
      if (d < nearDist) { nearDist = d; nearest = pt.id; }
    }
    if (nearest !== hoverRef.current) {
      hoverRef.current = nearest;
      setHovered(data.points.find(p => p.id === nearest) ?? null);
    }
  }, [data]);

  const onMouseUp = useCallback(() => { dragRef.current = null; }, []);
  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    zoomRef.current = Math.max(0.4, Math.min(3, zoomRef.current - e.deltaY * 0.001));
  }, []);

  return (
    <div className="relative" ref={containerRef} style={{ height: 420 }}>
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-grab active:cursor-grabbing"
        onMouseDown={onMouseDown} onMouseMove={onMouseMove}
        onMouseUp={onMouseUp} onMouseLeave={onMouseUp} onWheel={onWheel}
      />
      {hovered && (
        <div
          className="absolute bottom-4 left-4 rounded-xl border bg-card/90 backdrop-blur p-3 pointer-events-none"
          style={{ borderColor: CAT_COLOR[hovered.category] + "50" }}
        >
          <div className="text-xs font-mono uppercase tracking-widest mb-1"
               style={{ color: CAT_COLOR[hovered.category] }}>
            {CAT_LABEL[hovered.category]}
          </div>
          <div className="text-sm font-bold">{hovered.label}</div>
        </div>
      )}
    </div>
  );
}

export default function ExplainTab() {
  const [data, setData] = useState<SpaceData | null>(null);

  useEffect(() => {
    fetch("/data/probe_space_3d.json").then(r => r.json()).then(setData).catch(() => null);
  }, []);

  return (
    <div className="pt-10 space-y-10">
      {/* header */}
      <div>
        <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-muted-foreground mb-3">
          <span className="w-4 h-px bg-primary/60" />
          The Geometry
        </div>
        <h2 className="text-3xl font-bold tracking-tight mb-2">Style is structured. Bias is noise.</h2>
        <p className="text-sm text-muted-foreground max-w-xl">
          Each learned probe defines a direction in the model&apos;s 3072-dimensional residual stream.
          MDS projects those directions to 3D, preserving pairwise distances.
          <span className="text-foreground"> Drag to rotate.</span>
        </p>
      </div>

      {/* 3D canvas */}
      <div className="rounded-2xl border border-border bg-card overflow-hidden">
        {!data ? (
          <div className="h-[420px] flex items-center justify-center">
            <div className="w-2 h-2 rounded-full bg-primary/60 pulse-dot" />
          </div>
        ) : (
          <Canvas3D data={data} />
        )}
      </div>

      {/* three callouts */}
      <div className="grid grid-cols-3 gap-4">
        <div className="rounded-xl border border-border bg-card p-5">
          <div className="text-4xl font-bold mb-2" style={{ color: CAT_COLOR.style }}>17×</div>
          <div className="text-sm font-semibold mb-1">Style is structured</div>
          <div className="text-xs text-muted-foreground leading-relaxed">
            Pairwise cosine std for style directions: 0.32. Random vectors in 3072D: 0.018.
            Style directions are 17× more structured than chance — they occupy a coherent subspace.
          </div>
        </div>

        <div className="rounded-xl border border-border bg-card p-5">
          <div className="text-4xl font-bold mb-2" style={{ color: "oklch(0.72 0.20 270)" }}>cos 0.16</div>
          <div className="text-sm font-semibold mb-1">Probe ≠ behavior direction</div>
          <div className="text-xs text-muted-foreground leading-relaxed">
            For bias concepts, the probe weight vector (what the classifier &ldquo;sees&rdquo;) has only
            16% cosine overlap with the contrast direction that actually governs behavior.
            Erasure along the probe removes the wrong thing.
          </div>
        </div>

        <div className="rounded-xl border border-border bg-card p-5">
          <div className="text-4xl font-bold mb-2" style={{ color: CAT_COLOR.factual }}>1B = 3B</div>
          <div className="text-sm font-semibold mb-1">Cross-model universality</div>
          <div className="text-xs text-muted-foreground leading-relaxed">
            Llama 1B and 3B independently learn the same geometric structure.
            Top pair: legal ↔ scientific, cos=+0.58 (3B) / +0.44 (1B).
            The subspace is a property of the training data, not the model size.
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-border/50 bg-card/50 p-4">
        <p className="text-xs text-muted-foreground leading-relaxed">
          <span className="text-foreground font-medium">Why this matters:</span>{" "}
          The 17× structure gap predicts which steering interventions work.
          Style directions are coherent → steering is smooth and controllable.
          Bias directions are incoherent → there is no single direction to erase,
          which is why nullspace projection (a common erasure baseline) fails for bias
          while succeeding for factual knowledge.
        </p>
      </div>
    </div>
  );
}
