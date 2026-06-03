"use client";

import { useEffect, useRef, useState, useCallback } from "react";

type Point = { id: string; label: string; category: string; x: number; y: number; z: number };
type SpaceData = { points: Point[]; explained_variance_3d: number };

const CATEGORY_COLOR: Record<string, string> = {
  style:   "#a78bfa",  // violet
  bias:    "#f87171",  // red
  factual: "#fbbf24",  // amber
};

const CATEGORY_LABEL: Record<string, string> = {
  style:   "Stylistic",
  bias:    "Bias",
  factual: "Factual",
};

function rotateX(p: [number,number,number], a: number): [number,number,number] {
  const cos = Math.cos(a), sin = Math.sin(a);
  return [p[0], p[1]*cos - p[2]*sin, p[1]*sin + p[2]*cos];
}
function rotateY(p: [number,number,number], a: number): [number,number,number] {
  const cos = Math.cos(a), sin = Math.sin(a);
  return [p[0]*cos + p[2]*sin, p[1], -p[0]*sin + p[2]*cos];
}

export default function ProbeSpace3D() {
  const [data, setData] = useState<SpaceData | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const rotRef = useRef({ x: 0.3, y: -0.4 });
  const dragRef = useRef<{ startX: number; startY: number; rotX: number; rotY: number } | null>(null);
  const zoomRef = useRef(1.0);
  const animRef = useRef<number>(0);
  const autoRotRef = useRef(true);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const hoveredIdRef = useRef<string | null>(null);

  useEffect(() => {
    fetch("/data/probe_space_3d.json").then(r => r.json()).then(setData).catch(() => null);
  }, []);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const cx = W / 2, cy = H / 2;
    const scale = Math.min(W, H) * 0.35 * zoomRef.current;
    const fov = 3.5;

    // Project all points
    const projected = data.points.map(pt => {
      let p: [number,number,number] = [pt.x, pt.y, pt.z];
      p = rotateX(p, rotRef.current.x);
      p = rotateY(p, rotRef.current.y);
      const z = p[2] + fov;
      const px = cx + (p[0] * scale) / z;
      const py = cy - (p[1] * scale) / z;
      const depth = (p[2] + 1.5) / 3;  // 0..1
      return { ...pt, px, py, pz: p[2], depth };
    });

    // Sort by depth (back to front)
    projected.sort((a, b) => a.pz - b.pz);

    // Draw connection lines between same-category points (thin)
    ctx.save();
    for (let i = 0; i < projected.length; i++) {
      for (let j = i + 1; j < projected.length; j++) {
        const a = projected[i], b = projected[j];
        if (a.category !== b.category) continue;
        if (a.category === 'bias' || a.category === 'factual') continue; // only style edges
        const color = CATEGORY_COLOR[a.category];
        const alpha = 0.08 + 0.06 * (a.depth + b.depth) / 2;
        ctx.beginPath();
        ctx.moveTo(a.px, a.py);
        ctx.lineTo(b.px, b.py);
        ctx.strokeStyle = color + Math.round(alpha * 255).toString(16).padStart(2, '0');
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }
    }
    ctx.restore();

    // Draw points
    for (const pt of projected) {
      const color = CATEGORY_COLOR[pt.category];
      const isHovered = pt.id === hoveredIdRef.current;
      const r = (isHovered ? 14 : 9) + pt.depth * 5;
      const alpha = 0.55 + pt.depth * 0.45;

      // Glow for hovered
      if (isHovered) {
        const grd = ctx.createRadialGradient(pt.px, pt.py, r * 0.3, pt.px, pt.py, r * 2.5);
        grd.addColorStop(0, color + "44");
        grd.addColorStop(1, color + "00");
        ctx.beginPath();
        ctx.arc(pt.px, pt.py, r * 2.5, 0, Math.PI * 2);
        ctx.fillStyle = grd;
        ctx.fill();
      }

      // Shadow
      ctx.beginPath();
      ctx.arc(pt.px + 1.5, pt.py + 1.5, r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0,0,0,${0.3 * alpha})`;
      ctx.fill();

      // Main sphere with gradient
      const grd2 = ctx.createRadialGradient(pt.px - r * 0.3, pt.py - r * 0.3, r * 0.05, pt.px, pt.py, r);
      grd2.addColorStop(0, color + "ff");
      grd2.addColorStop(0.6, color + Math.round(alpha * 200).toString(16).padStart(2,'0'));
      grd2.addColorStop(1, color + "44");
      ctx.beginPath();
      ctx.arc(pt.px, pt.py, r, 0, Math.PI * 2);
      ctx.fillStyle = grd2;
      ctx.fill();

      // Label
      const labelAlpha = isHovered ? 1.0 : 0.55 + pt.depth * 0.45;
      ctx.font = isHovered ? "bold 11px monospace" : "10px monospace";
      ctx.fillStyle = `rgba(255,255,255,${labelAlpha})`;
      ctx.textAlign = "center";
      ctx.fillText(pt.label, pt.px, pt.py - r - 5);
    }

    // Legend
    const categories = ['style', 'bias', 'factual'];
    let lx = 16, ly = 16;
    ctx.font = "10px monospace";
    for (const cat of categories) {
      ctx.beginPath();
      ctx.arc(lx + 5, ly + 5, 5, 0, Math.PI * 2);
      ctx.fillStyle = CATEGORY_COLOR[cat];
      ctx.fill();
      ctx.fillStyle = "rgba(255,255,255,0.5)";
      ctx.textAlign = "left";
      ctx.fillText(CATEGORY_LABEL[cat], lx + 14, ly + 9);
      ly += 18;
    }
  }, [data]);

  // Auto-rotation loop
  useEffect(() => {
    if (!data) return;
    let last = 0;
    const loop = (t: number) => {
      if (autoRotRef.current && !dragRef.current) {
        const dt = Math.min(t - last, 50);
        rotRef.current.y += 0.003 * (dt / 16);
      }
      last = t;
      draw();
      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [data, draw]);

  // Resize canvas
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = el.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      canvas.style.width = rect.width + "px";
      canvas.style.height = rect.height + "px";
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      draw();
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [draw]);

  // Mouse / touch interaction
  const onMouseDown = useCallback((e: React.MouseEvent) => {
    autoRotRef.current = false;
    dragRef.current = { startX: e.clientX, startY: e.clientY, rotX: rotRef.current.x, rotY: rotRef.current.y };
  }, []);
  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (dragRef.current) {
      const dx = (e.clientX - dragRef.current.startX) * 0.008;
      const dy = (e.clientY - dragRef.current.startY) * 0.008;
      rotRef.current = { x: dragRef.current.rotX + dy, y: dragRef.current.rotY + dx };
    } else {
      // Hover detection
      const canvas = canvasRef.current;
      if (!canvas || !data) return;
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      const W = rect.width, H = rect.height;
      const cx = W / 2, cy = H / 2;
      const scale = Math.min(W, H) * 0.35 * zoomRef.current;
      const fov = 3.5;
      let nearest: string | null = null;
      let nearDist = 30;
      for (const pt of data.points) {
        let p: [number,number,number] = [pt.x, pt.y, pt.z];
        p = rotateX(p, rotRef.current.x);
        p = rotateY(p, rotRef.current.y);
        const z = p[2] + fov;
        const px = cx + (p[0] * scale) / z;
        const py = cy - (p[1] * scale) / z;
        const d = Math.hypot(mx - px, my - py);
        if (d < nearDist) { nearDist = d; nearest = pt.id; }
      }
      if (nearest !== hoveredIdRef.current) {
        hoveredIdRef.current = nearest;
        setHoveredId(nearest);
      }
    }
  }, [data]);
  const onMouseUp = useCallback(() => { dragRef.current = null; }, []);
  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    zoomRef.current = Math.max(0.4, Math.min(3.0, zoomRef.current - e.deltaY * 0.001));
  }, []);

  const hoveredPoint = data?.points.find(p => p.id === hoveredId);

  return (
    <section className="py-24 px-6 border-t border-border" id="probe-space">
      <div className="max-w-6xl mx-auto">
        <div className="mb-10">
          <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-muted-foreground mb-4">
            <span className="w-4 h-px bg-primary/60" />
            Vector Space
          </div>
          <h2 className="text-4xl font-bold tracking-tight mb-3">
            Concept directions in activation space
          </h2>
          <p className="text-muted-foreground max-w-xl">
            Each of the 15 learned concept probes defines a direction in the model&apos;s 3072-dimensional
            residual stream. Here we visualize those directions via MDS — preserving pairwise cosine distances
            in 3D. <span className="text-foreground">Drag to rotate.</span>
          </p>
        </div>

        <div className="grid grid-cols-3 gap-6">
          {/* 3D canvas */}
          <div className="col-span-2 rounded-2xl border border-border bg-card overflow-hidden"
               style={{ aspectRatio: "16/10" }}>
            <div ref={containerRef} className="relative w-full h-full">
              {!data && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-2 h-2 rounded-full bg-primary/60 pulse-dot" />
                </div>
              )}
              <canvas
                ref={canvasRef}
                className="w-full h-full cursor-grab active:cursor-grabbing"
                onMouseDown={onMouseDown}
                onMouseMove={onMouseMove}
                onMouseUp={onMouseUp}
                onMouseLeave={onMouseUp}
                onWheel={onWheel}
              />
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-4">
            {/* Hovered point card */}
            <div
              className="rounded-xl border bg-card p-4 min-h-[120px] transition-all duration-200"
              style={{
                borderColor: hoveredPoint ? CATEGORY_COLOR[hoveredPoint.category] + "50" : undefined,
              }}
            >
              {hoveredPoint ? (
                <>
                  <div className="text-xs font-mono uppercase tracking-widest mb-2"
                       style={{ color: CATEGORY_COLOR[hoveredPoint.category] }}>
                    {CATEGORY_LABEL[hoveredPoint.category]} concept
                  </div>
                  <div className="text-lg font-bold mb-1">{hoveredPoint.label}</div>
                  <div className="text-xs font-mono text-muted-foreground space-y-0.5">
                    <div>x: {hoveredPoint.x.toFixed(3)}</div>
                    <div>y: {hoveredPoint.y.toFixed(3)}</div>
                    <div>z: {hoveredPoint.z.toFixed(3)}</div>
                  </div>
                </>
              ) : (
                <p className="text-xs text-muted-foreground mt-6 text-center">
                  Hover a point to inspect
                </p>
              )}
            </div>

            {/* Key findings */}
            <div className="rounded-xl border border-border bg-card p-4 space-y-3">
              <p className="text-xs font-mono uppercase tracking-widest text-muted-foreground">Key geometry</p>
              <div className="space-y-2.5">
                <div>
                  <div className="text-sm font-semibold" style={{ color: CATEGORY_COLOR.style }}>Style is structured</div>
                  <div className="text-xs text-muted-foreground leading-relaxed">Pairwise cosine std = 0.32 vs 0.018 random — <span className="text-foreground font-medium">17× above noise</span></div>
                </div>
                <div className="h-px bg-border" />
                <div>
                  <div className="text-sm font-semibold" style={{ color: CATEGORY_COLOR.bias }}>Bias is noise</div>
                  <div className="text-xs text-muted-foreground leading-relaxed">Pairwise cosine std = 0.037 — barely 2× random. Directions scatter like random vectors.</div>
                </div>
                <div className="h-px bg-border" />
                <div>
                  <div className="text-sm font-semibold text-foreground">Cross-model stable</div>
                  <div className="text-xs text-muted-foreground leading-relaxed">Same structure in Llama 1B and 3B. Top pairs: legal↔scientific (+0.58/+0.44).</div>
                </div>
              </div>
            </div>

            <div className="rounded-xl border border-border bg-card p-4">
              <p className="text-xs font-mono text-muted-foreground">
                MDS projection · 3072D → 3D<br />
                33% variance explained<br />
                Drag to rotate · scroll to zoom
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
