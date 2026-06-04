import { NextRequest, NextResponse } from "next/server";

const BACKEND = "http://127.0.0.1:8787";

async function proxy(path: string, body?: unknown) {
  const res = await fetch(`${BACKEND}${path}`, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(90_000),
  });
  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}

export async function GET() {
  try { return await proxy("/health"); }
  catch { return NextResponse.json({ status: "offline" }, { status: 503 }); }
}

export async function POST(req: NextRequest) {
  const { pathname } = new URL(req.url);
  const endpoint = pathname.replace("/api/steer", "") || "/steer";
  const body = await req.json();
  try { return await proxy(endpoint, body); }
  catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      { error: "Inference server unavailable. Run: python scripts/serve_steer.py", detail: msg },
      { status: 503 },
    );
  }
}
