import { NextRequest, NextResponse } from "next/server";

const BACKEND = "http://127.0.0.1:8787";

export async function POST(req: NextRequest) {
  const body = await req.json();
  try {
    const res = await fetch(`${BACKEND}/steer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(60_000),
    });
    const data = await res.json();
    if (!res.ok) return NextResponse.json(data, { status: res.status });
    return NextResponse.json(data);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      { error: "Inference server unavailable. Run: python scripts/serve_steer.py", detail: msg },
      { status: 503 }
    );
  }
}

export async function GET() {
  try {
    const res = await fetch(`${BACKEND}/health`, { signal: AbortSignal.timeout(3_000) });
    const data = await res.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json({ status: "offline" }, { status: 503 });
  }
}
