import { NextRequest, NextResponse } from "next/server";
const BACKEND = "http://127.0.0.1:8787";
export async function POST(req: NextRequest) {
  const path = new URL(req.url).pathname.replace("/api/steer", "");
  const body = await req.json();
  try {
    const res = await fetch(`${BACKEND}${path}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body), signal: AbortSignal.timeout(90_000),
    });
    return NextResponse.json(await res.json(), { status: res.status });
  } catch (err: unknown) {
    return NextResponse.json({ error: "Inference server unavailable" }, { status: 503 });
  }
}
