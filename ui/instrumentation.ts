/**
 * Next.js instrumentation hook — runs once when the server process starts.
 * Spawns the Python inference server if it isn't already running.
 */
export async function register() {
  if (process.env.NEXT_RUNTIME !== "nodejs") return;

  const { spawn } = await import("child_process");
  const { resolve } = await import("path");

  const projectRoot = resolve(process.cwd(), "..");
  const script = resolve(projectRoot, "scripts", "serve_steer.py");
  const contrast = resolve(projectRoot, "results", "contrast_directions.json");

  // Check if already running
  const alive = await fetch("http://127.0.0.1:8787/health", { signal: AbortSignal.timeout(1000) })
    .then(r => r.ok)
    .catch(() => false);

  if (alive) {
    console.log("[steer] inference server already running");
    return;
  }

  console.log("[steer] starting inference server (loads Llama 3B — ~30s first time)…");

  const proc = spawn(
    "python3",
    [script, "--contrast", contrast, "--port", "8787"],
    {
      cwd: projectRoot,
      stdio: "inherit",
      detached: false,
    }
  );

  proc.on("error", err => console.error("[steer] failed to start server:", err.message));
  proc.on("exit",  code => console.log(`[steer] server exited (code ${code})`));

  // Give it a moment to bind the port before Next.js starts accepting requests
  await new Promise(r => setTimeout(r, 2000));
}
