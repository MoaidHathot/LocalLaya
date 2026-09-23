/**
 * Client for the Laya sidecar (serve.mjs): discover it, spawn it on demand, wait for it, call it, stop it.
 *
 * Deliberately imports nothing heavy (no onnxruntime), so `ask.mjs --sidecar` costs Node start-up + one HTTP
 * round trip instead of loading a 27 MB runtime DLL just to talk to a server.
 *
 *   import { ensureSidecar, decide, status, stop } from "./src/sidecar-client.mjs";
 *   const h = await ensureSidecar({ port: 8787, idle: "5m" });      // running (spawned if needed) and ready
 *   const r = await decide({ preset: "triage", text: "..." }, { port: 8787 });
 *
 * Contract with serve.mjs: /health returns { service: "laya", status: loading|ready|failed|stopping, pid, ... }.
 * Anything else on the port is treated as a foreign service (FOREIGN_PORT) and never touched.
 */
import { spawn } from "node:child_process";
import { mkdir, open, readFile, stat, truncate } from "node:fs/promises";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const DEFAULT_PORT = Number(process.env.LAYA_PORT ?? 8787);
export const DEFAULT_IDLE = process.env.LAYA_IDLE ?? "5m";
export const LOG_DIR = path.join(PROJECT_ROOT, ".laya");

export class SidecarError extends Error {
  constructor(code, message, extra = {}) {
    super(message);
    this.code = code;
    Object.assign(this, extra);
  }
}

/** "30s" | "5m" | "1h" | "250ms" | "300" (seconds) | "0" (never) -> milliseconds */
export function parseDuration(s) {
  if (typeof s === "number") return s;
  const m = String(s ?? "").trim().match(/^(\d+(?:\.\d+)?)\s*(ms|s|m|h)?$/i);
  if (!m) throw new SidecarError("BAD_DURATION", `invalid duration "${s}" (use e.g. 30s, 5m, 1h, 0)`);
  const n = Number(m[1]);
  const unit = (m[2] ?? "s").toLowerCase();
  return Math.round(n * { ms: 1, s: 1000, m: 60_000, h: 3_600_000 }[unit]);
}

export const logFileFor = (port = DEFAULT_PORT) => path.join(LOG_DIR, `sidecar-${port}.log`);

/**
 * Minimal HTTP JSON call on node:http. Without `agent` every request opens a fresh connection (agent: false),
 * which is what the one-shot CLI wants; createClient() passes a keep-alive agent for callers that make many
 * requests from one process.
 * Not fetch(): on Windows, Node 25's undici can crash the process at exit with
 * "Assertion failed: !(handle->flags & UV_HANDLE_CLOSING) (src\win\async.c)" -> exit code 0xC0000409
 * after a couple of requests; plain node:http has no such handle and also exits ~0.2 s sooner.
 */
function request(method, pathname, { port = DEFAULT_PORT, host = "127.0.0.1", body, timeoutMs = 5000, agent = false } = {}) {
  return new Promise((resolve, reject) => {
    const payload = body === undefined ? null : JSON.stringify(body);
    const req = http.request(
      { host, port, path: pathname, method, agent, timeout: timeoutMs, headers: payload ? { "content-type": "application/json", "content-length": Buffer.byteLength(payload) } : {} },
      (res) => {
        let text = "";
        res.setEncoding("utf8");
        res.on("data", (c) => (text += c));
        res.on("end", () => {
          let json = null;
          try {
            json = JSON.parse(text);
          } catch {
            /* not JSON */
          }
          resolve({ status: res.statusCode, ok: res.statusCode >= 200 && res.statusCode < 300, body: json, text });
        });
        res.on("error", reject);
      },
    );
    req.on("timeout", () => req.destroy(Object.assign(new Error(`timeout after ${timeoutMs} ms`), { code: "ETIMEDOUT" })));
    req.on("error", reject);
    if (payload) req.write(payload);
    req.end();
  });
}

/**
 * What is on the port right now.
 * @returns {Promise<{state:"none"|"foreign"|"loading"|"ready"|"failed"|"stopping", health?:object}>}
 */
// 1.5 s: while the model loads the sidecar's event loop blocks for up to ~1 s; a free port refuses instantly anyway.
export async function discover({ port = DEFAULT_PORT, host = "127.0.0.1", timeoutMs = 1500 } = {}) {
  try {
    const { ok, body } = await request("GET", "/health", { port, host, timeoutMs });
    if (!ok || body?.service !== "laya" || typeof body.pid !== "number") return { state: "foreign", health: body };
    return { state: body.status, health: body };
  } catch (e) {
    // ECONNREFUSED / timeout / reset -> nothing usable is listening
    return { state: "none", error: e?.code ?? e?.name ?? String(e) };
  }
}

/** Last `lines` lines of the sidecar log (for error messages). */
export async function logTail(port = DEFAULT_PORT, lines = 12) {
  try {
    const text = await readFile(logFileFor(port), "utf8");
    return text.trimEnd().split(/\r?\n/).slice(-lines).join("\n");
  } catch {
    return "";
  }
}

/**
 * Start serve.mjs as a detached background process (own console session, hidden window, stdout+stderr
 * appended to .laya/sidecar-<port>.log). Returns the ChildProcess; the caller must not wait on it.
 */
export async function spawnSidecar({ port = DEFAULT_PORT, idle = DEFAULT_IDLE, maxAge, lanes, calibration, nodeArgs = [] } = {}) {
  await mkdir(LOG_DIR, { recursive: true });
  const logFile = logFileFor(port);
  try {
    if ((await stat(logFile)).size > 5 * 1024 * 1024) await truncate(logFile, 0);
  } catch {
    /* no log yet */
  }
  const fh = await open(logFile, "a");
  const argv = [...nodeArgs, path.join(PROJECT_ROOT, "serve.mjs"), "--sidecar", "--port", String(port), "--idle", String(idle)];
  if (maxAge) argv.push("--max-age", String(maxAge));
  if (lanes) argv.push("--lanes", lanes);
  if (calibration) argv.push("--calibration", calibration);
  const child = spawn(process.execPath, argv, {
    cwd: PROJECT_ROOT,
    detached: true,
    windowsHide: true,
    stdio: ["ignore", fh.fd, fh.fd],
    env: { ...process.env, LAYA_SIDECAR_PARENT: String(process.pid) },
  });
  child.on("error", () => {});
  child.unref();
  // the fd is inherited by the child; close our handle once the child has started
  child.once("spawn", () => fh.close().catch(() => {}));
  child.once("error", () => fh.close().catch(() => {}));
  return child;
}

/**
 * Poll /health until status is "ready". Throws LOAD_FAILED / TIMEOUT (with the log tail) or FOREIGN_PORT.
 * If `child` is given and exits with a code other than 3 (EADDRINUSE = someone else won the race), fail fast.
 */
export async function waitReady({ port = DEFAULT_PORT, host = "127.0.0.1", timeoutMs = 120_000, intervalMs = 250, child, onProgress } = {}) {
  const t0 = Date.now();
  let lastState = "";
  while (Date.now() - t0 < timeoutMs) {
    if (child && child.exitCode !== null && child.exitCode !== 3) {
      throw new SidecarError("SPAWN_FAILED", `sidecar process exited with code ${child.exitCode} before becoming ready\n${await logTail(port)}`, { exitCode: child.exitCode });
    }
    const d = await discover({ port, host });
    if (d.state !== lastState) {
      lastState = d.state;
      onProgress?.(d.state, Date.now() - t0, d.health);
    }
    if (d.state === "ready") return d.health;
    if (d.state === "failed") throw new SidecarError("LOAD_FAILED", `sidecar failed to load: ${d.health?.error ?? "unknown"}\n${await logTail(port)}`);
    if (d.state === "foreign") throw new SidecarError("FOREIGN_PORT", `port ${port} is used by something that is not the Laya sidecar; set LAYA_PORT / --port to a free port`);
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new SidecarError("TIMEOUT", `sidecar not ready after ${Math.round(timeoutMs / 1000)} s\n${await logTail(port)}`);
}

/**
 * "ready" means the first lane serves; the others join in the background (/health.lanesLoading). Wait until
 * every configured lane is loaded or dropped. Returns the health object.
 */
export async function waitAllLanes({ port = DEFAULT_PORT, host = "127.0.0.1", timeoutMs = 120_000, intervalMs = 250 } = {}) {
  const t0 = Date.now();
  for (;;) {
    const d = await discover({ port, host });
    if (d.state === "ready" && !d.health.lanesLoading?.length) return d.health;
    if (d.state === "failed") throw new SidecarError("LOAD_FAILED", `sidecar failed to load: ${d.health?.error ?? "unknown"}\n${await logTail(port)}`);
    if (d.state === "none" || d.state === "foreign") throw new SidecarError(d.state === "none" ? "GONE" : "FOREIGN_PORT", `no Laya sidecar on port ${port} any more`);
    if (Date.now() - t0 > timeoutMs) throw new SidecarError("TIMEOUT", `lanes ${d.health?.lanesLoading?.join(", ")} still loading after ${Math.round(timeoutMs / 1000)} s`);
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}

/**
 * Make sure a ready sidecar is listening: reuse a running one, wait for a loading one, or spawn a new one.
 * @returns {Promise<{health:object, spawned:boolean}>}
 */
export async function ensureSidecar({ port = DEFAULT_PORT, host = "127.0.0.1", idle = DEFAULT_IDLE, maxAge, lanes, calibration, timeoutMs, onProgress } = {}) {
  const d = await discover({ port, host });
  if (d.state === "ready") return { health: d.health, spawned: false };
  if (d.state === "foreign") throw new SidecarError("FOREIGN_PORT", `port ${port} is used by something that is not the Laya sidecar; set LAYA_PORT / --port to a free port`);
  if (d.state === "loading") {
    onProgress?.("attaching", 0, d.health);
    return { health: await waitReady({ port, host, timeoutMs, onProgress }), spawned: false };
  }
  if (d.state === "stopping") {
    // an idle exit is in progress; give the port a moment to free up, then start fresh
    for (let i = 0; i < 40 && (await discover({ port, host })).state !== "none"; i++) await new Promise((r) => setTimeout(r, 100));
  }
  onProgress?.("spawning", 0);
  const child = await spawnSidecar({ port, idle, maxAge, lanes, calibration });
  const health = await waitReady({ port, host, timeoutMs, child, onProgress });
  return { health, spawned: true, pid: health.pid };
}

/** POST /decide. Retries briefly on 503 (instance still loading / being replaced). */
export async function decide(body, { port = DEFAULT_PORT, host = "127.0.0.1", timeoutMs = 60_000, agent = false } = {}) {
  const t0 = Date.now();
  for (;;) {
    const res = await request("POST", "/decide", { port, host, body, timeoutMs, agent });
    const out = res.body ?? { error: `HTTP ${res.status}` };
    if (res.ok) return out;
    if (res.status === 503 && Date.now() - t0 < timeoutMs) {
      await new Promise((r) => setTimeout(r, out.retryAfterMs ?? 500));
      continue;
    }
    throw new SidecarError(res.status === 400 ? "BAD_REQUEST" : "SERVER_ERROR", out.error ?? `HTTP ${res.status}`, { status: res.status });
  }
}

/**
 * A client for many calls from one process: one keep-alive connection to the sidecar (no TCP handshake per
 * call; measured ~1 ms of HTTP overhead on top of the inference). Does not spawn the sidecar - call
 * ensureSidecar() first or use demo/laya.mjs, which does both.
 *   const c = createClient({ port }); await c.decide({ preset, text }); ...; c.close();
 * close() destroys the pooled socket so the process can exit (the agent is also unref'd - a forgotten close()
 * does not keep the event loop alive).
 */
export function createClient({ port = DEFAULT_PORT, host = "127.0.0.1", timeoutMs = 60_000 } = {}) {
  const agent = new http.Agent({ keepAlive: true, maxSockets: 8, keepAliveMsecs: 10_000 });
  // http.Agent has no unref(); its sockets are unref'd when idle by Node itself ("agent sockets are unref'd when free")
  return {
    port,
    host,
    decide: (body) => decide(body, { port, host, timeoutMs, agent }),
    health: async () => (await request("GET", "/health", { port, host, timeoutMs: 5000, agent })).body,
    stats: async () => (await request("GET", "/stats", { port, host, timeoutMs: 5000, agent })).body,
    presets: async () => (await request("GET", "/presets", { port, host, timeoutMs: 5000, agent })).body,
    touch: async () => (await request("POST", "/touch", { port, host, timeoutMs: 2000, agent })).body,
    close: () => agent.destroy(),
  };
}

export async function status({ port = DEFAULT_PORT, host = "127.0.0.1" } = {}) {
  return discover({ port, host });
}

export async function touch({ port = DEFAULT_PORT, host = "127.0.0.1" } = {}) {
  return (await request("POST", "/touch", { port, host, timeoutMs: 2000 })).body;
}

export async function stats({ port = DEFAULT_PORT, host = "127.0.0.1" } = {}) {
  return (await request("GET", "/stats", { port, host, timeoutMs: 5000 })).body;
}

/** Graceful stop; resolves when the port is free again (or after `timeoutMs`). Returns false when nothing was running. */
export async function stop({ port = DEFAULT_PORT, host = "127.0.0.1", timeoutMs = 15_000 } = {}) {
  const d = await discover({ port, host });
  if (d.state === "none") return false;
  if (d.state === "foreign") throw new SidecarError("FOREIGN_PORT", `port ${port} is not a Laya sidecar; refusing to stop it`);
  await request("POST", "/shutdown", { port, host, timeoutMs: 3000 }).catch(() => {});
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    if ((await discover({ port, host })).state === "none") return true;
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new SidecarError("TIMEOUT", `sidecar (pid ${d.health?.pid}) did not exit within ${timeoutMs} ms`);
}
