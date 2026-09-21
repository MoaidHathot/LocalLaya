/**
 * Reusable Laya client for integration with other workflows.
 *
 * Design goals (vs. the naive `Laya.load()`):
 *  - Model is PINNED to a specific Hugging Face commit (immutable `resolve/<sha>/` URLs),
 *    never the mutable `main` branch.
 *  - Model files are cached UNDER THIS PROJECT (./models), not under ~/.cache.
 *  - After the first download, weights are verified by SHA256 against the LFS oids published by
 *    Hugging Face (the library itself only compares byte sizes). A `.verified-<sha>` marker is
 *    written so the 1.7 GB hash is computed once.
 *  - Once the bundle is on disk we load it via `modelDir`, which performs ZERO network calls
 *    (the library's default path issues a HEAD request per file on every start-up).
 *  - Execution provider is selectable: "cpu" | "dml" | "webgpu". On Windows x64 onnxruntime-node
 *    has NO CUDA EP; the NVIDIA GPU is reached via DirectML ("dml") or the experimental WebGPU EP.
 */
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { access, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Laya } from "@receptron/laya";

const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

/** Hugging Face repo that publishes the ONNX export of convaiinnovations/laya (English, 421M, fp32). */
export const MODEL_REPO = "receptron/laya-onnx";

/** Pinned commit of MODEL_REPO. Captured 2026-09-21 via https://huggingface.co/api/models/receptron/laya-onnx */
export const MODEL_REVISION = "68f27dfe5a27a54fb2b1fefc432f43f972e90868";

/** Expected sizes + SHA256 (LFS oids) from the HF tree API at MODEL_REVISION. */
export const EXPECTED_FILES = {
  "laya.onnx": { size: 3807291, sha256: "a874eb254b58b0fcb1e7ad56fbb188c29d64e08c9a46b689433e1f52c66dba1e" },
  "laya.onnx.data": { size: 1685258240, sha256: "487746363a8da57bcadb4345352997d22a0fb90d70aa22c6856668d023242aba" },
  // Small non-LFS files: HF exposes no sha256 for these, so only the size is checked.
  "laya_config.json": { size: 369 },
  "tokenizer/tokenizer.json": { size: 3583228 },
  "tokenizer/tokenizer_config.json": { size: 308 },
};

/** Local cache root (everything downloaded lives under the project). Override with LAYA_CACHE. */
export const CACHE_DIR = process.env.LAYA_CACHE ?? path.join(PROJECT_ROOT, "models");

/** Directory where the pinned bundle ends up (mirrors the library's own layout). */
export const MODEL_DIR = path.join(CACHE_DIR, MODEL_REPO.replace("/", "--"), MODEL_REVISION);

export const SUPPORTED_EPS = ["cpu", "dml", "webgpu"];

async function exists(p) {
  try {
    await access(p);
    return true;
  } catch {
    return false;
  }
}

async function sha256File(p) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(p, { highWaterMark: 8 * 1024 * 1024 })) hash.update(chunk);
  return hash.digest("hex");
}

/**
 * Verify every bundle file against EXPECTED_FILES. Throws on mismatch.
 * Returns an array of { file, size, sha256?, ok } for reporting.
 */
export async function verifyBundle(modelDir = MODEL_DIR, { force = false, log = () => {} } = {}) {
  const marker = path.join(modelDir, `.verified-${MODEL_REVISION}`);
  if (!force && (await exists(marker))) {
    log(`bundle previously verified (${path.basename(marker)})`);
    return JSON.parse(await readFile(marker, "utf8"));
  }
  const { stat } = await import("node:fs/promises");
  const report = [];
  for (const [file, exp] of Object.entries(EXPECTED_FILES)) {
    const p = path.join(modelDir, file);
    const st = await stat(p);
    const entry = { file, size: st.size, expectedSize: exp.size, ok: st.size === exp.size };
    if (!entry.ok) throw new Error(`size mismatch for ${file}: got ${st.size}, expected ${exp.size}`);
    if (exp.sha256) {
      log(`hashing ${file} (${(st.size / 1e6).toFixed(1)} MB)...`);
      const t0 = performance.now();
      entry.sha256 = await sha256File(p);
      entry.expectedSha256 = exp.sha256;
      entry.hashMs = Math.round(performance.now() - t0);
      entry.ok = entry.sha256 === exp.sha256;
      if (!entry.ok) throw new Error(`SHA256 mismatch for ${file}: got ${entry.sha256}, expected ${exp.sha256}`);
    }
    report.push(entry);
  }
  await writeFile(marker, JSON.stringify(report, null, 2));
  log("bundle verified OK");
  return report;
}

/** True when all five bundle files are already present in MODEL_DIR. */
export async function bundleOnDisk(modelDir = MODEL_DIR) {
  for (const file of Object.keys(EXPECTED_FILES)) if (!(await exists(path.join(modelDir, file)))) return false;
  return true;
}

/**
 * Number of logical processors that belong to performance cores. Intel hybrid CPUs enumerate P-cores
 * first (i9-14900KF: 8 P-cores x 2 HT = logical 0-15, then 16 E-cores = logical 16-31).
 * Override with LAYA_PCORE_LOGICAL when the layout differs.
 */
export const PCORE_LOGICAL = Number(process.env.LAYA_PCORE_LOGICAL ?? 16);

/**
 * ORT affinity string for `threads` intra-op threads pinned to the P-cores, one thread per logical
 * processor, spreading over physical cores first (logical 0,2,4,.. are distinct cores; 1,3,5,.. their
 * hyper-threads) so that 8 threads land on 8 cores rather than on 4 cores x 2 HT.
 * ORT expects threads-1 entries (the calling thread is not in the pool); processor ids are 1-based.
 */
export function pCoreAffinity(threads, pLogical = PCORE_LOGICAL) {
  const order = [];
  for (let i = 0; i < pLogical; i += 2) order.push(i);
  for (let i = 1; i < pLogical; i += 2) order.push(i);
  const n = Math.min(threads, pLogical) - 1;
  return order.slice(0, n).map((i) => String(i + 1)).join(";");
}

/**
 * Build the onnxruntime-node options for a given execution provider.
 * - dml: DirectML forbids memory-pattern optimisation and parallel execution; set both explicitly
 *        (the C API does this internally, but being explicit costs nothing and documents intent).
 * - cpu: `threads` = intra-op thread count. ORT's default (= all 24 physical cores here) spreads every
 *        parallel op over P- and E-cores and waits for the slowest E-core thread: measured 213 ms p50 but
 *        1174 ms p95 for 3 questions. `pinToPCores` pins the pool to the P-cores (16 threads by default):
 *        235 ms p50 / 284 ms p95, stable.
 */
export function buildSessionConfig(ep = "cpu", { threads, deviceId = 0, logSeverityLevel, optLevel, pinToPCores = false } = {}) {
  if (!SUPPORTED_EPS.includes(ep)) throw new Error(`unsupported ep "${ep}" (Windows x64 options: ${SUPPORTED_EPS.join(", ")})`);
  const sessionOptions = {};
  if (logSeverityLevel !== undefined) sessionOptions.logSeverityLevel = logSeverityLevel;
  // "disabled" | "basic" | "extended" | "all" (Laya defaults to "all"; lower levels can work around EP-specific fusion bugs)
  if (optLevel) sessionOptions.graphOptimizationLevel = optLevel;
  let executionProviders;
  if (ep === "cpu") {
    executionProviders = ["cpu"];
    const t = threads ?? (pinToPCores ? PCORE_LOGICAL : undefined);
    if (t) sessionOptions.intraOpNumThreads = t;
    if (pinToPCores && t > 1) sessionOptions.extra = { session: { intra_op_thread_affinities: pCoreAffinity(t) } };
  } else if (ep === "dml") {
    executionProviders = [{ name: "dml", deviceId }];
    sessionOptions.enableMemPattern = false;
    sessionOptions.executionMode = "sequential";
  } else {
    executionProviders = ["webgpu"];
  }
  return { executionProviders, sessionOptions };
}

/**
 * Load Laya with the pinned model, project-local cache and requested execution provider.
 *
 * @param {object} opts
 * @param {"cpu"|"dml"|"webgpu"} [opts.ep="cpu"]
 * @param {number} [opts.threads]           intra-op threads (cpu only)
 * @param {number} [opts.deviceId=0]        GPU adapter index (dml only)
 * @param {number} [opts.logSeverityLevel]  0 verbose .. 4 fatal (ORT default 2 = warning)
 * @param {string} [opts.modelDir]           load this bundle directory instead of the pinned download (no verification)
 * @param {boolean} [opts.pinToPCores]       cpu only: pin intra-op threads to the P-cores (see buildSessionConfig)
 * @param {string|object} [opts.calibration] path to a table written by calibrate.mjs (or the parsed object);
 *                                          its temperature_by_options override the shipped ones
 * @param {(msg:string)=>void} [opts.log]
 * @param {(info:{file:string,received:number,total:number|null})=>void} [opts.onProgress]
 * @returns {Promise<{ laya: import("@receptron/laya").Laya, modelDir: string, source: "modelDir"|"download", loadMs: number, ep: string, calibration: object|null }>}
 */
export async function loadLaya(opts = {}) {
  const { ep = "cpu", log = () => {}, onProgress } = opts;
  const { executionProviders, sessionOptions } = buildSessionConfig(ep, opts);
  const t0 = performance.now();
  let laya;
  let source;
  let modelDir = MODEL_DIR;
  if (opts.modelDir) {
    // Explicit bundle (e.g. models/laya-onnx-fp16 from tools/convert_fp16.py): no download, no hash check.
    modelDir = path.resolve(PROJECT_ROOT, opts.modelDir);
    laya = await Laya.load({ modelDir, executionProviders, sessionOptions });
    source = "custom";
  } else if (await bundleOnDisk()) {
    // Fast path: no network at all.
    await verifyBundle(MODEL_DIR, { log });
    laya = await Laya.load({ modelDir: MODEL_DIR, executionProviders, sessionOptions });
    source = "modelDir";
  } else {
    log(`bundle not found under ${MODEL_DIR}; downloading pinned revision ${MODEL_REVISION.slice(0, 12)} from ${MODEL_REPO}`);
    laya = await Laya.load({
      repo: MODEL_REPO,
      revision: MODEL_REVISION,
      cacheDir: CACHE_DIR,
      onProgress,
      executionProviders,
      sessionOptions,
    });
    if (path.resolve(laya.modelDir) !== path.resolve(MODEL_DIR)) {
      throw new Error(`unexpected model dir ${laya.modelDir} (expected ${MODEL_DIR})`);
    }
    // Verify what we just downloaded. Loading before verifying is acceptable here because ONNX is a
    // protobuf graph + raw tensors (no code execution on load); a mismatch still aborts before use.
    try {
      await verifyBundle(MODEL_DIR, { force: true, log });
    } catch (e) {
      await laya.close();
      throw e;
    }
    source = "download";
  }
  let calibration = null;
  if (opts.calibration) {
    calibration = typeof opts.calibration === "string" ? JSON.parse(await readFile(path.resolve(PROJECT_ROOT, opts.calibration), "utf8")) : opts.calibration;
    const { applyCalibration } = await import("./calibration.mjs");
    const { before, after } = applyCalibration(laya, calibration);
    log(`calibration applied: ${Object.keys(calibration.temperature_by_options).map((k) => `${k} ${before[k]?.toFixed(3) ?? "-"}->${after[k].toFixed(3)}`).join(", ")}`);
  }
  return { laya, modelDir, source, loadMs: performance.now() - t0, ep, calibration };
}

/**
 * Minimal facade for other workflows: `const d = await createDecider(); await d.decide(state, questions)`.
 */
export async function createDecider(opts = {}) {
  const loaded = await loadLaya(opts);
  return {
    ...loaded,
    decide: (state, questions) => loaded.laya.systemOne(state, questions),
    close: () => loaded.laya.close(),
  };
}
