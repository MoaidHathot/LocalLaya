/**
 * The request -> decision logic shared by serve.mjs (HTTP) and demo/laya.mjs (in-process "local" mode), so both
 * accept the same body and return the same shape:
 *
 *   body: { text?, state?, preset?, questions?, lane?, policy?, deadlineMs?, calibration?, exec? }
 *   ->    { answers, usage, routing, state, questions, preset, calibration }
 *
 * - `text` is wrapped by the preset's state template; `state` is used verbatim.
 * - `questions` replaces the preset's question set (validated here, before the model sees them).
 * - the calibration table is, in order: the request's `calibration.temperature_by_options`, else the
 *   per-preset file calibration/<preset>.json (re-read when its mtime changes), else the shipped temperatures.
 * - `lane` / `policy` / `deadlineMs` / `exec` are the router's per-call overrides (see ep-router.mjs).
 * Errors carry `status` (400 for bad input, 503 for a lane still loading) so an HTTP layer can map them.
 */
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { DEFAULT_PRESET } from "../data/presets.mjs";

export const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const POLICIES = ["auto", "prefer-gpu", "prefer-cpu", "min-cpu"];

export const httpError = (status, message, extra = {}) => Object.assign(new Error(message), { status, ...extra });

export function validateQuestions(q) {
  if (!q || typeof q !== "object" || Array.isArray(q) || !Object.keys(q).length) throw httpError(400, "questions must be a non-empty object");
  for (const [id, x] of Object.entries(q)) {
    if (!x || !["choice", "score", "noul"].includes(x.type)) throw httpError(400, `question ${id}: type must be choice | score | noul`);
    if (x.instructions === undefined) throw httpError(400, `question ${id}: instructions required`);
    if (x.type === "choice" && !(Array.isArray(x.criteria) ? x.criteria.length >= 2 : x.criteria && Object.keys(x.criteria).length >= 2)) throw httpError(400, `question ${id}: choice needs >= 2 criteria`);
    if (x.type === "score" && !(Array.isArray(x.criteria) && x.criteria.length >= 2)) throw httpError(400, `question ${id}: score needs an ordered array of >= 2 levels`);
  }
}

/** Per-preset calibration tables, re-read when the file changes. `override` = one table for every preset (--calibration). */
export class CalibrationCache {
  constructor({ root = PROJECT_ROOT, override = null } = {}) {
    this.root = root;
    this.override = override;
    this.cache = new Map(); // preset -> { file, mtimeMs, table }
  }
  async for(presetName) {
    const file = path.resolve(this.root, this.override ?? `calibration/${presetName}.json`);
    let mtimeMs;
    try {
      mtimeMs = (await stat(file)).mtimeMs;
    } catch {
      this.cache.delete(presetName);
      return null;
    }
    const cached = this.cache.get(presetName);
    if (cached && cached.file === file && cached.mtimeMs === mtimeMs) return cached.table;
    const table = { ...JSON.parse(await readFile(file, "utf8")), file: path.relative(this.root, file).replace(/\\/g, "/") };
    this.cache.set(presetName, { file, mtimeMs, table });
    return table;
  }
}

/**
 * Turn a request body into the router call and run it.
 * @param {import("./ep-router.mjs").LayaRouter} router
 * @param {object} body
 * @param {object} presets   from loadPresets()
 * @param {CalibrationCache} calibration
 */
export async function decideRequest(router, body, presets, calibration) {
  if (!body || typeof body !== "object") throw httpError(400, "request body must be a JSON object");
  const presetName = body.preset ?? DEFAULT_PRESET;
  const preset = presets[presetName];
  if (!preset || preset.invalid) throw httpError(400, preset?.invalid ? `preset ${presetName} is invalid: ${preset.description}` : `unknown preset ${presetName}; have ${Object.keys(presets).join(", ")}`);
  const questions = body.questions ?? preset.questions;
  validateQuestions(questions);
  let state;
  if (body.state !== undefined) state = body.state;
  else if (typeof body.text === "string" && body.text.trim()) state = preset.state(body.text.trim());
  else throw httpError(400, "provide text (string) or state (any JSON)");
  const opts = { calibration: body.calibration?.temperature_by_options ? { temperature_by_options: body.calibration.temperature_by_options, file: body.calibration.file ?? "request" } : await calibration.for(presetName) };
  if (body.lane) {
    if (!router.lanes.has(body.lane)) {
      if (router.pendingLanes.includes(body.lane)) throw httpError(503, `lane ${body.lane} is still loading`, { retryAfterMs: 500 });
      throw httpError(400, `lane ${body.lane} not loaded; have ${[...router.lanes.keys()].join(", ")}`);
    }
    opts.lane = body.lane;
  }
  if (body.policy !== undefined) {
    if (!POLICIES.includes(body.policy)) throw httpError(400, `policy must be one of ${POLICIES.join(", ")}`);
    opts.policy = body.policy;
  }
  if (body.deadlineMs !== undefined) {
    const d = Number(body.deadlineMs);
    if (!(d > 0)) throw httpError(400, "deadlineMs must be a positive number");
    opts.deadlineMs = d;
  }
  if (body.exec !== undefined) {
    if (!body.exec || typeof body.exec !== "object") throw httpError(400, "exec must be an object, e.g. { graph: false }");
    opts.exec = body.exec;
  }
  const r = await router.decide(state, questions, opts);
  return { answers: r.answers, usage: r.usage, routing: r.routing, state, questions, preset: body.questions ? null : presetName, calibration: opts.calibration?.file ?? null };
}
