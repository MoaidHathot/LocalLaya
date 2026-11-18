/**
 * The request -> decision logic shared by serve.mjs (HTTP) and demo/laya.mjs (in-process "local" mode), so both
 * accept the same body and return the same shape:
 *
 *   body: { text?, state?, preset?, questions?, model?, device?, maxLen?, readout?, acceleration?, graphs?, lane?, policy?, deadlineMs?, calibration?, exec? }
 *   ->    { answers, usage, routing, state, questions, preset, calibration }
 *
 * - `text` is wrapped by the preset's state template; `state` is used verbatim.
 * - `questions` replaces the preset's question set (validated and normalized here, before inference).
 * - the calibration table is, in order: the request's `calibration.temperature_by_options`, else the
 *   model-scoped per-preset file (re-read when its mtime changes), else the shipped temperatures.
 * - `lane` / `policy` / `deadlineMs` / `exec` are the router's per-call overrides (see ep-router.mjs).
 * Errors carry `status` (400 for bad input, 503 for a lane still loading) so an HTTP layer can map them.
 */
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { DEFAULT_PRESET } from "../data/presets.mjs";
import { modelOptions } from "./models.mjs";
import { normalizeQuestions } from "./question-schema.mjs";
export { validateQuestions } from "./question-schema.mjs";

export const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const POLICIES = ["auto", "prefer-gpu", "prefer-cpu", "min-cpu"];

export const httpError = (status, message, extra = {}) => Object.assign(new Error(message), { status, ...extra });

/** Shared validation for HTTP bodies and complete CLI request files. */
export function validateCalibration(calibration) {
  if (calibration != null && (!calibration.temperature_by_options || typeof calibration.temperature_by_options !== "object" || Array.isArray(calibration.temperature_by_options))) throw httpError(400, "calibration needs a temperature_by_options object");
}

export function executionOptions(body) {
  const opts = {};
  if (body.lane) opts.lane = body.lane;
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
    if (!body.exec || typeof body.exec !== "object" || Array.isArray(body.exec)) throw httpError(400, "exec must be an object, e.g. { graph: false }");
    opts.exec = body.exec;
  }
  return opts;
}

/** Per-preset calibration tables, re-read when the file changes. `override` = one table for every preset (--calibration). */
export class CalibrationCache {
  constructor({ root = PROJECT_ROOT, override = null, overrideModel = "laya" } = {}) {
    this.root = root;
    this.override = override;
    this.overrideModel = overrideModel;
    this.cache = new Map(); // preset -> { file, mtimeMs, table }
  }
  async for(presetName, model = "laya") {
    if (model === "vega") return null;
    const key = `${model}:${presetName}`;
    const file = path.resolve(this.root, (model === this.overrideModel ? this.override : null) ?? `calibration/${model === "laya" ? "" : `${model}/`}${presetName}.json`);
    let mtimeMs;
    try {
      mtimeMs = (await stat(file)).mtimeMs;
    } catch {
      this.cache.delete(key);
      return null;
    }
    const cached = this.cache.get(key);
    if (cached && cached.file === file && cached.mtimeMs === mtimeMs) return cached.table;
    const table = { ...JSON.parse(await readFile(file, "utf8")), file: path.relative(this.root, file).replace(/\\/g, "/") };
    this.cache.set(key, { file, mtimeMs, table });
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
  if (!body || typeof body !== "object" || Array.isArray(body)) throw httpError(400, "request body must be a JSON object");
  const selection = router.selection ? router.selection(body) : modelOptions(body);
  if (!router.selection && (selection.model !== "laya" || selection.device !== "auto")) throw httpError(400, "model/device selection requires ModelRouter; this router only serves the original Laya lanes");
  const presetName = body.preset ?? DEFAULT_PRESET;
  const preset = presets[presetName];
  if (!preset || preset.invalid) throw httpError(400, preset?.invalid ? `preset ${presetName} is invalid: ${preset.description}` : `unknown preset ${presetName}; have ${Object.keys(presets).join(", ")}`);
  const questions = normalizeQuestions(body.questions ?? preset.questions);
  let state;
  if (body.state !== undefined) state = body.state;
  else if (typeof body.text === "string" && body.text.trim()) state = preset.state(body.text.trim());
  else throw httpError(400, "provide text (string) or state (any JSON)");
  validateCalibration(body.calibration);
  const opts = { ...selection, ...executionOptions(body), calibration: body.calibration === null ? null : body.calibration?.temperature_by_options ? { temperature_by_options: body.calibration.temperature_by_options, file: body.calibration.file ?? "request" } : await calibration.for(presetName, selection.model) };
  if (body.lane) {
    if (!router.selection && !router.lanes.has(body.lane)) {
      if (router.pendingLanes.includes(body.lane)) throw httpError(503, `lane ${body.lane} is still loading`, { retryAfterMs: 500 });
      throw httpError(400, `lane ${body.lane} not loaded; have ${[...router.lanes.keys()].join(", ")}`);
    }
  }
  const r = await router.decide(state, questions, opts);
  return { answers: r.answers, usage: r.usage, routing: r.routing, state, questions, preset: body.questions ? null : presetName, calibration: opts.calibration?.file ?? null };
}
