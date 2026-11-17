/** Model selection is independent of the legacy execution-provider (EP) router. No inference imports. */
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const PYTHON_MANIFEST = JSON.parse(readFileSync(new URL("../data/python-models.json", import.meta.url), "utf8"));
export const MODELS = {
  laya: { name: "LocalLaya / English ONNX", maxLen: 512, backend: "onnx" },
  ...Object.fromEntries(Object.entries(PYTHON_MANIFEST.models).map(([id, p]) => [id, { ...p, backend: "python" }])),
};
export const MODEL_NAMES = Object.keys(MODELS);
export const DEVICES = ["auto", "cpu", "cuda"];
export const READOUTS = ["auto", "engine"];
export const ACCELERATIONS = ["auto", "off"];
MODELS.vega.accelerations = ACCELERATIONS;
MODELS.multilingual.accelerations = ACCELERATIONS;
export const PYTHON = process.env.LAYA_MODEL_PYTHON ?? path.join(PROJECT_ROOT, ".venv-laya", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
export const MODEL_TIMEOUT_MS = 600_000;

const bad = (message) => Object.assign(new Error(message), { status: 400, code: "BAD_REQUEST" });
export function modelName(value = "laya") {
  if (typeof value !== "string") throw bad("model must be a string");
  const name = { "vega-u": "vega", "laya-multilingual": "multilingual", english: "laya" }[value] ?? value;
  if (!Object.hasOwn(MODELS, name)) throw bad(`model must be one of ${MODEL_NAMES.join(", ")}`);
  return name;
}

/** Resolve only model controls. Backend-specific controls are refused rather than silently ignored. */
export function modelOptions(o = {}, defaults = {}) {
  const model = modelName(o.model ?? defaults.model ?? "laya");
  const sameModel = model === modelName(defaults.model ?? "laya");
  const device = o.device ?? defaults.device ?? "auto";
  if (!DEVICES.includes(device)) throw bad(`device must be one of ${DEVICES.join(", ")}`);
  if (o.lane !== undefined && (typeof o.lane !== "string" || !o.lane)) throw bad("lane must be a non-empty string");
  const maxLen = o.maxLen ?? (sameModel ? defaults.maxLen : undefined) ?? MODELS[model].maxLen;
  if (!Number.isSafeInteger(maxLen) || maxLen < 64 || maxLen > MODELS[model].maxLen) throw bad(`maxLen for ${model} must be an integer from 64 to ${MODELS[model].maxLen}`);
  const readout = o.readout ?? (sameModel ? defaults.readout : undefined) ?? "auto";
  if (!READOUTS.includes(readout)) throw bad(`readout must be one of ${READOUTS.join(", ")}; fitted heads are not loaded by this adapter`);
  if (model !== "vega" && readout !== "auto") throw bad("readout is a Vega option; use model=vega");
  if (model === "laya" && maxLen !== 512) throw bad("the pinned LocalLaya ONNX bundle uses maxLen=512; select multilingual or vega for another context budget");
  if (model !== "laya") {
    for (const k of ["lane", "policy", "deadlineMs", "exec"]) if (o[k] !== undefined) throw bad(`${k} controls the LocalLaya ONNX lanes; use device for ${model}`);
  }
  if (model === "vega" && o.calibration != null) throw bad("LocalLaya temperature tables cannot be applied to Vega; Vega uses its own checkpoint calibration");
  if (model === "laya" && o.acceleration !== undefined) throw bad("acceleration controls the Python models; use model=vega or multilingual (original Laya uses exec.graph)");
  const acceleration = o.acceleration ?? (sameModel ? defaults.acceleration : undefined) ?? "auto";
  if (!ACCELERATIONS.includes(acceleration)) throw bad("acceleration must be auto or off");
  return { model, device, maxLen, readout, ...(model !== "laya" ? { acceleration } : {}) };
}

export function modelArgs(args) {
  return Object.fromEntries([
    ["model", args.model], ["device", args.device], ["maxLen", args["max-len"] === undefined ? undefined : Number(args["max-len"])], ["readout", args.readout], ["acceleration", args.acceleration],
  ].filter(([, v]) => v !== undefined));
}

export const MODEL_CLI_OPTIONS = {
  model: { type: "string" }, device: { type: "string" }, "max-len": { type: "string" }, readout: { type: "string" }, acceleration: { type: "string" },
};
