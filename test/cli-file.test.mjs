/** CLI request-file parsing/transport checks with a local HTTP stub; no models or GPU needed. */
import { test, before, beforeEach, after } from "node:test";
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { createServer } from "node:http";
import { mkdir, mkdtemp, writeFile, rm } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import { PROJECT_ROOT } from "../src/models.mjs";

const run = promisify(execFile);
const questions = { refund: { type: "predicate", instructions: "Refund requested?" } };
const env = Object.fromEntries(Object.entries(process.env).filter(([k]) => !["LAYA_DEVICE", "LAYA_MODEL", "LAYA_SIDECAR", "LAYA_LANES"].includes(k)));
const calls = [];
let server, dir, port, healthCalls;
const writeRequest = async (body, name = "ticket with spaces.json", bom = false) => {
  await writeFile(path.join(dir, name), (bom ? "\uFEFF" : "") + JSON.stringify(body) + "\n");
  return name;
};
const invoke = (args, { script = "ask.mjs", environment = {}, mode = ["--mode", "sidecar"] } = {}) =>
  run(process.execPath, [path.join(PROJECT_ROOT, script), "--port", String(port), ...mode, ...args], {
    cwd: dir, env: { ...env, ...environment }, timeout: 15_000, maxBuffer: 8 * 1024 * 1024,
  });
const json = async (args, options) => JSON.parse((await invoke([...args, "--json"], options)).stdout);

before(async () => {
  await mkdir(path.join(PROJECT_ROOT, ".laya"), { recursive: true });
  dir = await mkdtemp(path.join(PROJECT_ROOT, ".laya", "cli file-"));
  server = createServer(async (req, res) => {
    res.setHeader("content-type", "application/json");
    if (req.url === "/health") {
      healthCalls++;
      const graphs = ["auto", "on", "off"], accelerations = ["auto", "on", "off"];
      return res.end(JSON.stringify({ service: "laya", version: "test", status: "ready", pid: process.pid,
        lanes: ["cpu:8"], lanesLoading: [], idleS: 300, models: { available: {
          laya: { graphs }, multilingual: { graphs, accelerations }, vega: { graphs, accelerations },
        } } }));
    }
    let raw = "";
    for await (const chunk of req) raw += chunk;
    const body = JSON.parse(raw);
    calls.push(body);
    const answers = Object.fromEntries(Object.entries(body.questions).map(([id, q]) => {
      if (q.type === "choice") {
        const keys = Array.isArray(q.criteria) ? q.criteria : Object.keys(q.criteria);
        return [id, { type: "choice", choice: keys[0], probabilities: Object.fromEntries(keys.map((k, i) => [k, i === 0 ? 1 : 0])) }];
      }
      if (q.type === "score") return [id, { type: "score", score: 0,
        legend: Object.fromEntries(q.criteria.map((s, i) => [String(i), s])),
        probabilities: Object.fromEntries(q.criteria.map((_, i) => [String(i), i === 0 ? 1 : 0])) }];
      return [id, { type: "noul", noul: 0.75 }];
    }));
    res.end(JSON.stringify({ answers, state: body.state, usage: { input_tokens: 1 },
      routing: { model: body.model, device: "cpu", lane: "cpu:8", contextTokens: body.maxLen, ms: 1, queueMs: 0 } }));
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  port = server.address().port;
});
beforeEach(() => { calls.length = 0; healthCalls = 0; });
after(async () => {
  if (server) await new Promise(resolve => server.close(resolve));
  if (dir) await rm(dir, { recursive: true, force: true });
});

test("--file reads a complete UTF-8/BOM request relative to the caller, with file settings above env defaults", async () => {
  const body = { state: "تم خصم المبلغ مرتين. Please refund it.", questions, model: "multilingual", device: "cpu",
    maxLen: 1024, acceleration: "off", graphs: "off", preset: "triage", calibration: null };
  const file = await writeRequest(body, undefined, true);
  const result = await json(["--file", file], { environment: { LAYA_MODEL: "vega", LAYA_DEVICE: "cuda" } });
  assert.equal(result.backend, "remote");
  assert.equal(result.state, body.state);
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0], { ...body, readout: "auto", questions: { refund: { ...questions.refund, type: "noul" } } });
  assert.doesNotMatch(JSON.stringify(result.state), /state.*questions/);
});

test("CLI flags replace file model, preset and question settings; omitted model settings use env defaults", async () => {
  const file = await writeRequest({ state: "original state", questions, model: "multilingual", device: "cpu",
    maxLen: 1024, acceleration: "off", graphs: "off", preset: "triage", calibration: null });
  const replacement = { other: { type: "predicate", instructions: "Another question?" } };
  await writeFile(path.join(dir, "replacement.json"), JSON.stringify(replacement));
  const result = await json(["--file", file, "--model", "vega", "--device", "cuda", "--max-len", "4096",
    "--readout", "engine", "--acceleration", "auto", "--graphs", "on", "--preset", "guard", "--questions", "replacement.json"]);
  assert.equal(result.routing.model, "vega");
  assert.equal(result.routing.contextTokens, 4096);
  assert.equal(calls[0].device, "cuda");
  assert.equal(calls[0].readout, "engine");
  assert.equal(calls[0].acceleration, "auto");
  assert.equal(calls[0].graphs, "on");
  assert.equal(calls[0].preset, "guard");
  assert.equal(calls[0].state, "original state");
  assert.deepEqual(calls[0].questions, { other: { ...replacement.other, type: "noul" } });

  const basic = await writeRequest({ state: "env defaults", questions });
  await json(["--file", basic], { environment: { LAYA_MODEL: "multilingual", LAYA_DEVICE: "cpu" } });
  assert.equal(calls[1].model, "multilingual");
  assert.equal(calls[1].device, "cpu");
  assert.equal(calls[1].maxLen, 8192);
});

test("file state is verbatim, including falsy values; file text uses its preset template", async () => {
  for (const state of [null, false, 0, "", { ticket: "refund" }, [{ role: "user", content: "refund" }]]) {
    const file = await writeRequest({ state, text: "ignored", questions });
    const result = await json(["--file", file]);
    assert.deepEqual(result.state, state);
    assert.deepEqual(calls.at(-1).state, state);
  }
  const file = await writeRequest({ text: "  Please refund the duplicate  ", preset: "triage" });
  const result = await json(["--file", file]);
  assert.deepEqual(result.state, { message: "Please refund the duplicate" });
  assert.ok(calls.at(-1).questions.department);
  const positional = await json(["--preset", "triage", "Plain CLI text"]);
  assert.deepEqual(positional.state, { message: "Plain CLI text" });
});

test("file execution/calibration controls reach the request and explicit CLI overrides win", async () => {
  const body = { state: "refund", questions, lane: "cpu:8", policy: "prefer-cpu", deadlineMs: "50",
    exec: { graph: true }, calibration: { temperature_by_options: { "noul:2": 2 } } };
  const file = await writeRequest(body);
  await json(["--file", file]);
  assert.equal(calls[0].lane, "cpu:8");
  assert.equal(calls[0].policy, "prefer-cpu");
  assert.equal(calls[0].deadlineMs, 50);
  assert.deepEqual(calls[0].exec, { graph: true });
  assert.deepEqual(calls[0].calibration, { ...body.calibration, file: "request" });
  const calibrationFile = path.join(dir, "temperature.json");
  await writeFile(calibrationFile, JSON.stringify({ temperature_by_options: { "noul:2": 4 } }));
  await json(["--file", file, "--lane", "cpu:16", "--policy", "auto", "--no-graph", "--calibration", calibrationFile]);
  assert.equal(calls[1].lane, "cpu:16");
  assert.equal(calls[1].policy, "auto");
  assert.deepEqual(calls[1].exec, { graph: false });
  assert.equal(calls[1].calibration.temperature_by_options["noul:2"], 4);
  await json(["--file", await writeRequest({ state: "refund", questions, calibration: null })]);
  assert.equal(calls[2].calibration, null);
});

test("bad files, invalid controls and mixed input sources fail before connecting or loading", async () => {
  const invalid = [
    ["{broken", /invalid JSON/], ["[]", /one JSON request object/], ["null", /one JSON request object/],
    [JSON.stringify({ questions }), /provide state or non-empty text/],
    [JSON.stringify({ text: "  " }), /provide state or non-empty text/],
    [JSON.stringify({ state: "x", mode: "sidecar" }), /unsupported request field/],
    [JSON.stringify({ state: "x", policy: "unknown" }), /policy must be/],
    [JSON.stringify({ state: "x", deadlineMs: 0 }), /deadlineMs must be/],
    [JSON.stringify({ state: "x", exec: [] }), /exec must be an object/],
    [JSON.stringify({ state: "x", calibration: "table.json" }), /calibration needs/],
    [JSON.stringify({ state: "x", questions: { q: { type: "score", instructions: "?", levels: ["one"] } } }), /ordered array/],
  ];
  for (const [contents, message] of invalid) {
    await writeFile(path.join(dir, "bad.json"), contents);
    await assert.rejects(invoke(["--file", "bad.json"]), e => e.code === 2 && e.stdout === "" && message.test(e.stderr));
  }
  await assert.rejects(invoke(["--file", "missing.json"]), e => e.code === 2 && /--file.*ENOENT/.test(e.stderr));
  const file = await writeRequest({ state: "valid", questions });
  for (const conflict of [["other text"], ["--state", "other.json"], ["--text-file", "other.txt"], ["--start"], ["--stop"], ["--status"]]) {
    await assert.rejects(invoke(["--file", file, ...conflict]), e => e.code === 2 && /--file cannot be combined/.test(e.stderr));
  }
  assert.equal(healthCalls, 0);
  assert.equal(calls.length, 0);
});

test("skill --file delegation preserves caller paths and JSON/human output modes", async () => {
  const file = await writeRequest({ state: "refund", questions, model: "multilingual", device: "cpu" });
  const options = { script: "skills/laya-decisions/scripts/laya.mjs", mode: [] };
  const result = JSON.parse((await invoke(["--file", file], options)).stdout);
  assert.equal(result.state, "refund");
  assert.equal(result.backend, "remote");
  assert.equal(result.routing.model, "multilingual");
  const pretty = await invoke(["--file", file, "--pretty", "--no-color"], options);
  assert.match(pretty.stdout, /refund\s+yes/);
  assert.doesNotMatch(pretty.stdout, /"answers"/);
  assert.equal(calls.length, 2);
});
