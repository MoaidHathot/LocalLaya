/** Real request-schema checks: prepared Laya, Vega and Multilingual models, plus CUDA. */
import { test, after } from "node:test";
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import { createLaya } from "../demo/laya.mjs";
import { ensureSidecar, createClient, discover, stop, PROJECT_ROOT } from "../src/sidecar-client.mjs";

const PORT = 8794;
const run = promisify(execFile);
const request = JSON.parse(await readFile(new URL("../demo/data/ticket-request.json", import.meta.url), "utf8"));
const canonical = {
  refund: { type: "noul", instructions: request.questions.refund.instructions },
  team: { type: "choice", instructions: request.questions.team.instructions, criteria: request.questions.team.choices },
  severity: { type: "score", instructions: request.questions.severity.instructions, criteria: request.questions.severity.levels },
};
const env = Object.fromEntries(Object.entries(process.env).filter(([k]) => !["LAYA_DEVICE", "LAYA_MODEL", "LAYA_SIDECAR", "LAYA_LANES"].includes(k)));
let ownsSidecar = false;
const valid = (r) => {
  assert.deepEqual(Object.keys(r.answers), ["refund", "team", "severity"]);
  assert.equal(r.answers.refund.type, "noul");
  assert.ok(r.answers.refund.noul >= 0 && r.answers.refund.noul <= 1);
  assert.equal(r.answers.team.choice, "billing");
  assert.ok(r.answers.severity.score >= 0 && r.answers.severity.score <= 2);
  assert.deepEqual(Object.values(r.answers.severity.legend), ["Low", "Medium", "High"]);
};
after(async () => { if (ownsSidecar) await stop({ port: PORT, timeoutMs: 30_000 }).catch(() => {}); });

test("the user's JSON works over HTTP for all three models, both CLIs and the skill wrapper", { timeout: 240_000 }, async () => {
  assert.equal((await discover({ port: PORT })).state, "none", "test port must be free");
  await mkdir(path.join(PROJECT_ROOT, ".laya"), { recursive: true });
  const dir = await mkdtemp(path.join(PROJECT_ROOT, ".laya", "question aliases-"));
  const client = createClient({ port: PORT });
  try {
    ownsSidecar = true;
    await ensureSidecar({ port: PORT, model: "vega", device: "cuda", acceleration: "off", graphs: "off", idle: "5m" });
    const first = await client.decide(request); // Entire sample file, including its original field names.
    valid(first);
    assert.equal(first.state, request.state);
    assert.deepEqual(first.questions, canonical);
    for (const model of ["vega", "laya", "multilingual"]) {
      const controls = { model, ...(model === "laya" ? {} : { acceleration: "off" }) };
      const aliased = await client.decide({ ...request, ...controls });
      const reference = await client.decide({ ...request, ...controls, questions: canonical });
      valid(aliased);
      assert.deepEqual(aliased.answers, reference.answers, `${model}: aliases must have the same inference semantics`);
      assert.deepEqual(aliased.questions, canonical);
    }
    await assert.rejects(client.decide({ ...request, model: "vega", questions: {
      team: { ...request.questions.team, criteria: ["different", "options"] },
    } }), (e) => e.status === 400 && /criteria and choices conflict/.test(e.message));
    assert.equal((await client.health()).models.active.model, "multilingual", "invalid questions must not trigger a model reload");

    await writeFile(path.join(dir, "state.json"), JSON.stringify(request.state) + "\n");
    await writeFile(path.join(dir, "questions.json"), JSON.stringify(request.questions) + "\n");
    for (const script of ["ask.mjs", "demo/cli.mjs", "skills/laya-decisions/scripts/laya.mjs"]) {
      const mode = script === "skills/laya-decisions/scripts/laya.mjs" ? ["--sidecar"] : ["--mode", "sidecar"];
      const { stdout } = await run(process.execPath, [path.join(PROJECT_ROOT, script), ...mode, "--port", String(PORT),
        "--model", "multilingual", "--device", "cuda", "--acceleration", "off", "--graphs", "off",
        "--state", "state.json", "--questions", "questions.json", "--json"], { cwd: dir, env, timeout: 120_000 });
      valid(JSON.parse(stdout));
    }
    await writeFile(path.join(dir, "ticket request.json"), "\uFEFF" + JSON.stringify({ ...request, model: "multilingual",
      device: "cuda", acceleration: "off", graphs: "off", calibration: null }) + "\n");
    for (const script of ["ask.mjs", "skills/laya-decisions/scripts/laya.mjs"]) {
      const { stdout } = await run(process.execPath, [path.join(PROJECT_ROOT, script), "--file", "ticket request.json",
        "--sidecar", "--port", String(PORT), "--json"], { cwd: dir, env, timeout: 120_000 });
      const result = JSON.parse(stdout);
      valid(result);
      assert.equal(result.state, request.state);
      assert.equal(result.routing.model, "multilingual");
      assert.equal(result.backend, "remote");
    }
  } finally {
    client.close();
    try { await stop({ port: PORT, timeoutMs: 30_000 }); ownsSidecar = false; }
    finally { await rm(dir, { recursive: true, force: true }); }
  }
});

test("local --file and library requests accept aliases and preserve caller-owned questions", { timeout: 120_000 }, async () => {
  const { stdout } = await run(process.execPath, ["ask.mjs", "--model", "multilingual", "--local", "--json",
    "--file", "demo/data/ticket-request.json"], { cwd: PROJECT_ROOT, env, timeout: 90_000 });
  const result = JSON.parse(stdout);
  valid(result);
  assert.equal(result.state, request.state);
  assert.equal(result.routing.profile.lifetime, "one-shot");
  assert.equal(result.routing.exec.acceleration.backend, "eager");
  const local = await createLaya({ model: "laya", mode: "local", device: "cuda", graphs: "off", lifetime: "one-shot" });
  const original = structuredClone(request.questions);
  try {
    const r = await local.decide({ state: request.state }, { questions: request.questions, calibration: null });
    valid(r);
    assert.deepEqual(r.questions, canonical);
    assert.deepEqual(request.questions, original);
    const direct = await local.info.router.backend.decide(request.state, request.questions);
    valid(direct); // Original EP router also supports aliases without the HTTP/model-router layer.
  } finally { await local.close(); }
});
