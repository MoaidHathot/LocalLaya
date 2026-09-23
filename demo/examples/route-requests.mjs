/**
 * Route developer requests to the right handler: the `dev-request` preset says what the user wants (task),
 * in which language, and how big it is; this script maps that to "which tool would run" - the decision Laya
 * is for. Nothing here calls the tools.
 *
 *   node demo/examples/route-requests.mjs [demo/data/requests.json] [--mode local]
 */
import { createLaya, gate, readInputs, top } from "../laya.mjs";

const file = process.argv.find((a) => a.endsWith(".json")) ?? new URL("../data/requests.json", import.meta.url);
const mode = process.argv.includes("--mode") ? process.argv[process.argv.indexOf("--mode") + 1] : "auto";
const requests = readInputs(file);

/** task x language -> handler. Deterministic code, not the model: the model only picks task / language. */
function handler(task, language) {
  if (task === "validate") return language === "json" ? "json.parse()" : language === "sql" ? "sql-linter" : language === "other" ? "yaml/regex validator" : `validator(${language})`;
  if (task === "run_command") return language === "powershell" || language === "none" ? "shell (ask to confirm)" : `runner(${language})`;
  if (task === "fix_bug" || task === "write_code" || task === "convert") return `coding LLM (${language})`;
  if (task === "explain") return "small LLM";
  return "chat LLM";
}

const laya = await createLaya({ mode, log: (m) => console.error(`  ${m}`) });
const t0 = performance.now();
const results = await laya.decideMany(requests, { preset: "dev-request" });
const wall = performance.now() - t0;
console.log(`${"request".padEnd(56)}  ${"task".padEnd(18)} ${"language".padEnd(16)} ${"effort".padEnd(20)} -> handler`);
for (let i = 0; i < requests.length; i++) {
  const a = results[i].answers;
  const task = top(a.task);
  const lang = top(a.language);
  const sure = gate(a.task) !== "unsure" && gate(a.language) !== "unsure";
  const shown = requests[i].replace(/\s+/g, " ");
  console.log(`${(shown.length > 54 ? shown.slice(0, 51) + "..." : shown).padEnd(56)}  ${`${task.label} ${(100 * task.p).toFixed(0)}%`.padEnd(18)} ${`${lang.label} ${(100 * lang.p).toFixed(0)}%`.padEnd(16)} ${top(a.effort).label.split(":")[0].padEnd(20)} -> ${sure ? handler(task.label, lang.label) : `ask the user (task ${(100 * task.p).toFixed(0)}%, language ${(100 * lang.p).toFixed(0)}%)`}`);
}
console.log(`\n${requests.length} requests in ${wall.toFixed(0)} ms via ${laya.mode} (${(wall / requests.length).toFixed(1)} ms each; inference p50 ${results.map((r) => r.routing.ms).sort((x, y) => x - y)[Math.floor(results.length / 2)].toFixed(0)} ms on ${[...new Set(results.map((r) => r.routing.lane))].join(", ")})`);
console.log("measured on 40 labelled requests: task 0.75, language 0.85 accuracy (calibration/dev-request.json applied). 'effort' is unlabelled.");
await laya.close();
