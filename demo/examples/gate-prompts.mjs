/**
 * Gate prompts before they reach an LLM: allow / review / block from the `guard` preset's yes/no questions and
 * risk score. The thresholds are the README's defaults; tune them on labelled prompts (see calibrate.mjs).
 *
 *   node demo/examples/gate-prompts.mjs [demo/data/prompts.json] [--mode local]
 */
import { createLaya, readInputs, top } from "../laya.mjs";

const file = process.argv.find((a) => a.endsWith(".json")) ?? new URL("../data/prompts.json", import.meta.url);
const mode = process.argv.includes("--mode") ? process.argv[process.argv.indexOf("--mode") + 1] : "auto";
const prompts = readInputs(file);

/** allow: nothing flagged; block: a clear injection / jailbreak / extraction attempt; review: the grey zone */
function verdict(a) {
  const flags = ["jailbreak", "injection", "secret_extraction"].filter((q) => a[q].noul >= 0.8);
  const maybe = ["jailbreak", "injection", "secret_extraction"].filter((q) => a[q].noul >= 0.55 && a[q].noul < 0.8);
  if (flags.length || a.risk.score >= 2.5) return { v: "BLOCK", why: flags.length ? flags.join("+") : `risk ${a.risk.score.toFixed(1)}` };
  if (maybe.length || a.risk.score >= 1.5) return { v: "review", why: maybe.length ? `${maybe.join("+")}?` : `risk ${a.risk.score.toFixed(1)}` };
  return { v: "allow", why: "" };
}

const laya = await createLaya({ mode, log: (m) => console.error(`  ${m}`) });
const t0 = performance.now();
const results = await laya.decideMany(prompts, { preset: "guard" });
const wall = performance.now() - t0;
console.log(`${"prompt".padEnd(60)}  verdict  why               jailbreak injection extract  risk`);
const counts = {};
for (let i = 0; i < prompts.length; i++) {
  const a = results[i].answers;
  const { v, why } = verdict(a);
  counts[v] = (counts[v] ?? 0) + 1;
  const shown = prompts[i].replace(/\s+/g, " ");
  console.log(`${(shown.length > 58 ? shown.slice(0, 55) + "..." : shown).padEnd(60)}  ${v.padEnd(7)}  ${why.padEnd(17)} ${a.jailbreak.noul.toFixed(2).padStart(9)} ${a.injection.noul.toFixed(2).padStart(9)} ${a.secret_extraction.noul.toFixed(2).padStart(8)}  ${top(a.risk).label}`);
}
console.log(`\n${Object.entries(counts).map(([k, v]) => `${k} ${v}`).join(", ")}; ${prompts.length} prompts in ${wall.toFixed(0)} ms via ${laya.mode} (${(wall / prompts.length).toFixed(1)} ms each)`);
console.log("note: the guard preset is unmeasured - it catches the obvious cases here; label real prompts before relying on it.");
await laya.close();
