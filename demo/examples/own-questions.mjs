/**
 * Your own questions, no preset: an email as a JSON state and three ad-hoc questions (demo/data/email-questions.json).
 * Shows the full probability distributions - the point being that ad-hoc questions are unmeasured: read them as
 * hints and label a few dozen examples (calibrate.mjs) before gating on them.
 *
 *   node demo/examples/own-questions.mjs [--mode local]
 */
import { readFile } from "node:fs/promises";
import { createLaya, readInputs, top } from "../laya.mjs";

const mode = process.argv.includes("--mode") ? process.argv[process.argv.indexOf("--mode") + 1] : "auto";
const emails = readInputs(new URL("../data/emails.json", import.meta.url));
const questions = JSON.parse(await readFile(new URL("../data/email-questions.json", import.meta.url), "utf8"));

const laya = await createLaya({ mode, log: (m) => console.error(`  ${m}`) });
const results = await laya.decideMany(emails.map((e) => ({ state: e.state })), { questions });
for (let i = 0; i < emails.length; i++) {
  const { from, subject, body } = emails[i].state;
  const a = results[i].answers;
  console.log(`\n${subject}  (${from})\n  ${body}`);
  console.log(`  needs_reply  ${top(a.needs_reply).label.padEnd(10)} P(true) ${a.needs_reply.noul.toFixed(3)}`);
  console.log(`  category     ${top(a.category).label.padEnd(10)} ${Object.entries(a.category.probabilities).sort((x, y) => y[1] - x[1]).map(([k, v]) => `${k} ${(100 * v).toFixed(0)}%`).join(", ")}`);
  console.log(`  priority     ${top(a.priority).label.padEnd(10)} score ${a.priority.score.toFixed(2)}/3  ${Object.entries(a.priority.probabilities).map(([k, v]) => `${a.priority.legend[k]} ${(100 * v).toFixed(0)}%`).join(", ")}`);
}
console.log(`\n${emails.length} emails via ${laya.mode}; ${results[0].usage.input_tokens} tokens for the first (state + 3 questions), ${results[0].routing.lane} ${results.map((r) => r.routing.ms).sort((x, y) => x - y)[Math.floor(results.length / 2)].toFixed(0)} ms p50`);
console.log("to measure these questions: write demo/data/emails.eval.json with gold answers and run\n  node calibrate.mjs --preset <a preset file with these questions> --eval <that file>");
await laya.close();
