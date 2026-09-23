/**
 * Triage an inbox: route every ticket to a team, flag urgency, churn risk and refund requests - one call per
 * ticket, all five questions of the `triage` preset in that call.
 *
 *   node demo/examples/triage-inbox.mjs [demo/data/tickets.json] [--mode local]
 */
import { createLaya, gate, readInputs, top } from "../laya.mjs";

const file = process.argv.find((a) => a.endsWith(".json")) ?? new URL("../data/tickets.json", import.meta.url);
const mode = process.argv.includes("--mode") ? process.argv[process.argv.indexOf("--mode") + 1] : "auto";
const tickets = readInputs(file);

const laya = await createLaya({ mode, log: (m) => console.error(`  ${m}`) });
const t0 = performance.now();
const results = await laya.decideMany(tickets, { preset: "triage" });
const wall = performance.now() - t0;

const byTeam = {};
console.log(`${"ticket".padEnd(52)}  ${"team".padEnd(16)} ${"urgency".padEnd(22)} churn  refund  ms`);
for (let i = 0; i < tickets.length; i++) {
  const a = results[i].answers;
  const team = top(a.department);
  const g = gate(a.department); // act = route automatically, ask = route but flag for a human, unsure = human decides
  const label = `${team.label} ${(100 * team.p).toFixed(0)}%${g === "act" ? "" : g === "ask" ? " ?" : " ??"}`;
  (byTeam[g === "unsure" ? "human" : team.label] ??= []).push(i);
  console.log(`${(tickets[i].length > 50 ? tickets[i].slice(0, 47) + "..." : tickets[i]).padEnd(52)}  ${label.padEnd(16)} ${`${top(a.urgency).label} (${a.urgency.score.toFixed(1)})`.padEnd(22)} ${a.churn_risk.noul >= 0.5 ? "yes " : "no  "}   ${a.refund_requested.noul >= 0.5 ? "yes " : "no  "}   ${results[i].routing.ms.toFixed(0)}`);
}
console.log(`\nrouting: ${Object.entries(byTeam).map(([team, ids]) => `${team} -> ${ids.length}`).join(", ")}   (? = route but flag, ?? = let a human decide)`);
console.log(`${tickets.length} tickets in ${wall.toFixed(0)} ms via ${laya.mode} (${(wall / tickets.length).toFixed(1)} ms each end to end; lanes ${[...new Set(results.map((r) => r.routing.lane))].join(", ")})`);
console.log("note: the triage preset is unmeasured - label 30-60 real tickets and run calibrate.mjs before trusting these numbers.");
await laya.close();
