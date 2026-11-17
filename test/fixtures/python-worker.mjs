// Protocol fixture: run by PythonBackend with Node as its executable; no model or Python dependency.
import { createInterface } from "node:readline";
const send = (o) => process.stdout.write(JSON.stringify(o) + "\n");
send({ ready: true, model: "vega", device: "cpu", maxLen: 73728, loadMs: 1, pid: process.pid });
for await (const line of createInterface({ input: process.stdin })) {
  const m = JSON.parse(line);
  if (m.op === "close") process.exit(0);
  if (m.state === "die") process.exit(7);
  if (m.state === "hang") continue;
  if (m.state === "malformed") { send({ id: m.id, result: {} }); continue; }
  if (m.state === "bad-input") { send({ id: m.id, error: "invalid question", status: 400 }); continue; }
  send({ id: m.id, result: { answers: { q: { type: "noul", noul: 0.75 } }, usage: { input_tokens: 10, output_tokens: 0 } }, exec: { device: "cpu", mode: "torch-eager", readout: m.readout }, ms: 1 });
}
