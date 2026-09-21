/**
 * Run bench.mjs once per configuration (separate processes) and print a combined comparison table.
 *
 *   node bench-all.mjs                                          # cpu+pin, cpu:8+pin, cpu (ORT default), webgpu, webgpu+fp16
 *   node bench-all.mjs --configs cpu+pin,webgpu+fp16            # subset; config = ep[:threads][+pin][+fp16]
 *   node bench-all.mjs --configs dml                            # DirectML: loads, but inference fails on this graph (Reshape 'node_view')
 *   node bench-all.mjs --runs 30 --sizes 1,3,10
 */
import { spawn } from "node:child_process";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { parseArgs } from "node:util";

const { values: args } = parseArgs({
  options: {
    configs: { type: "string", default: "cpu+pin,cpu:8+pin,cpu,webgpu,webgpu+fp16" },
    sizes: { type: "string", default: "1,3,10" },
    runs: { type: "string", default: "20" },
    warmup: { type: "string", default: "3" },
  },
});

const configs = args.configs.split(",").map((c) => {
  const [base, ...flags] = c.split("+");
  const [ep, threads] = base.split(":");
  return { ep, threads: threads ? Number(threads) : undefined, pin: flags.includes("pin"), fp16: flags.includes("fp16") };
});

const stamp = new Date().toISOString().replace(/[:.]/g, "-");
await mkdir("results", { recursive: true });
const reports = [];

for (const cfg of configs) {
  const label = `${cfg.ep}${cfg.threads ? `-t${cfg.threads}` : ""}${cfg.pin ? "-pin" : ""}${cfg.fp16 ? "-fp16" : ""}`;
  const out = path.join("results", `${stamp}-${label}.json`);
  const cmd = ["bench.mjs", "--ep", cfg.ep, "--sizes", args.sizes, "--runs", args.runs, "--warmup", args.warmup, "--out", out];
  if (cfg.threads) cmd.push("--threads", String(cfg.threads));
  if (cfg.pin) cmd.push("--pin");
  if (cfg.fp16) cmd.push("--fp16");
  console.log(`\n=== ${label} ===`);
  const code = await new Promise((resolve) => {
    const p = spawn(process.execPath, cmd, { stdio: ["ignore", "ignore", "inherit"], windowsHide: true });
    p.on("exit", resolve);
  });
  try {
    reports.push(JSON.parse(await readFile(out, "utf8")));
  } catch {
    reports.push({ label, ep: cfg.ep, error: `exit code ${code}, no report`, cases: [] });
  }
}

// ---- table -----------------------------------------------------------------------------------------
const sizes = args.sizes.split(",").map(Number);
const f1 = (v) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(1) : "-");
const f0 = (v) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(0) : "-");

let md = `# Laya ONNX benchmark (${stamp})\n\n`;
const m = reports.find((r) => r.machine)?.machine;
if (m) md += `Machine: ${m.model} (${m.logical} threads), ${f0(m.totalMemGiB)} GiB RAM, GPU: ${m.gpu ?? "n/a"}; Node ${reports[0]?.node}, onnxruntime-node ${reports[0]?.ort}\n\n`;
md += `Latency = one \`systemOne\` call answering N questions in one forward pass. ${args.runs} timed runs after ${args.warmup} warm-ups.\n\n`;

md += `## Latency p50 / mean / p95 (ms)\n\n| config | load (s) | ` + sizes.map((n) => `${n} q`).join(" | ") + ` |\n|---|---|` + sizes.map(() => "---").join("|") + `|\n`;
for (const r of reports) {
  if (r.error && !r.cases?.length) {
    md += `| ${r.label} | FAILED | ${sizes.map(() => r.error.slice(0, 60)).join(" | ")} |\n`;
    continue;
  }
  const cells = sizes.map((n) => {
    const c = r.cases.find((x) => x.questions === n);
    if (!c || c.error) return c?.error ? `ERR: ${c.error.slice(0, 40)}` : "-";
    return `${f1(c.latency.p50)} / ${f1(c.latency.mean)} / ${f1(c.latency.p95)}`;
  });
  md += `| ${r.label} | ${f1((r.load?.ms ?? NaN) / 1000)} | ${cells.join(" | ")} |\n`;
}

md += `\n## Resources during the timed runs\n\n| config | q | proc CPU avg (% of 1 core) | machine CPU avg % | RSS max (MiB) | GPU util avg/max % | GPU mem max (MiB) | GPU mem delta on load (MiB) | GPU power avg (W) |\n|---|---|---|---|---|---|---|---|---|\n`;
for (const r of reports) {
  for (const c of r.cases ?? []) {
    if (c.error) continue;
    const g = c.metrics?.gpu;
    const gpuDelta = r.load?.gpuMemAfterMiB != null && r.load?.gpuMemBeforeMiB != null ? r.load.gpuMemAfterMiB - r.load.gpuMemBeforeMiB : NaN;
    md += `| ${r.label} | ${c.questions} | ${f0(c.metrics?.cpuCorePct?.avg)} | ${f1(c.metrics?.cpuMachinePct?.avg)} | ${f0(c.metrics?.rssMiB?.max)} | ${f0(g?.utilPct?.avg)} / ${f0(g?.utilPct?.max)} | ${f0(g?.memUsedMiB?.max)} | ${f0(gpuDelta)} | ${f1(g?.powerW?.avg)} |\n`;
  }
}

md += `\n## Answer consistency across execution providers (3-question set)\n\n| config | intent | P(control_device) | should_execute | urgency |\n|---|---|---|---|---|\n`;
for (const r of reports) {
  const c = (r.cases ?? []).find((x) => x.questions === 3 && !x.error);
  if (!c) continue;
  const a = c.sample;
  md += `| ${r.label} | ${a.intent.choice} | ${a.intent.probabilities.control_device} | ${a.should_execute.noul} | ${a.urgency.score} |\n`;
}

const mdPath = path.join("results", `${stamp}-summary.md`);
await writeFile(mdPath, md);
console.log("\n" + md);
console.log(`Summary written to ${mdPath}`);
