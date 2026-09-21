/**
 * Re-verify the cached model bundle against the pinned sizes / SHA256 values (forces a re-hash).
 *   node verify-model.mjs
 */
import { bundleOnDisk, MODEL_DIR, MODEL_REVISION, verifyBundle } from "./src/laya-client.mjs";

if (!(await bundleOnDisk())) {
  console.error(`bundle not present at ${MODEL_DIR}; run "node poc.mjs" once to download it`);
  process.exit(1);
}
console.log(`Verifying ${MODEL_DIR} (revision ${MODEL_REVISION})`);
const report = await verifyBundle(MODEL_DIR, { force: true, log: (m) => console.log(`  ${m}`) });
for (const r of report) {
  console.log(`  ${r.ok ? "OK  " : "FAIL"} ${r.file.padEnd(34)} ${String(r.size).padStart(12)} bytes${r.sha256 ? `  sha256=${r.sha256}` : ""}${r.hashMs ? `  (${r.hashMs} ms)` : ""}`);
}
