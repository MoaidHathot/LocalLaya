/** Read one complete /decide-style request from a caller-relative UTF-8 JSON file. */
import { readFile } from "node:fs/promises";

const FIELDS = new Set([
  "state", "text", "questions", "preset", "model", "device", "maxLen", "readout", "acceleration", "graphs",
  "lane", "policy", "deadlineMs", "calibration", "exec",
]);

export async function readRequestFile(file) {
  const bad = (message) => Object.assign(new Error(`--file ${file}: ${message}`), { status: 400, code: "BAD_REQUEST" });
  let contents;
  try { contents = await readFile(file, "utf8"); }
  catch (e) { throw bad(e.message); }
  let body;
  // Windows editors / PowerShell may write a UTF-8 BOM.
  try { body = JSON.parse(contents.replace(/^\uFEFF/, "")); }
  catch (e) { throw bad(`invalid JSON: ${e.message}`); }
  if (!body || typeof body !== "object" || Array.isArray(body)) throw bad("expected one JSON request object with state or text");
  for (const key of Object.keys(body)) if (!FIELDS.has(key)) throw bad(`unsupported request field ${JSON.stringify(key)}; use CLI flags for execution mode and sidecar settings`);
  if (!Object.hasOwn(body, "state") && !(typeof body.text === "string" && body.text.trim())) throw bad("provide state or non-empty text");
  return body;
}
