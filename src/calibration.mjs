/**
 * Calibration toolkit for Laya.
 *
 * Background. Laya's head produces one raw logit per option; the library turns them into probabilities
 * with softmax(logit / T) where T is a temperature read from laya_config.json, one per
 * (question type, option-count bucket): "choice:3-5", "noul:2", "score:3-5", ...  Temperature scaling
 * never changes the arg-max (accuracy) - it only flattens (T > 1) or sharpens (T < 1) the distribution,
 * i.e. it changes how *trustworthy* the probabilities are. The shipped temperatures were fitted on the
 * authors' data; on a new domain the model is usually over-confident, so refit T on a labelled sample.
 *
 * This module:
 *  - captures the raw logits of every systemOne call (by wrapping the ONNX session's run()),
 *  - computes accuracy / NLL / Brier / ECE and a reliability table,
 *  - fits one temperature per bucket by minimising NLL (golden-section search on log T),
 *  - evaluates the refit honestly with leave-one-out cross-validation,
 *  - applies a fitted table to a live Laya instance (mutates laya.config.temperature_by_options, which
 *    the library reads on every call).
 *
 * Relies on two internals of @receptron/laya 0.1.2: the `session` field and the mutable `config`
 * object. Both are plain runtime properties; pinned package version, see package.json.
 */

const QTYPE_INDEX = { choice: 0, score: 1, noul: 2 };

/** Same bucketing as the library's tempBucket(). */
export function bucketKey(type, k) {
  const size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
  return `${type}:${size}`;
}

/** Number of options / logits a question produces (noul is always [false, true]). */
export function optionCount(q) {
  if (q.type === "noul") return 2;
  if (q.type === "score") return q.criteria.length;
  return Array.isArray(q.criteria) ? q.criteria.length : Object.keys(q.criteria).length;
}

/** Option labels in logit order (matches the library's answer layout). */
export function optionLabels(q) {
  if (q.type === "noul") return ["false", "true"];
  if (q.type === "score") return q.criteria.map((_, i) => String(i));
  return Array.isArray(q.criteria) ? [...q.criteria] : Object.keys(q.criteria);
}

export function softmax(z) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/**
 * Wrap a Laya instance so that the raw logits of the most recent systemOne() are available.
 * Returns { last(): { logits: Float32Array, K: number, n: number } }.
 */
export function captureLogits(laya) {
  const session = laya.session;
  if (!session || typeof session.run !== "function") throw new Error("captureLogits: Laya instance has no accessible session (library internals changed?)");
  if (session.__layaCapture) return session.__layaCapture;
  const orig = session.run.bind(session);
  const cap = { lastOut: null, last: () => (cap.lastOut ? { logits: cap.lastOut.logits.data, n: cap.lastOut.logits.dims[0], K: cap.lastOut.logits.dims[1] } : null) };
  session.run = async (...a) => {
    const out = await orig(...a);
    cap.lastOut = out;
    return out;
  };
  session.__layaCapture = cap;
  return cap;
}

/**
 * Run systemOne and return, per question id, the raw logits (length k) next to the library's answer.
 */
export async function systemOneWithLogits(laya, capture, state, questions) {
  const result = await laya.systemOne(state, questions);
  const { logits, K } = capture.last();
  const raw = {};
  Object.keys(questions).forEach((qid, r) => {
    const k = optionCount(questions[qid]);
    raw[qid] = Array.from(logits.subarray(r * K, r * K + k));
  });
  return { result, raw };
}

/** Current temperature the library would use for a question. */
export function currentTemperature(laya, q) {
  const k = optionCount(q);
  return laya.config.temperature_by_options[bucketKey(q.type, k)] ?? laya.config.temperature[QTYPE_INDEX[q.type]] ?? 1;
}

// ---- metrics ------------------------------------------------------------------------------------------

/** items: [{ probs: number[], gold: number }] */
export function metrics(items, bins = 10) {
  let nll = 0;
  let brier = 0;
  let correct = 0;
  const binned = Array.from({ length: bins }, () => ({ n: 0, conf: 0, acc: 0 }));
  for (const { probs, gold } of items) {
    const p = Math.max(probs[gold] ?? 0, 1e-12);
    nll -= Math.log(p);
    brier += probs.reduce((s, v, i) => s + (v - (i === gold ? 1 : 0)) ** 2, 0);
    const pred = probs.indexOf(Math.max(...probs));
    const hit = pred === gold ? 1 : 0;
    correct += hit;
    const conf = probs[pred];
    const b = Math.min(bins - 1, Math.floor(conf * bins));
    binned[b].n++;
    binned[b].conf += conf;
    binned[b].acc += hit;
  }
  const n = items.length;
  let ece = 0;
  for (const b of binned) if (b.n) ece += (b.n / n) * Math.abs(b.acc / b.n - b.conf / b.n);
  const meanConf = items.reduce((s, it) => s + Math.max(...it.probs), 0) / n;
  return { n, accuracy: correct / n, nll: nll / n, brier: brier / n, ece, meanConfidence: meanConf, reliability: binned.map((b, i) => ({ bin: `${(i / bins).toFixed(1)}-${((i + 1) / bins).toFixed(1)}`, n: b.n, avgConf: b.n ? b.conf / b.n : null, accuracy: b.n ? b.acc / b.n : null })) };
}

// ---- temperature fitting --------------------------------------------------------------------------------

function nllAtT(samples, T) {
  let s = 0;
  for (const { logits, gold } of samples) {
    const p = softmax(logits.map((v) => v / T));
    s -= Math.log(Math.max(p[gold], 1e-12));
  }
  return s / samples.length;
}

/** Minimise mean NLL over T in [lo, hi] (golden-section on log T). */
export function fitTemperature(samples, { lo = 0.05, hi = 30 } = {}) {
  let a = Math.log(lo);
  let b = Math.log(hi);
  const gr = (Math.sqrt(5) - 1) / 2;
  let c = b - gr * (b - a);
  let d = a + gr * (b - a);
  let fc = nllAtT(samples, Math.exp(c));
  let fd = nllAtT(samples, Math.exp(d));
  for (let i = 0; i < 80 && Math.abs(b - a) > 1e-4; i++) {
    if (fc < fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - gr * (b - a);
      fc = nllAtT(samples, Math.exp(c));
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + gr * (b - a);
      fd = nllAtT(samples, Math.exp(d));
    }
  }
  const T = Math.exp((a + b) / 2);
  return { T, nll: nllAtT(samples, T) };
}

/**
 * Leave-one-out: for each sample, fit T on the others and score this one. Returns items usable with
 * metrics() - an honest estimate of how the refit generalises within the domain.
 */
export function leaveOneOut(samples) {
  return samples.map((s, i) => {
    const rest = samples.filter((_, j) => j !== i);
    const { T } = fitTemperature(rest);
    return { probs: softmax(s.logits.map((v) => v / T)), gold: s.gold, T };
  });
}

/** Apply a calibration table ({ temperature_by_options: {...} }) to a live instance. */
export function applyCalibration(laya, table) {
  const t = table?.temperature_by_options;
  if (!t || typeof t !== "object") throw new Error("applyCalibration: table.temperature_by_options missing");
  const before = { ...laya.config.temperature_by_options };
  Object.assign(laya.config.temperature_by_options, t);
  return { before, after: { ...laya.config.temperature_by_options } };
}
