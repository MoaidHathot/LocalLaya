/**
 * Human-readable rendering of a systemOne result for terminals.
 */
const pct = (p) => `${Math.round(p * 100)}%`;
const bar = (p, width = 10) => {
  const full = Math.round(p * width);
  return "\u2588".repeat(full) + "\u2591".repeat(width - full);
};
const c = {
  dim: (s) => `\x1b[2m${s}\x1b[0m`,
  bold: (s) => `\x1b[1m${s}\x1b[0m`,
  green: (s) => `\x1b[32m${s}\x1b[0m`,
  yellow: (s) => `\x1b[33m${s}\x1b[0m`,
  red: (s) => `\x1b[31m${s}\x1b[0m`,
  cyan: (s) => `\x1b[36m${s}\x1b[0m`,
};
const colourByP = (p) => (p >= 0.8 ? c.green : p >= 0.55 ? c.yellow : c.red);

/** One line per answer. `questions` supplies score legends / option order. */
export function formatAnswers(answers, questions, { color = true } = {}) {
  const paint = (fn, s) => (color ? fn(s) : s);
  const width = Math.max(...Object.keys(answers).map((k) => k.length));
  const lines = [];
  for (const [qid, a] of Object.entries(answers)) {
    const name = qid.padEnd(width);
    if (a.type === "choice") {
      const p = a.probabilities[a.choice];
      const others = Object.entries(a.probabilities)
        .filter(([k]) => k !== a.choice)
        .sort((x, y) => y[1] - x[1])
        .map(([k, v]) => `${k} ${pct(v)}`)
        .join(", ");
      lines.push(`  ${name}  ${paint(colourByP(p), `${a.choice.padEnd(16)} ${pct(p).padStart(4)}`)}  ${bar(p)}  ${paint(c.dim, others)}`);
    } else if (a.type === "noul") {
      const yes = a.noul >= 0.5;
      const p = yes ? a.noul : 1 - a.noul;
      lines.push(`  ${name}  ${paint(colourByP(p), `${(yes ? "yes" : "no").padEnd(16)} ${pct(p).padStart(4)}`)}  ${bar(a.noul)}  ${paint(c.dim, `P(true) = ${a.noul.toFixed(3)}`)}`);
    } else if (a.type === "score") {
      const levels = Object.values(a.legend);
      const max = levels.length - 1;
      const nearest = levels[Math.min(max, Math.max(0, Math.round(a.score)))];
      const dist = Object.entries(a.probabilities).map(([i, v]) => `${levels[Number(i)]} ${pct(v)}`).join(", ");
      lines.push(`  ${name}  ${paint(c.cyan, `${nearest.slice(0, 16).padEnd(16)} ${a.score.toFixed(2)}/${max}`)}  ${bar(a.score / max)}  ${paint(c.dim, dist)}`);
    }
  }
  return lines.join("\n");
}

/** Header line: state summary + routing. */
export function formatHeader(stateText, result, { color = true, via = "" } = {}) {
  const paint = (fn, s) => (color ? fn(s) : s);
  const r = result.routing;
  const where = r ? `${r.lane} ${r.ms.toFixed(0)} ms${r.queueMs > 5 ? ` (+${r.queueMs.toFixed(0)} queued)` : ""}${via ? ` via ${via}` : ""}` : "";
  const shown = stateText.length > 90 ? `${stateText.slice(0, 87)}...` : stateText;
  return `${paint(c.bold, JSON.stringify(shown))}  ${paint(c.dim, `${where}  ${result.usage.input_tokens} tokens${r?.explored ? "  (exploration)" : ""}`)}`;
}

export const colors = c;
