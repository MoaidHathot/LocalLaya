# Laya decisions - presets, questions, calibration

## Built-in presets

| preset | state wrapper | questions | measured? |
|---|---|---|---|
| `smart-home` | `{ application, userMessage: $TEXT, time }` | intent (choice 4), should_execute (noul), target_device (choice 5), desired_state (choice 4), urgency (score 4) | yes: intent 0.72, should_execute 0.66, target_device 0.77 (65 items) |
| `dev-request` | `{ request: $TEXT }` | task (choice 7), language (choice 8), effort (score 4) | yes: task 0.75, language 0.85 (40 items) |
| `triage` | `{ message: $TEXT }` | department (choice 5), urgency (score 4), frustration (score 4), churn_risk, refund_requested (noul) | no |
| `guard` | `{ prompt: $TEXT }` | jailbreak, injection, secret_extraction (noul), risk (score 4) | no |
| `moderation` | `{ post: $TEXT }` | category (choice 6), toxicity (score 4), harassment, threat (noul) | no |
| `route` | `{ request: $TEXT }` | complexity (score 5), needs_tools, needs_reasoning (noul), best_model (choice 3) | no |
| `sentiment` | `{ text: $TEXT }` | sentiment (choice 3), emotion (choice 6), sarcasm (noul) | no |

Unmeasured presets are reasonable question sets, not validated classifiers. Measure before relying on them.

## Writing a preset: `presets/<name>.json`

```json
{
  "description": "one line: what this preset decides",
  "state": { "request": "$TEXT", "app": "my tool" },
  "questions": {
    "task": {
      "type": "choice",
      "instructions": "What is the user asking the assistant to do?",
      "criteria": {
        "validate": "check whether a given input is valid",
        "write_code": "write or generate new code",
        "explain": "explain a concept, API or error"
      }
    },
    "needs_tool": { "type": "noul", "instructions": "Must a program run to do this properly?",
                    "criteria": { "true": "yes, something must execute", "false": "no, an answer from knowledge is enough" } },
    "effort": { "type": "score", "instructions": "How much work is this?", "criteria": ["trivial", "small", "medium", "large"] }
  }
}
```

- `state` is a template: `"$TEXT"` is replaced by the message; other fields are literal context (time, app,
  user role, ...). `"state": "request"` is shorthand for `{ "request": "$TEXT" }`; omitted = `{ "text": "$TEXT" }`.
- `choice.criteria`: `{ key: "short description" }` (or an array of keys). `score.criteria`: ordered array, index 0
  = lowest. `noul.criteria` (optional): `{ true: "...", false: "..." }` descriptions.
- Question ids become the keys of `answers`.

Rules of thumb for options (they are model input, scored at their own `[MASK]` token):

- Short, concrete, mutually exclusive descriptions; name the vocabulary you expect ("TV, speakers, music,
  volume, playback").
- Fewer than ~20 options; all options of one question share a 192-token budget.
- Wording effects are non-monotonic: a richer description can attract unrelated texts (measured: a richer
  "lights" description dropped target_device accuracy 0.77 -> 0.66). Change one question at a time and re-measure.
- Do not ask the model what code can compute (does the text contain braces? a version number?).
- Questions that answer at chance level poison their bucket's temperature (temperatures are shared per
  type x option-count); remove them.

You can also build questions interactively in the REPL (`node ask.mjs`): `/choice`, `/noul`, `/score`,
`/set key=value`, then `/save <name>` writes `presets/<name>.json`.

## Measuring and calibrating: `presets/<name>.eval.json`

```json
{ "items": [
  { "text": "is this valid json {bla: 1}", "gold": { "task": "validate", "language": "json" } },
  { "state": { "request": "..." }, "gold": { "task": "explain" } }
] }
```

Gold values: choice -> option key; noul -> `true`/`false`; score -> level index or level text. Questions with
fewer than 10 labelled items are reported but not fitted.

```powershell
node calibrate.mjs --preset <name> --eval presets/<name>.eval.json
```

Prints, per question: accuracy vs the majority-class and chance baselines with a verdict (`usable` /
`weak` / `NOT USABLE zero-shot`), NLL / Brier / ECE for raw, shipped and refit temperatures (leave-one-out),
a reliability table (stated confidence vs actual accuracy per bin), the most confident mistakes and the
confusion pairs. Writes `calibration/<name>.json`, which `ask.mjs` and `serve.mjs` apply automatically for
that preset from then on.

What calibration does and does not do: it rescales probabilities (`softmax(logits / T)`, one `T` per
question type x option-count bucket) so that "80 %" means right about 80 % of the time. It never changes
which option wins. Systematic confusions (e.g. device *questions* classified as device *commands*) need
better wording or fine-tuning, not temperature.

## Iteration loop

1. Write the preset; try a dozen inputs with `node ask.mjs --preset <name> "..."` (the sidecar makes this cheap).
2. Label 30-60 real inputs; run `calibrate.mjs`; read the verdicts and the confident mistakes.
3. Reword or drop failing questions (one at a time), re-run.
4. If a question stays weak and matters: replace it with code where possible, or fine-tune (PyTorch, the
   original repo's notebook) and re-export the ONNX bundle.
