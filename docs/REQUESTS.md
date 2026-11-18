# Request and question schema

`ask.mjs --file` and `POST /decide` accept a JSON object containing the state to inspect and named questions.
The same question definitions work in the CLI's separate `--questions` files, the Node library, and custom
presets, for all three models.

## A complete request

This example is available as [`demo/data/ticket-request.json`](../demo/data/ticket-request.json):

```json
{
  "state": "I was charged twice for my order. Please refund the duplicate charge.",
  "questions": {
    "refund": {
      "type": "predicate",
      "instructions": "Does the customer request a refund?"
    },
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this ticket?",
      "choices": {
        "billing": "Payments, invoices, and refunds",
        "engineering": "Software bugs and outages",
        "support": "Other requests"
      }
    },
    "severity": {
      "type": "score",
      "instructions": "How severe is this ticket?",
      "levels": ["Low", "Medium", "High"]
    }
  }
}
```

`state` supplies the content verbatim, commonly a string or JSON object. Alternatively, `text` supplies a
string that the selected preset wraps in its state template. Supplying `questions` replaces the preset's
questions. Question IDs (`refund`, `team`, `severity`) become the keys in `answers`.

Optional top-level controls include `model`, `device`, `maxLen`, `acceleration`, `graphs`, `preset` and
`calibration`; Vega also supports `readout`. HTTP requests inherit omitted model controls from the server;
the CLI uses its environment/defaults after applying the file and explicit flags. See
[models and execution modes](MODELS.md) for their values and model-specific restrictions.

## Question fields and aliases

| Question | Accepted input | Meaning |
|---|---|---|
| Yes/no | `type: "predicate"` or `type: "noul"` | Probability that the instructions hold true |
| Choice | `type: "choice"`, with `choices` or `criteria` | Object mapping option keys to descriptions, or an array of option keys; at least two options |
| Score | `type: "score"`, with `levels` or `criteria` | Ordered array of at least two level labels; index 0 is the lowest |

Every question requires `instructions`. For predicates, optional `criteria` can describe `false` and `true`:
`{ "false": "No refund requested", "true": "A refund is requested" }`.

The adapter normalizes `predicate` to `noul`, `choices` to `criteria` for choice questions, and `levels` to
`criteria` for score questions before inference and calibration. Existing spellings remain supported, and
either style can appear in the same request. Caller-owned question objects are not modified.

If both `criteria` and its alias are supplied, their values must match; conflicting values return HTTP 400.
`choices` belongs to choice questions, and `levels` belongs to score questions. Invalid types, missing
instructions, and missing/undersized option sets also produce input errors.

## Run a complete JSON file from the CLI

With the `LocalLaya` PowerShell profile function that forwards arguments to `ask.mjs`:

```powershell
LocalLaya --model multilingual --sidecar --file .\ticket.json --json
LocalLaya --model multilingual --local --file .\ticket.json --json
```

Or invoke the project entry point directly:

```powershell
node ask.mjs --model multilingual --sidecar --file demo/data/ticket-request.json --json
```

The file contains one complete request, including `state` and `questions`, or `text` with a `preset`.
The adapter understands `predicate`, `choices` and `levels`. `--json` prints the result as JSON; omit it
for the normal human-readable answer table. File paths are resolved relative to your current working
directory. UTF-8 files with or without a BOM are supported.

Settings take precedence in this order: **explicit CLI flags > request file > environment > defaults**.
For example, `--model vega` overrides `"model": "multilingual"` in the file; `--questions other.json`
replaces its question set. The file may also supply `device`, `maxLen`, `readout`, `acceleration`, `graphs`,
`preset`, and the supported ONNX `lane`, `policy`, `deadlineMs` and `exec` controls.

`calibration` in a request is an inline `{ "temperature_by_options": { ... } }` object or `null` to select
shipped temperatures. The CLI's `--calibration <path>` takes precedence over that field.

Choose execution mode, port and sidecar lifetime through CLI flags. `--file` cannot be combined with
positional text, `--state`, `--text-file`, `--start`, `--stop` or `--status`. Malformed JSON, unsupported
fields and invalid question schemas fail before model startup. The skill wrapper also accepts `--file`.
The separate `demo/cli.mjs --file` command accepts its existing batch input format (an array or `{ items: [...] }`).

## Send a JSON file through HTTP

Start a sidecar, then send the complete file from PowerShell:

```powershell
node ask.mjs --start --model multilingual --idle 0 --max-age 0

$response = Invoke-RestMethod `
    -Uri "http://127.0.0.1:8787/decide" `
    -Method Post `
    -ContentType "application/json" `
    -InFile ".\demo\data\ticket-request.json"

$response.answers | ConvertTo-Json -Depth 20
```

You can add `"model": "laya"`, `"model": "vega"`, or `"model": "multilingual"` to the request to select a
model explicitly. An already-running sidecar needs a restart to load updated server code.

The CLI also accepts separate files: `state.json` contains just the JSON state value, and `questions.json`
contains just the questions object, using either field spelling:

```powershell
node ask.mjs --model multilingual --sidecar --state state.json --questions questions.json --json
node ask.mjs --model multilingual --local --state state.json --questions questions.json --json
```

## Read a complete request file in Node

```js
import { readFile } from "node:fs/promises";
import { createLaya } from "./demo/laya.mjs";

const { state, ...options } = JSON.parse(await readFile("ticket.json", "utf8"));
const d = await createLaya({ model: "multilingual", mode: "sidecar" });
try {
  const result = await d.decide({ state }, options);
  console.log(result.answers);
} finally {
  await d.close();
}
```

## Response fields

Answers use the canonical output schema:

- `answers.refund.type` is `"noul"`, and `answers.refund.noul` is the probability of true, from 0 to 1.
- `answers.team.choice` is an option key; `answers.team.probabilities` contains the option distribution.
- `answers.severity.score` is the expected level index, which can be fractional. For this example,
  0 = Low, 1 = Medium and 2 = High. `legend` maps indices to labels, and `probabilities` gives their distribution.
- The HTTP/library response's `questions` contains the normalized definitions (`noul` and `criteria`).
- `usage` and `routing` contain token counts, execution information and timings.

Run `npm test` for schema validation checks, and `npm run test:questions` for real-model HTTP, library,
CLI and file-input checks (requires the prepared models and CUDA).
