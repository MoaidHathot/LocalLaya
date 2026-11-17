"""Numerical/decision comparison over varied languages, question types and lengths."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from multilingual_speed import load_agent, QUESTIONS, differences, align_precision
from multilingual_fast import GraphCache


def main():
    from laya.fast import FastLaya

    agent = load_agent()
    stock = agent.model.forward
    fast = FastLaya(agent.model, max_len=8192, dtype=agent.dtype)
    fast.graphs = GraphCache()
    aligned = align_precision(copy.copy(fast))
    # Share immutable resident weights and compiled kernels; graph/input buffers are separate.
    variants = {"eager": None, "tilelang": fast, "aligned": aligned}
    rows = json.loads((ROOT / "presets/dev-request.eval.json").read_text(encoding="utf-8"))["items"]
    dev = json.loads((ROOT / "presets/dev-request.json").read_text(encoding="utf-8"))["questions"]
    cases = [(r["text"], dev) for r in rows]
    for text in ["I was charged twice. Please refund the duplicate today.", "The app crashes whenever I open settings.",
                 "I do not want a refund. Please explain the upgrade price.",
                 "تم خصم المبلغ مرتين من حسابي. أرجو إعادة المبلغ الزائد اليوم.",
                 "Me cobraron dos veces. Por favor, devuelvan el cargo duplicado.",
                 "J'ai été facturé deux fois. Veuillez rembourser le doublon.",
                 "我被扣了两次款，请退还重复扣款。", "अभी भुगतान दो बार कट गया, कृपया अतिरिक्त राशि वापस करें।"]:
        for qs in [QUESTIONS, *({k: q} for k, q in QUESTIONS.items())]:
            cases.append((text, qs))
    for n in (63, 64, 127, 128, 255, 256, 511, 512, 1024, 4096, 7800):
        cases.append(("background " * n + "Please refund the duplicate charge.", QUESTIONS if n <= 1024 else {"team": QUESTIONS["team"]}))
    report = {"questions": 0, "cases": [], "summary": {v: {"maxProbabilityDelta": 0, "argmaxFlips": 0} for v in variants}}
    for i, (state, qs) in enumerate(cases):
        outputs = {}
        for name, backend in variants.items():
            agent._fast = backend
            agent.model.forward = stock if backend is None else backend.forward
            outputs[name] = agent.predict(state, qs)["answers"]
        comparisons = {v: differences(outputs["eager"], output) for v, output in outputs.items()}
        report["questions"] += len(qs)
        report["cases"].append({"index": i, "characters": len(state), "answers": outputs, "comparison": comparisons})
        for v, delta in comparisons.items():
            report["summary"][v]["maxProbabilityDelta"] = max(report["summary"][v]["maxProbabilityDelta"], delta["maxProbabilityDelta"])
            report["summary"][v]["argmaxFlips"] += len(delta["argmaxFlips"])
        print(f"{i + 1}/{len(cases)}: native dp={comparisons['tilelang']['maxProbabilityDelta']:.4f}, aligned dp={comparisons['aligned']['maxProbabilityDelta']:.4f}, flips={comparisons['aligned']['argmaxFlips']}", flush=True)
        (ROOT / "results/multilingual-fidelity-2026-11-17.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
