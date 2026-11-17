"""Compare all typed outputs/decision flags across reference, fused, and graph Vega paths."""
import json
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from laya_worker import MANIFEST, prepared
from vega_fast import reference_kernels, fused_kernels
from vega_graph import HeadReplay
from vega_stream import streamed_prefill
from vega_speed import QUESTIONS, differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-only", action="store_true", help="varied 33K/50K-token documents, including all typed questions")
    args = parser.parse_args()
    import torch
    import laya

    torch.set_num_threads(1)
    reference_kernels()
    spec = MANIFEST["models"]["vega"]
    agent = laya.Agent(vegaml=prepared(spec), backbone=prepared(spec["backbone"]), device="cuda", max_len=73728)
    graph = HeadReplay(agent.vega.engine.engine)
    stream = streamed_prefill(agent.vega.engine.perc, symbolic_masks=True)
    rows = json.loads((ROOT / "presets/dev-request.eval.json").read_text(encoding="utf-8"))["items"]
    dev = json.loads((ROOT / "presets/dev-request.json").read_text(encoding="utf-8"))["questions"]
    cases = [(r["text"], dev) for r in rows]
    for text in ["I was charged twice. Please refund the duplicate today.", "The app crashes whenever I open settings.",
                 "I do not want a refund. Please tell me the price of an upgrade.", "تم خصم المبلغ مرتين من حسابي.",
                 "".join(["background "] * 180) + "Please reset my password."]:
        for qs in [QUESTIONS, *({k: q} for k, q in QUESTIONS.items())]:
            cases.append((text, qs))
    for n in (127, 128, 4095, 4096, 7800):
        cases.append(("background " * n + "I was charged twice. Please refund me.", QUESTIONS))
    if args.long_only:
        tok = agent.vega.engine.perc.tok
        corpus = "\n".join(f"Record {i}: {rows[i % len(rows)]['text']}" for i in range(5000))
        tokens = tok(corpus, add_special_tokens=False)["input_ids"]
        cases = [(tok.decode(tokens[:n]) + "\nFinal request: I was charged twice. Please refund the duplicate today.", QUESTIONS) for n in (33000, 50000)]
    report = {"cases": [], "questions": 0, "kernelArgmaxFlips": 0, "kernelFlagChanges": 0,
              "maxKernelProbabilityDelta": 0, "maxGraphProbabilityDelta": 0, "maxStreamProbabilityDelta": 0, "streamFlagChanges": 0}
    for i, (state, qs) in enumerate(cases):
        reference_kernels(); graph.enabled = False; stream["enabled"] = False
        ref = agent.predict(state, qs)["answers"]
        fused_kernels(); graph.enabled = False
        fused = agent.predict(state, qs)["answers"]
        graph.enabled = True
        graphed = agent.predict(state, qs)["answers"]
        stream["enabled"] = True
        fast = agent.predict(state, qs)["answers"]
        kernel_diff, graph_diff, stream_diff = differences(ref, fast), differences(fused, graphed), differences(graphed, fast)
        if graph_diff["argmaxFlips"] or graph_diff["flagChanges"] or graph_diff["maxProbabilityDelta"] > 1e-5:
            raise AssertionError(f"graph changed an answer/flag: case {i}: {graph_diff}")
        if stream_diff["argmaxFlips"] or stream_diff["maxProbabilityDelta"] > 0.03:
            raise AssertionError(f"streamed prefill mismatch: case {i}: {stream_diff}")
        report["questions"] += len(qs)
        report["kernelArgmaxFlips"] += len(kernel_diff["argmaxFlips"])
        report["kernelFlagChanges"] += len(kernel_diff["flagChanges"])
        report["maxKernelProbabilityDelta"] = max(report["maxKernelProbabilityDelta"], kernel_diff["maxProbabilityDelta"])
        report["maxGraphProbabilityDelta"] = max(report["maxGraphProbabilityDelta"], graph_diff["maxProbabilityDelta"])
        report["maxStreamProbabilityDelta"] = max(report["maxStreamProbabilityDelta"], stream_diff["maxProbabilityDelta"])
        report["streamFlagChanges"] += len(stream_diff["flagChanges"])
        report["cases"].append({"index": i, "characters": len(state), "reference": ref, "optimized": fast,
                                "kernelComparison": kernel_diff, "graphComparison": graph_diff, "streamComparison": stream_diff})
        print(f"{i + 1}/{len(cases)}: dp={kernel_diff['maxProbabilityDelta']:.6f}, flips={kernel_diff['argmaxFlips']}, graph dp={graph_diff['maxProbabilityDelta']:.6f}", flush=True)
    report["graphs"] = graph.stats()
    report["streaming"] = stream
    if report["graphs"]["failed"] or not report["graphs"]["hits"]:
        raise AssertionError(report["graphs"])
    output = "vega-long-fidelity-2026-11-17.json" if args.long_only else "vega-fast-fidelity-2026-11-17.json"
    (ROOT / "results" / output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
