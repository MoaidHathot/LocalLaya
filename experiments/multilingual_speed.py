"""Interleaved Multilingual Laya A/B and optional operator profiling, with one shared model.

    .venv-laya/Scripts/python.exe experiments/multilingual_speed.py --variants eager --profile
    .venv-laya/Scripts/python.exe experiments/multilingual_speed.py --variants eager,tilelang --words 0,1024,4096,7800
"""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
os.environ.update(TILELANG_CACHE_DIR=str(ROOT / ".laya/tilelang"),
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0")
from laya_worker import MANIFEST, prepared, multilingual_runtime, versions
from vega_fast import pin_process
from multilingual_fast import GraphCache

QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team handles this request?", "criteria": {
        "billing": "invoices, payments and refunds", "technical": "bugs and outages", "sales": "pricing and plans"}},
    "refund": {"type": "noul", "instructions": "Does the customer request a refund?"},
    "urgency": {"type": "score", "instructions": "How urgent is this request?", "criteria": ["can wait", "soon", "today"]},
}


def probabilities(answer):
    return answer.get("probabilities", {"false": 1 - answer.get("noul", 0), "true": answer.get("noul", 0)})


def differences(reference, actual):
    delta, score_delta, flips = 0, 0, []
    for qid, ans in reference.items():
        a, b = probabilities(ans), probabilities(actual[qid])
        delta = max(delta, *(abs(p - b[k]) for k, p in a.items()))
        before = ans.get("choice", max(a, key=a.get))
        after = actual[qid].get("choice", max(b, key=b.get))
        if before != after:
            flips.append(qid)
        if ans["type"] == "score":
            score_delta = max(score_delta, abs(ans["score"] - actual[qid]["score"]))
    return {"maxProbabilityDelta": delta, "maxScoreDelta": score_delta, "argmaxFlips": flips}


def load_agent(device="cuda"):
    import torch
    import laya

    pin_process()
    torch.set_num_threads(8)
    spec = MANIFEST["models"]["multilingual"]
    agent = laya.Agent(multilingual_runtime(spec, prepared(spec)), device=device, backend="eager")
    agent.cfg["max_len"] = 8192
    return agent


def align_precision(fast):
    """Experimental variant only: it did not beat the stock fast path's maximum probability error."""
    import torch
    encoder = fast.m.encoder
    pos = torch.arange(fast.max_len, device=fast.dev).float()
    fast.rope = {}
    for kind in set(encoder.config.layer_types):
        frequency = pos[:, None] * getattr(encoder.rotary_emb, f"{kind}_inv_freq").float()[None, :]
        scale = getattr(encoder.rotary_emb, f"{kind}_attention_scaling")
        fast.rope[kind] = ((frequency.cos() * scale).contiguous(), (frequency.sin() * scale).contiguous())
    fast.type_emb = fast.m.type_emb.weight.detach().float().contiguous()
    fast._rope_tab = {}
    fast.graphs = GraphCache()
    return fast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="eager,tilelang")
    parser.add_argument("--words", default="0,1024,4096,7800")
    parser.add_argument("--questions", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--out", default="results/multilingual-speed.json")
    args = parser.parse_args()
    import torch

    agent = load_agent()
    variants = args.variants.split(",")
    stock = agent.model.forward
    fast = None
    aligned = None
    if any(v.startswith("tilelang") for v in variants):
        from laya.fast import FastLaya
        fast = FastLaya(agent.model, max_len=8192, use_graphs=True, verbose=True, dtype=agent.dtype)
        fast.graphs = GraphCache()
        if "tilelang-aligned" in variants:
            aligned = align_precision(copy.copy(fast))
    for v in variants:
        if v not in ("eager", "tilelang", "tilelang-aligned"):
            raise ValueError(f"unknown variant {v}")

    def select(v):
        selected = aligned if v == "tilelang-aligned" else fast
        agent.model.forward = stock if v == "eager" else selected.forward
        agent._fast = None if v == "eager" else selected

    qs = {f"q{i}": list(QUESTIONS.values())[i % 3] for i in range(args.questions)}
    report = {"versions": versions(), "dtype": str(agent.dtype), "questions": qs, "cases": []}
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    def save():
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    for n in map(int, args.words.split(",")):
        state = "background " * n + "I was charged twice. Please refund the duplicate today."
        times, outputs, cold, peaks = {v: [] for v in variants}, {}, {}, {}
        for v in variants:
            select(v)
            print(f"Warm-up {v}: {n} filler words", flush=True)
            t0 = time.perf_counter()
            agent.predict(state, qs)
            torch.cuda.synchronize()
            cold[v] = (time.perf_counter() - t0) * 1000
        for r in range(args.rounds):
            for v in variants[::1 if r % 2 == 0 else -1]:
                select(v)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                outputs[v] = agent.predict(state, qs)
                torch.cuda.synchronize()
                times[v].append((time.perf_counter() - t0) * 1000)
                peaks[v] = torch.cuda.max_memory_allocated() / 1024**2
        row = {"words": n, "timesMs": times, "medianMs": {v: statistics.median(t) for v, t in times.items()},
               "coldMs": cold, "peakMiB": peaks, "usage": {v: o["usage"] for v, o in outputs.items()},
               "answers": {v: o["answers"] for v, o in outputs.items()},
               "comparison": {v: differences(outputs[variants[0]]["answers"], o["answers"]) for v, o in outputs.items()}}
        report["cases"].append(row)
        print(json.dumps({k: row[k] for k in ("words", "medianMs", "peakMiB", "comparison")}), flush=True)
        save()
        if args.profile:
            for v in variants:
                select(v)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
                    agent.predict(state, qs)
                    torch.cuda.synchronize()
                table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=35)
                print(table, flush=True)
                (ROOT / f"results/multilingual-profile-{v}-{n}.txt").write_text(table, encoding="utf-8", newline="\n")
                row.setdefault("profile", {})[v] = [{"name": e.key, "count": e.count, "cpuUs": e.cpu_time_total,
                    "selfCpuUs": e.self_cpu_time_total, "deviceUs": e.device_time_total} for e in prof.key_averages()]
                save()


if __name__ == "__main__":
    main()
