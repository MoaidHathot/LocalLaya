"""Vega kernel A/B and profiling, using one loaded model and identical inputs.

    .venv-laya/Scripts/python.exe experiments/vega_speed.py --variants reference --profile
    .venv-laya/Scripts/python.exe experiments/vega_speed.py --variants reference,fla --words 0,1024,4096 --rounds 6
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
os.environ.update(TRITON_CACHE_DIR=str(ROOT / ".laya/triton"),
                  TORCHINDUCTOR_CACHE_DIR=str(ROOT / ".laya/torchinductor"),
                  TRITON_DEFAULT_FP_FUSION="1",
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0")
from laya_worker import MANIFEST, prepared


def kernel_modes():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    names = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update")
    reference = {name: inspect.unwrap(getattr(qwen, name)) for name in names}

    def select(mode):
        for name, function in reference.items():
            setattr(qwen, name, function)
        if mode == "reference":
            return
        if mode not in ("fla", "fla-conv", "fla-graph", "optimized", "streamed", "streamed-flash"):
            raise ValueError(f"unknown variant {mode}")
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

        for name, fn in (("torch_chunk_gated_delta_rule", chunk_gated_delta_rule),
                         ("torch_recurrent_gated_delta_rule", fused_recurrent_gated_delta_rule)):
            params = set(inspect.signature(fn).parameters)
            def wrapped(*args, _fn=fn, _params=params, **kw):
                return _fn(*args, **{k: v for k, v in kw.items() if k in _params})
            setattr(qwen, name, wrapped)
        if mode in ("fla-conv", "optimized", "streamed", "streamed-flash"):
            from fla.modules.causal_conv1d import causal_conv1d

            def convolution(x, weight, bias=None, activation=None, **kw):
                out, _ = causal_conv1d(x.transpose(1, 2), weight=weight, bias=bias, activation=activation, backend="triton")
                return out.transpose(1, 2)
            qwen.causal_conv1d_fn = convolution
    return select


QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team handles this request?", "criteria": {
        "billing": "invoices, payments and refunds", "technical": "bugs and outages", "sales": "pricing and plans"}},
    "refund": {"type": "noul", "instructions": "Does the customer request a refund?"},
    "urgency": {"type": "score", "instructions": "How urgent is this request?", "criteria": ["can wait", "soon", "today"]},
}


def differences(a, b):
    max_dp, flips, flags = 0, [], []
    for qid, ans in a.items():
        other = b[qid]
        for key, p in ans["probabilities"].items():
            max_dp = max(max_dp, abs(p - other["probabilities"][key]))
        if max(ans["probabilities"], key=ans["probabilities"].get) != max(other["probabilities"], key=other["probabilities"].get):
            flips.append(qid)
        for key in ("prediction_set", "route", "abstain", "unbound"):
            if ans.get(key) != other.get(key):
                flags.append(f"{qid}.{key}")
    return {"maxProbabilityDelta": max_dp, "argmaxFlips": flips, "flagChanges": flags}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", default="reference,fla")
    p.add_argument("--words", default="0,1024,4096")
    p.add_argument("--questions", type=int, default=3)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--out", default="results/vega-speed.json")
    args = p.parse_args()
    from vega_fast import pin_process
    affinity = pin_process()
    import torch
    import laya

    torch.set_num_threads(1)
    select = kernel_modes()
    spec = MANIFEST["models"]["vega"]
    print("Loading Vega for interleaved kernel comparison", flush=True)
    agent = laya.Agent(vegaml=prepared(spec), backbone=prepared(spec["backbone"]), device="cuda", max_len=73728)
    from vega_graph import HeadReplay
    from vega_stream import streamed_prefill
    graphs = HeadReplay(agent.vega.engine.engine)
    streaming = streamed_prefill(agent.vega.engine.perc, min_tokens=0)
    select_kernel = select
    def select(mode):
        select_kernel(mode)
        graphs.enabled = mode in ("fla-graph", "optimized", "streamed", "streamed-flash")
        streaming["enabled"] = mode in ("streamed", "streamed-flash")
        streaming["symbolicMasks"] = mode == "streamed-flash"
    variants = args.variants.split(",")
    source = list(QUESTIONS.values())
    qs = {f"q{i}": source[i % len(source)] for i in range(args.questions)}
    report = {"versions": {"torch": torch.__version__, "cuda": torch.version.cuda}, "affinity": affinity, "variants": variants, "questions": qs, "cases": []}
    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)

    def save():
        destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    for words in map(int, args.words.split(",")):
        state = "background " * words
        state += "I was charged twice. Please refund the duplicate today."
        times = {v: [] for v in variants}
        outputs, peaks, cold = {}, {}, {}
        for v in variants:
            select(v)
            print(f"Warm-up {v}, {words} filler words", flush=True)
            t0 = time.perf_counter()
            agent.predict(state, qs)
            torch.cuda.synchronize()
            cold[v] = (time.perf_counter() - t0) * 1000
        for round_id in range(args.rounds):
            for v in variants[::1 if round_id % 2 == 0 else -1]:
                select(v)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                outputs[v] = agent.predict(state, qs)
                torch.cuda.synchronize()
                times[v].append((time.perf_counter() - t0) * 1000)
                peaks[v] = torch.cuda.max_memory_allocated() / 1024**2
        row = {"words": words, "timesMs": times, "medianMs": {v: statistics.median(t) for v, t in times.items()},
               "coldMs": cold, "peakMiB": peaks, "answers": {v: r["answers"] for v, r in outputs.items()},
               "comparison": {v: differences(outputs[variants[0]]["answers"], r["answers"]) for v, r in outputs.items()}}
        report["cases"].append(row)
        report["headGraphs"] = graphs.stats()
        print(json.dumps({k: row[k] for k in ("words", "medianMs", "peakMiB", "comparison")}), flush=True)
        save()
        if args.profile:
            for v in variants:
                select(v)
                perc = agent.vega.engine.perc
                head = agent.vega.engine.engine
                original_pool, original_head = perc.pooled, head.forward
                def pool(*a, **kw):
                    with torch.profiler.record_function("vega_reader"):
                        return original_pool(*a, **kw)
                def forward(*a, **kw):
                    with torch.profiler.record_function("vega_engine"):
                        return original_head(*a, **kw)
                perc.pooled, head.forward = pool, forward
                try:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
                        agent.predict(state, qs)
                        torch.cuda.synchronize()
                    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=35)
                    print(table, flush=True)
                    (ROOT / f"results/vega-profile-{v}-{words}.txt").write_text(table, encoding="utf-8", newline="\n")
                    row.setdefault("profile", {})[v] = [{"name": e.key, "count": e.count, "cpuUs": e.cpu_time_total,
                        "selfCpuUs": e.self_cpu_time_total, "deviceUs": e.device_time_total} for e in prof.key_averages()]
                    save()
                finally:
                    perc.pooled, head.forward = original_pool, original_head


if __name__ == "__main__":
    main()
