"""One offline Laya-Python model per process; newline-delimited JSON on stdin/stdout.

Model/runtime diagnostics go to stderr. Exiting this process releases its PyTorch/CUDA memory.
The legacy ONNX worker and its environment are independent of this worker.
"""
import argparse
import contextlib
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from vega_fast import configure_reader, pin_process
from multilingual_fast import MultilingualAcceleration
from runtime_progress import StartupProgress

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(HF_HOME=str(ROOT / "models/.huggingface"), HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", USE_TF="0")
MANIFEST = json.loads((ROOT / "data/python-models.json").read_text(encoding="utf-8"))


def emit(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    sys.stdout.flush()


def prepared(spec):
    directory = ROOT / spec["directory"]
    marker = directory / "local-laya-verified.json"
    try:
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved["revision"] != spec["revision"] or saved["repo"] != spec["repo"]:
            raise ValueError("revision mismatch")
        for name in spec["files"]:
            if (directory / name).stat().st_size != saved["files"][name]["size"]:
                raise ValueError(f"size mismatch: {name}")
    except (OSError, KeyError, ValueError) as exc:
        raise FileNotFoundError(f"Model is not prepared at {directory}: {exc}. Run npm run models:download -- --model <vega|multilingual>.") from exc
    return str(directory)


def versions():
    return {name: importlib.metadata.version(name) for name in ("laya", "vegaml", "torch", "transformers", "huggingface_hub")}


def load_auto_device(factory, requested, torch):
    """Auto may retry a GPU allocation failure on CPU; explicit CUDA and model errors stay errors."""
    cuda = torch.cuda.is_available()
    if requested == "cuda" and not cuda:
        raise RuntimeError("CUDA was requested but no working CUDA device is available; use --device cpu or rerun python:setup")
    selected = ("cuda" if cuda else "cpu") if requested == "auto" else requested
    reason = ("CUDA available" if selected == "cuda" else "CUDA unavailable; selected CPU") if requested == "auto" else f"explicit {selected}"
    try:
        agent = factory(selected)
    except torch.cuda.OutOfMemoryError as exc:
        if requested != "auto" or selected != "cuda":
            raise
        reason = f"CUDA load ran out of memory; selected CPU: {exc}"
        exc.__traceback__ = None
        gc.collect()
        torch.cuda.empty_cache()
        selected = "cpu"
        agent = factory(selected)
    actual = str(agent.device).split(":")[0]
    if actual != selected:
        if requested != "auto":
            raise RuntimeError(f"requested {selected} but Laya loaded on {actual}")
        reason = f"Laya loaded on {actual} instead of {selected}"
    return agent, actual, reason


def multilingual_runtime(spec, model_dir):
    """Laya rewrites mmBERT's tokenizer config on load. Give it a derived local view.

    Weights/tokenizer data are hard-linked where supported; small configs are copied.
    The pinned download and its digest manifest remain intact across repeated loads.
    """
    source = Path(model_dir)
    target = source / f"runtime-laya-{MANIFEST['runtime']['laya']}"
    for name in spec["files"]:
        src, dst = source / name, target / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".json") and name != "tokenizer/tokenizer.json":
            data = json.loads(src.read_text(encoding="utf-8"))
            if name == "tokenizer/tokenizer_config.json" and isinstance(data.get("extra_special_tokens"), list):
                data["extra_special_tokens"] = {f"extra_{i}": token for i, token in enumerate(data["extra_special_tokens"])}
            # Atomic replace: concurrent sidecars may prepare the same derived view.
            temporary = dst.with_name(f"{dst.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
            temporary.replace(dst)
        elif not dst.exists():
            try:
                os.link(src, dst)
            except FileExistsError:
                pass
            except OSError:
                temporary = dst.with_name(f"{dst.name}.{os.getpid()}.tmp")
                shutil.copyfile(src, temporary)
                temporary.replace(dst)
    return str(target)


def vega_usage(agent, state, questions):
    """Use the pinned runtime's actual prompt spans, not summed tokens across question rows.

    laya 0.4.35 predict_long infers truncation from sum(input_tokens) >= max_len, which
    misreports multi-question calls. build_prompt also truncates within 0.16.3. Report
    exact dropped state tokens per question, using the very same prompt builder.
    """
    from laya.vega import _to_vega_questions
    from vegaml._common import build_prompt, serialize_state

    engine = agent.vega.engine
    text = serialize_state(state)
    state_ids = engine.perc.tok(text, add_special_tokens=False)["input_ids"]
    dropped = {}
    for qid, q in _to_vega_questions(questions).items():
        _, spans = build_prompt(engine.perc.tok, text, engine._internal(q), engine.cfg["max_len"],
                                opt_max_tokens=engine.cfg["opt_max_tokens"],
                                query_first_max_state=engine.cfg.get("query_first_max_state", 0), state_ids=state_ids)
        kept = spans["state"][1] - spans["state"][0] if state_ids else 0
        dropped[qid] = max(0, len(state_ids) - kept)
    return {"state_tokens": len(state_ids), "state_tokens_dropped": max(dropped.values(), default=0),
            "truncated": any(dropped.values()), "truncated_questions": [k for k, n in dropped.items() if n]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MANIFEST["models"], default="vega")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--acceleration", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--graphs", choices=("on", "off"), default="on")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    progress = StartupProgress(enabled=not args.check)
    progress.log("Python worker started")
    with contextlib.redirect_stdout(sys.stderr):
        with progress.stage("import_torch", "Importing PyTorch"):
            import torch

        with progress.stage("runtime_check", "Checking runtime and CUDA availability"):
            runtime = versions()
            for name, version in MANIFEST["runtime"].items():
                if runtime[name].split("+")[0] != version:
                    raise RuntimeError(f"{name}=={version} required, found {runtime[name]}; run npm run python:setup")
            cuda = torch.cuda.is_available()
        if args.check:
            info = {"versions": runtime, "cuda": cuda, "cudaRuntime": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0) if cuda else None}
            # Prove the installed CUDA build can execute, not only enumerate a GPU.
            if cuda:
                info["cudaOk"] = (torch.ones(2, device="cuda") + 1).tolist() == [2.0, 2.0]
        else:
            affinity = pin_process()
            profile = MANIFEST["models"][args.model]
            max_len = args.max_len or profile["maxLen"]
            if not 64 <= max_len <= profile["maxLen"]:
                raise ValueError(f"max_len must be between 64 and {profile['maxLen']}")
            with progress.stage("model_files", "Checking local model files"):
                model_dir = prepared(profile)
            # Separate process: thread settings cannot affect the legacy ONNX lane.
            torch.set_num_threads(1 if args.model == "vega" else 8)
            with progress.stage("import_laya", "Importing Laya model runtime"):
                # Agent is lazy in laya.__init__; resolve it here so these imports
                # are timed separately from checkpoint loading.
                from laya import Agent

            def construct(device):
                if device == "cuda":
                    with progress.stage("cuda_init", "Initializing CUDA"):
                        torch.cuda.init()
                if args.model == "vega":
                    with progress.stage("reader_kernels", "Configuring Vega reader kernels"):
                        configure_reader(device, args.acceleration)
                    with progress.stage(f"load_model_{device}", f"Loading Vega checkpoint on {device}"):
                        return Agent(vegaml=model_dir, backbone=prepared(profile["backbone"]), device=device, max_len=max_len)
                with progress.stage(f"load_model_{device}", f"Loading Multilingual checkpoint on {device}"):
                    return Agent(multilingual_runtime(profile, model_dir), device=device, backend="eager")

            agent, device, device_reason = load_auto_device(construct, args.device, torch)
            graph_replay = args.graphs == "on" and device == "cuda" and args.acceleration != "off"
            if args.model == "vega":
                acceleration = configure_reader(device, args.acceleration)
                graphs = None
                streaming = None
                if args.acceleration != "off" and device == "cuda":
                    from vega_stream import streamed_prefill
                    if graph_replay:
                        from vega_graph import HeadReplay
                        graphs = HeadReplay(agent.vega.engine.engine)
                    streaming = streamed_prefill(agent.vega.engine.perc, symbolic_masks=torch.cuda.get_device_capability()[0] >= 8,
                                                 capacity_tokens=max_len)
            else:
                agent.cfg["max_len"] = max_len
                with progress.stage("accelerator", "Selecting Multilingual execution backend"):
                    multilingual_acceleration = MultilingualAcceleration(agent, args.acceleration, graph_replay=graph_replay)
                progress.log(f"Multilingual backend: {agent.backend}" + (f" ({multilingual_acceleration.reason})" if multilingual_acceleration.reason else ""))
                graph_replay = graph_replay and agent.backend == "tilelang"
            base_temps = dict(getattr(agent, "temperature_by_options", {}))
            info = {"ready": True, "model": args.model, "device": device, "deviceReason": device_reason, "graphReplayEnabled": graph_replay, "maxLen": max_len,
                    "repo": profile["repo"], "revision": profile["revision"], "versions": runtime,
                    "loadMs": (time.perf_counter() - started) * 1000, "pid": os.getpid(), "affinity": affinity,
                    "startup": progress.report()}
            if args.model == "vega":
                info["acceleration"] = acceleration
            else:
                info["acceleration"] = multilingual_acceleration.stats()
    emit(info)
    if args.check:
        return

    first_prediction = True
    for line in sys.stdin:
        message = None
        try:
            message = json.loads(line)
            if message.get("op") == "close":
                return
            if message.get("op", "decide") != "decide":
                raise ValueError("unknown worker operation")
            state, questions = message["state"], message["questions"]
            if state is None or not isinstance(state, (str, dict, list)):
                raise ValueError("Python models accept a string, object or conversation list as state")
            t0 = time.perf_counter()
            if first_prediction:
                print(f"[inference] First decision on {device}; initializing any remaining kernels ...", file=sys.stderr, flush=True)
            with contextlib.redirect_stdout(sys.stderr):
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                fallbacks = getattr(agent, "cpu_fallback_count", 0)
                if args.model == "vega":
                    if message.get("calibration"):
                        raise ValueError("LocalLaya temperature tables cannot be applied to Vega")
                    readout = message.get("readout", "auto")
                    if readout not in ("auto", "engine"):
                        raise ValueError("readout must be auto or engine")
                    usage = vega_usage(agent, state, questions)
                    stream_before = (streaming["calls"], streaming["maskedCalls"]) if streaming else (0, 0)
                    result = agent.predict_long(state, questions, mode=readout)
                    result["usage"].update(usage)
                    if usage["truncated"]:
                        result["long"]["why"] = f"state truncated to fit {max_len} tokens; see usage.state_tokens_dropped"
                    # LocalLaya's score is the expected ordinal level. The Vega facade calls
                    # its argmax `score` and carries the expectation separately.
                    for answer in result["answers"].values():
                        if answer["type"] == "score" and "expected_score" in answer:
                            answer["selected_level"] = answer["score"]
                            answer["score"] = answer["expected_score"]
                else:
                    overrides = (message.get("calibration") or {}).get("temperature_by_options", {})
                    from laya.common import clamp_temperature

                    agent.temperature_by_options = {**base_temps, **{k: clamp_temperature(v) for k, v in overrides.items()}}
                    multilingual_acceleration.prepare()
                    result = agent.predict(state, questions, max_len=max_len)
                fallback = getattr(agent, "cpu_fallback_count", 0) > fallbacks
                actual = "cpu" if fallback else str(agent.device).split(":")[0]
                result["usage"]["context_tokens"] = max_len
                execution = {"mode": "torch-eager", "device": actual, "requestedDevice": args.device,
                             "deviceReason": device_reason, "graphReplayEnabled": graph_replay and actual == "cuda",
                             "maxLen": max_len, "cpuFallback": fallback,
                             "peakVramMiB": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device == "cuda" else 0,
                             "reservedVramMiB": round(torch.cuda.memory_reserved() / 1024**2, 1) if device == "cuda" else 0}
                if args.model == "vega":
                    execution["readout"] = readout
                    prefill = {**streaming, "used": streaming["calls"] > stream_before[0],
                               "symbolicMaskUsed": streaming["maskedCalls"] > stream_before[1]} if streaming else {"enabled": False, "used": False}
                    execution["acceleration"] = {**acceleration, "head": graphs.stats() if graphs else {"mode": "eager"}, "prefill": prefill}
                    if acceleration["reader"] == "fla-triton" or (graphs and graphs.last == "graph") or prefill["used"]:
                        execution["mode"] = "torch-accelerated"
                else:
                    execution["acceleration"] = multilingual_acceleration.stats(actual, fallback)
                    if execution["acceleration"]["backend"] == "tilelang":
                        execution["mode"] = "torch-accelerated"
                elapsed = (time.perf_counter() - t0) * 1000
            if first_prediction:
                print(f"[inference] First decision completed in {elapsed / 1000:.2f}s", file=sys.stderr, flush=True)
                first_prediction = False
            emit({"id": message["id"], "result": result, "exec": execution, "ms": elapsed})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            if "torch" in locals() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            emit({"id": message.get("id") if isinstance(message, dict) else None,
                  "error": f"{type(exc).__name__}: {exc}", "status": 400 if isinstance(exc, (ValueError, TypeError, KeyError)) else 500})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        sys.exit(1)
