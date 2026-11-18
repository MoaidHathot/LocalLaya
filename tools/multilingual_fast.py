"""Bounded TileLang acceleration for the pinned ModernBERT Multilingual checkpoint."""
from collections import OrderedDict
import importlib.metadata
import json
import os
from pathlib import Path
import weakref

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TILELANG_CACHE_DIR", str(ROOT / ".laya/tilelang"))


class GraphCache(OrderedDict):
    def __init__(self, limit=8):
        super().__init__()
        self.limit = limit
        self.hits = 0
        self.builds = 0
        self.last = None
        self.mode = "kernels"
        self.candidate = None

    def get(self, key, default=None):
        self.last = list(key)
        if key in self:
            self.move_to_end(key)
            self.hits += 1
            return self[key]
        return default

    def __setitem__(self, key, value):
        if key not in self:
            while len(self) >= self.limit:
                self.popitem(last=False)
            self.builds += 1
        super().__setitem__(key, value)

    def stats(self):
        return {"entries": len(self), "limit": self.limit, "hits": self.hits, "builds": self.builds, "bucket": self.last, "mode": self.mode}


def repeat_graphs(fast):
    """Use fused kernels immediately; capture only when a new shape is repeated consecutively."""
    if getattr(fast, "_local_laya_graph_schedule", False) or not hasattr(type(fast), "_encode_graphed"):
        return
    original = type(fast)._encode_graphed
    owner = weakref.ref(fast)

    def run(ids, lens, qtype):
        instance = owner()
        if instance is None:
            raise RuntimeError("Multilingual backend was released")
        cache = instance.graphs
        key = tuple(ids.shape)
        repeated = cache.candidate == key
        cache.candidate = key
        cache.last = list(key)
        if key in cache:
            cache.mode = "replay"
            return original(instance, ids, lens, qtype)
        if repeated:
            # Release before capturing, so a ninth large shape cannot exceed the bound transiently.
            if len(cache) >= cache.limit:
                cache.popitem(last=False)
            cache.mode = "capture"
            return original(instance, ids, lens, qtype)
        cache.mode = "kernels"
        return instance._encode(ids, lens, qtype)

    fast._encode_graphed = run
    fast._local_laya_graph_schedule = True


class MultilingualAcceleration:
    def __init__(self, agent, requested="auto", graph_replay=True):
        self.agent = agent
        self.requested = requested
        self.graph_replay = graph_replay
        self.versions = {}
        self.reason = None
        if requested == "off" or agent.device.type != "cuda":
            self.reason = "requested off" if requested == "off" else "CPU uses the upstream eager optimizations"
            return
        try:
            pins = json.loads((ROOT / "data/python-models.json").read_text(encoding="utf-8"))["multilingualAcceleration"]
            for package, version in pins.items():
                self.versions[package] = importlib.metadata.version(package)
                if self.versions[package] != version:
                    raise ImportError(f"{package}=={version} required; run npm run multilingual:setup")
            agent.set_backend("tilelang", strict=True, use_graphs=graph_replay)
            self.prepare()
        except Exception as exc:
            # Optional acceleration must leave the existing eager model usable.
            agent.set_backend("eager")
            self.reason = f"TileLang unavailable: {type(exc).__name__}: {exc}"

    def prepare(self):
        # Upstream can reinstall the backend after an OOM/CPU retry. Bound the new
        # cache too, without retaining an old FastLaya and its device weights.
        fast = getattr(self.agent.backend_object, "fast", None)
        if fast is not None:
            fast.use_graphs = self.graph_replay
        if fast is not None and not isinstance(fast.graphs, GraphCache):
            cache = GraphCache()
            cache.update(fast.graphs)
            fast.graphs = cache
        if fast is not None:
            repeat_graphs(fast)
        return fast

    def stats(self, actual_device=None, cpu_fallback=False):
        fast = self.prepare()
        active = fast is not None and self.agent.backend == "tilelang"
        used = active and actual_device != "cpu" and not cpu_fallback
        return {"requested": self.requested, "backend": "tilelang" if used else "eager",
                "versions": self.versions, "reason": self.reason,
                "graphs": {**(fast.graphs.stats() if fast is not None else {"entries": 0, "hits": 0, "builds": 0}), "enabled": used and self.graph_replay},
                "kernelVariants": len(fast._attn_k) if fast is not None else 0}


def check():
    from laya_worker import MANIFEST, prepared, multilingual_runtime
    from vega_fast import pin_process
    import torch
    import laya

    if not torch.cuda.is_available():
        raise RuntimeError("Multilingual CUDA acceleration requires a CUDA-enabled PyTorch build")
    pin_process()
    torch.set_num_threads(8)
    spec = MANIFEST["models"]["multilingual"]
    agent = laya.Agent(multilingual_runtime(spec, prepared(spec)), device="cuda", backend="eager")
    agent.cfg["max_len"] = spec["maxLen"]
    controller = MultilingualAcceleration(agent)
    if agent.backend != "tilelang":
        raise RuntimeError(controller.reason)
    questions = {"team": {"type": "choice", "instructions": "Which team handles this request?", "criteria": {
        "billing": "invoices, payments and refunds", "technical": "bugs and outages"}}}
    for _ in range(3):  # fused-only, graph capture, graph replay
        result = agent.predict("تم خصم المبلغ مرتين من حسابي. أرجو إعادة المبلغ الزائد.", questions)
    torch.cuda.synchronize()
    if not all(0 <= p <= 1 for p in result["answers"]["team"]["probabilities"].values()):
        raise RuntimeError("Multilingual check returned invalid probabilities")
    return {**controller.stats(), "cudaOk": True, "gpu": torch.cuda.get_device_name(),
            "choice": result["answers"]["team"]["choice"], "cache": os.environ["TILELANG_CACHE_DIR"]}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
