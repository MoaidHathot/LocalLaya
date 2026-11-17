"""CPU-only checks of bounded graph ownership and optional-acceleration fallbacks."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import importlib.metadata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from multilingual_fast import GraphCache, MultilingualAcceleration, repeat_graphs


class FastStub:
    def __init__(self):
        self.graphs = GraphCache(2)

    def _encode(self, ids, lens, qtype):
        return "kernels"

    def _encode_graphed(self, ids, lens, qtype):
        key = tuple(ids.shape)
        if self.graphs.get(key) is None:
            self.graphs[key] = object()
        return "graph"


class AgentStub:
    def __init__(self, device="cuda"):
        self.device = SimpleNamespace(type=device)
        self.backend = "eager"
        self.backend_object = None
        self.calls = []

    def set_backend(self, name, **options):
        self.calls.append(name)
        self.backend = name


class MultilingualAccelerationTests(unittest.TestCase):
    def test_one_off_shapes_skip_capture_but_repeated_shapes_replay(self):
        fast = FastStub()
        repeat_graphs(fast)
        a, b = SimpleNamespace(shape=(1, 64)), SimpleNamespace(shape=(1, 8192))
        for ids in (a, b, a):
            self.assertEqual(fast._encode_graphed(ids, None, None), "kernels")
        self.assertEqual(fast.graphs.builds, 0)
        self.assertEqual(fast._encode_graphed(a, None, None), "graph")
        self.assertEqual(fast.graphs.mode, "capture")
        self.assertEqual(fast._encode_graphed(b, None, None), "kernels")
        self.assertEqual(fast._encode_graphed(a, None, None), "graph")
        self.assertEqual(fast.graphs.mode, "replay")
        self.assertEqual(fast.graphs.hits, 1)

    def test_graph_cache_bounds_live_shapes_and_tracks_reuse(self):
        cache = GraphCache(limit=2)
        cache[(1, 64)] = object()
        cache[(4, 128)] = object()
        self.assertIsNotNone(cache.get((1, 64)))
        cache[(1, 8192)] = object()
        self.assertEqual(list(cache), [(1, 64), (1, 8192)])
        self.assertEqual(cache.builds, 3)
        self.assertEqual(cache.hits, 1)

    def test_off_and_cpu_do_not_require_optional_packages(self):
        with patch("importlib.metadata.version", side_effect=AssertionError("dependency probe")):
            for device, requested in [("cpu", "auto"), ("cuda", "off")]:
                agent = AgentStub(device)
                controller = MultilingualAcceleration(agent, requested)
                self.assertEqual(controller.stats()["backend"], "eager")
                self.assertEqual(agent.calls, [])

    def test_missing_dependency_preserves_eager(self):
        agent = AgentStub()
        with patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError("tilelang")):
            controller = MultilingualAcceleration(agent)
        self.assertEqual(controller.stats()["backend"], "eager")
        self.assertIn("unavailable", controller.reason)
        self.assertEqual(agent.calls, ["eager"])

    def test_reinstalled_backend_gets_a_new_bounded_cache(self):
        agent = AgentStub()
        controller = MultilingualAcceleration(agent, "off")
        agent.backend = "tilelang"
        for _ in range(2):
            fast = SimpleNamespace(graphs={}, _attn_k={})
            agent.backend_object = SimpleNamespace(fast=fast)
            controller.prepare()
            self.assertIsInstance(fast.graphs, GraphCache)
            self.assertEqual(controller.stats("cuda")["backend"], "tilelang")
            self.assertEqual(controller.stats("cpu", cpu_fallback=True)["backend"], "eager")


if __name__ == "__main__":
    unittest.main()
