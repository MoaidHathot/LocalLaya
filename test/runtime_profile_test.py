"""CPU-only checks of automatic device selection and explicit-device precedence."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from laya_worker import load_auto_device


class OOM(RuntimeError):
    pass


class DeviceTests(unittest.TestCase):
    def torch(self, available=True):
        return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available, OutOfMemoryError=OOM, empty_cache=lambda: None))

    def test_auto_and_explicit_device(self):
        for requested, available, expected in [("auto", True, "cuda"), ("auto", False, "cpu"), ("cpu", True, "cpu")]:
            calls = []
            def load(device):
                calls.append(device)
                return SimpleNamespace(device=device)
            _, device, _ = load_auto_device(load, requested, self.torch(available))
            self.assertEqual(device, expected)
            self.assertEqual(calls, [expected])

    def test_only_auto_retries_gpu_oom(self):
        calls = []
        def load(device):
            calls.append(device)
            if device == "cuda":
                raise OOM("test allocation failure")
            return SimpleNamespace(device=device)
        _, device, reason = load_auto_device(load, "auto", self.torch())
        self.assertEqual(calls, ["cuda", "cpu"])
        self.assertEqual(device, "cpu")
        self.assertIn("out of memory", reason)
        calls.clear()
        with self.assertRaises(OOM):
            load_auto_device(load, "cuda", self.torch())
        self.assertEqual(calls, ["cuda"])

    def test_model_errors_are_not_hidden_and_forced_cuda_never_silently_uses_cpu(self):
        def broken(device):
            raise RuntimeError("bad checkpoint")
        with self.assertRaisesRegex(RuntimeError, "bad checkpoint"):
            load_auto_device(broken, "auto", self.torch())
        cpu = lambda _: SimpleNamespace(device="cpu")
        self.assertEqual(load_auto_device(cpu, "auto", self.torch())[1], "cpu")
        with self.assertRaisesRegex(RuntimeError, "requested cuda"):
            load_auto_device(cpu, "cuda", self.torch())
        with self.assertRaisesRegex(RuntimeError, "no working CUDA"):
            load_auto_device(cpu, "cuda", self.torch(False))


if __name__ == "__main__":
    unittest.main()
