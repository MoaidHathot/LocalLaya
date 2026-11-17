"""Process-local Qwen kernel selection for the pinned Vega reader. No weight changes."""
import importlib.metadata
import inspect
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".laya/triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".laya/torchinductor"))
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

KERNEL_NAMES = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update")


def reference_kernels():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    # Transformers auto-imports FLA even for CPU models if it is installed. Restore
    # its actual torch implementations explicitly for CPU and the off/reference mode.
    refs = getattr(qwen, "_local_laya_reference", None)
    if refs is None:
        refs = {name: inspect.unwrap(getattr(qwen, name)) for name in KERNEL_NAMES}
        qwen._local_laya_reference = refs
    for name, function in refs.items():
        setattr(qwen, name, function)
    return qwen


def fused_kernels(convolution=True):
    qwen = reference_kernels()
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

    for name, fn in (("torch_chunk_gated_delta_rule", chunk_gated_delta_rule),
                     ("torch_recurrent_gated_delta_rule", fused_recurrent_gated_delta_rule)):
        params = set(inspect.signature(fn).parameters)
        def wrapped(*args, _fn=fn, _params=params, **kw):
            return _fn(*args, **{k: v for k, v in kw.items() if k in _params})
        setattr(qwen, name, wrapped)
    if convolution:
        from fla.modules.causal_conv1d import causal_conv1d

        def conv(x, weight, bias=None, activation=None, **kw):
            out, _ = causal_conv1d(x.transpose(1, 2), weight=weight, bias=bias, activation=activation, backend="triton")
            return out.transpose(1, 2)
        qwen.causal_conv1d_fn = conv


def configure_reader(device, requested="auto"):
    reference_kernels()
    info = {"requested": requested, "reader": "torch-reference", "versions": {}, "reason": None}
    if requested == "off" or device != "cuda":
        info["reason"] = "requested off" if requested == "off" else "CPU uses reference kernels"
        return info
    try:
        pins = json.loads((ROOT / "data/python-models.json").read_text(encoding="utf-8"))["acceleration"]
        for package in ("fla-core", "flash-linear-attention", "einops", "triton-windows" if os.name == "nt" else "triton"):
            info["versions"][package] = importlib.metadata.version(package)
            if info["versions"][package] != pins[package]:
                raise ImportError(f"{package}=={pins[package]} required; run npm run vega:setup")
        fused_kernels()
        info["reader"] = "fla-triton"
    except (ImportError, OSError) as exc:
        reference_kernels()
        info["reason"] = f"optional kernels unavailable: {exc}"
    return info


def pin_process():
    """Use the project's existing Windows P-core convention, only on this child process."""
    count = int(os.environ.get("LAYA_PCORE_LOGICAL", "16"))
    if os.name != "nt" or count <= 0 or count >= (os.cpu_count() or 1) or count > 62:
        return {"applied": False}
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    handle = kernel.GetCurrentProcess()
    current, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(handle, ctypes.byref(current), ctypes.byref(system)):
        return {"applied": False, "error": ctypes.get_last_error()}
    mask = ((1 << count) - 1) & current.value
    ok = bool(mask and kernel.SetProcessAffinityMask(handle, mask))
    return {"applied": ok, "mask": hex(mask)}


def check():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Vega acceleration requires CUDA; the CPU reference mode remains available")
    info = configure_reader("cuda")
    if info["reader"] != "fla-triton":
        raise RuntimeError(info["reason"])
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    torch.manual_seed(0)
    q, k, v = [torch.randn(1, 64, 16, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    out, _ = qwen.torch_chunk_gated_delta_rule(q, k, v, g=-torch.rand(1, 64, 16, device="cuda"),
        beta=torch.rand(1, 64, 16, device="cuda", dtype=torch.bfloat16), use_qk_l2norm_in_kernel=True)
    conv = qwen.causal_conv1d_fn(torch.randn(1, 6144, 64, device="cuda", dtype=torch.bfloat16),
        torch.randn(6144, 4, device="cuda", dtype=torch.bfloat16), activation="silu")
    # Unequal query/key lengths must use lower-right causality, not drop earlier history.
    from torch.nn.attention.bias import causal_lower_right
    q, k, v = [torch.randn(*shape, device="cuda", dtype=torch.bfloat16) for shape in ((1, 8, 7, 256), (1, 2, 19, 256), (1, 2, 19, 256))]
    mask = torch.ones(7, 19, device="cuda", dtype=torch.bool).tril(diagonal=12)
    dense = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
    symbolic = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=causal_lower_right(7, 19), enable_gqa=True)
    torch.testing.assert_close(symbolic, dense, atol=0.015, rtol=0.01)
    torch.cuda.synchronize()
    if not torch.isfinite(out).all() or not torch.isfinite(conv).all():
        raise RuntimeError("fused kernel check returned non-finite output")
    return {**info, "cudaOk": True, "lowerRightMaskOk": True, "gpu": torch.cuda.get_device_name(), "cache": os.environ["TRITON_CACHE_DIR"]}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
