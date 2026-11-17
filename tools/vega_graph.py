"""Bounded, validated CUDA Graph replay for Vega-U's fixed-size decision head.

The upstream engine and weights do the computation. Tracing includes both optional
branches (score rail and document memory); their tensor masks still select the actual
behavior at runtime. Shape/kwargs mismatch or a failed numerical check uses eager code.
"""
from collections import OrderedDict
import warnings

import torch

TENSOR_KEYS = ("logits", "z0", "zT", "s", "s_next", "centers", "log_a", "sigma", "kinetic", "grad_norm", "nearest", "steps_used")


class _TraceHead(torch.nn.Module):
    def __init__(self, source, forward):
        super().__init__()
        self.source = source
        self.original_forward = forward

    def forward(self, feats, opts, mask, qtype, chunks, chunk_mask):
        out = self.original_forward(feats, opts, mask, qtype, halt=True, return_traj=False,
                                    chunks=chunks, chunk_mask=chunk_mask)
        return tuple(out[k] for k in TENSOR_KEYS) + tuple(out["logits_traj"]) + tuple(out["H_traj"])


class HeadReplay:
    def __init__(self, head, max_entries=8):
        self.head = head
        self.original = head.forward
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.failed = OrderedDict()
        self.enabled = True
        self.hits = 0
        self.builds = 0
        self.last = "eager"
        head.forward = self.forward

    @staticmethod
    def unpack(tensors):
        out = dict(zip(TENSOR_KEYS, tensors[:len(TENSOR_KEYS)]))
        out.update(logits_traj=list(tensors[-4:-2]), H_traj=list(tensors[-2:]), gates=None)
        return out

    @torch.inference_mode()
    def forward(self, feats, opts, opt_mask, qtype, **kw):
        chunks, chunk_mask = kw.get("chunks"), kw.get("chunk_mask")
        eligible = (self.enabled and feats.device.type == "cuda" and chunks is not None and chunk_mask is not None
                    and kw.get("halt") is True and kw.get("return_traj") is False
                    and not kw.get("prestandardized", False) and kw.get("n_steps") is None
                    and len(self.head.gate_nets) == 0)
        if not eligible:
            self.last = "eager"
            return self.original(feats, opts, opt_mask, qtype, **kw)
        inputs = (feats, opts, opt_mask, qtype, chunks, chunk_mask)
        key = tuple((tuple(t.shape), str(t.dtype), str(t.device)) for t in inputs)
        if key in self.failed:
            self.last = "eager"
            return self.original(feats, opts, opt_mask, qtype, **kw)
        entry = self.entries.get(key)
        if entry is None:
            expected = self.original(feats, opts, opt_mask, qtype, **kw)
            try:
                entry = self.build(inputs, expected)
            except Exception as exc:
                while len(self.failed) >= self.max_entries:
                    self.failed.popitem(last=False)
                self.failed[key] = f"{type(exc).__name__}: {exc}"
                warnings.warn(f"Vega head graph unavailable: {self.failed[key]}", RuntimeWarning)
                self.last = "eager"
                return expected
            while len(self.entries) >= self.max_entries:
                self.entries.popitem(last=False)
            self.entries[key] = entry
            self.builds += 1
        self.entries.move_to_end(key)
        static, graph, output, traced = entry
        for dst, src in zip(static, inputs):
            dst.copy_(src)
        graph.replay()
        self.hits += 1
        self.last = "graph"
        # Callers may retain a view's outputs while another view replays this graph.
        return self.unpack(tuple(t.clone() for t in output))

    @torch.inference_mode()
    def build(self, inputs, expected):
        example = tuple(t.clone() for t in inputs)
        example[3].fill_(1)  # score rail must be traced; actual qtype masks still choose it
        example[5].fill_(True)  # memory reader must be traced; an empty mask returns zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", message=r"`torch\.jit\.trace.*deprecated.*", category=FutureWarning)
            traced = torch.jit.trace(_TraceHead(self.head, self.original).eval(), example, check_trace=False, strict=True)
        static = tuple(t.clone() for t in inputs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                traced(*static)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = traced(*static)
        torch.cuda.current_stream().wait_stream(stream)
        graph.replay()
        actual = self.unpack(output)
        for name in TENSOR_KEYS:
            if not torch.allclose(actual[name], expected[name], atol=1e-4, rtol=1e-4):
                raise ValueError(f"graph disagrees with eager on {name}")
        return static, graph, output, traced

    def stats(self):
        return {"mode": self.last, "hits": self.hits, "builds": self.builds,
                "entries": len(self.entries), "failed": list(self.failed.values())}
