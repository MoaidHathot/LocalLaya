"""Bound the working set of long causal prefills while preserving the full cache and features."""
import torch
from transformers.cache_utils import DynamicLayer


class _ReservedKV(DynamicLayer):
    """Private, fixed-storage KV layer with dynamic-length views for Vega's branchable cache."""
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self._keys_storage = key_states.new_empty((*key_states.shape[:-2], self.capacity, key_states.shape[-1]))
        self._values_storage = value_states.new_empty((*value_states.shape[:-2], self.capacity, value_states.shape[-1]))
        self.keys = self._keys_storage[..., :0, :]
        self.values = self._values_storage[..., :0, :]
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        start = self.get_seq_length()
        end = start + key_states.shape[-2]
        if end > self.capacity:
            raise ValueError(f"Vega KV cache needs {end} tokens, capacity is {self.capacity}")
        self._keys_storage[..., start:end, :].copy_(key_states)
        self._values_storage[..., start:end, :].copy_(value_states)
        self.keys = self._keys_storage[..., :end, :]
        self.values = self._values_storage[..., :end, :]
        return self.keys, self.values


def streamed_prefill(perception, chunk_tokens=2048, min_tokens=32768, symbolic_masks=False, capacity_tokens=73728):
    from torch.nn.attention.bias import CausalBias, causal_lower_right
    from transformers.integrations import sdpa_attention

    # A symbolic lower-right causal mask is supported by Flash SDPA with grouped
    # KV heads. Transformers' generic mask check otherwise repeats the KV tensors.
    if not hasattr(sdpa_attention, "_local_laya_use_gqa"):
        original_gqa = sdpa_attention.use_gqa_in_sdpa
        def use_gqa(mask, key, value):
            if isinstance(mask, CausalBias) and key.is_cuda and key.shape[-1] == value.shape[-1] <= 256:
                return True
            return original_gqa(mask, key, value)
        sdpa_attention._local_laya_use_gqa = original_gqa
        sdpa_attention.use_gqa_in_sdpa = use_gqa
    original = perception._run
    control = {"enabled": True, "symbolicMasks": symbolic_masks, "chunkTokens": chunk_tokens, "threshold": min_tokens,
               "calls": 0, "lastChunks": 0, "maskedCalls": 0, "fixedKvStorage": True, "capacityTokens": capacity_tokens}

    def cached_mask(ids, cache):
        seen = cache.get_seq_length() if cache is not None else 0
        if control["symbolicMasks"] and seen > 0:
            control["maskedCalls"] += 1
            return {"full_attention": causal_lower_right(ids.shape[1], seen + ids.shape[1]), "linear_attention": None}
        return None

    def run(ids, mask=None, cache=None):
        length = ids.shape[1]
        if (not control["enabled"] or ids.device.type != "cuda" or length <= max(min_tokens, chunk_tokens)
                or mask is not None or cache is None or cache.get_seq_length() != 0):
            if (control["enabled"] and ids.is_cuda and mask is None and cache is not None
                    and cache.get_seq_length() + length > min_tokens):
                mask = cached_mask(ids, cache)
            return original(ids, mask, cache)
        # Every block attends to all prior blocks through the same KV / recurrent cache.
        # Stitch the tapped hidden states back together, so upstream's pooling and memory
        # summaries consume exactly the same token positions as a whole-prefix forward.
        # DynamicLayer's repeated torch.cat grows many differently sized CUDA allocations;
        # at 70K we measured 14 GiB reserved for only 3.2 GiB of live tensors. Fixed storage
        # avoids that fragmentation. Visible views retain the actual length, not capacity.
        # deepcopy(cache) used by each question also copies these buffers and their views.
        for i, layer in enumerate(cache.layers):
            if type(layer) is DynamicLayer and not layer.is_initialized:
                cache.layers[i] = _ReservedKV(capacity_tokens)
        features = {}
        for start in range(0, length, chunk_tokens):
            end = min(length, start + chunk_tokens)
            part = original(ids[:, start:end], cached_mask(ids[:, start:end], cache), cache)
            for layer, hidden in part.items():
                if layer not in features:
                    features[layer] = torch.empty((hidden.shape[0], length, hidden.shape[2]), dtype=hidden.dtype, device=hidden.device)
                features[layer][:, start:end].copy_(hidden)
        perception._cap = features
        control["calls"] += 1
        control["lastChunks"] = (length + chunk_tokens - 1) // chunk_tokens
        return features

    perception._run = run
    return control
