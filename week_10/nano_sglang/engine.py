"""Part 2: Inference Engine

Two phases:
  Prefill:  process entire prompt in one pass -> compute-bound
  Decode:   generate one token at a time from cache -> memory-bound
"""

import sys
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from .model import Model, Tokenizer
from .sampling import SamplingParams, sample_token
from .sequence import Sequence, SequenceStatus


# ---------------------------------------------------------------------------
# Cache compatibility layer
# ---------------------------------------------------------------------------
# The assignment skeleton uses DynamicCache.key_cache[layer] and
# DynamicCache.value_cache[layer], which worked in transformers <= 4.51.
# In newer transformers (v5+), DynamicCache stores per-layer objects in
# self.layers, each with .keys and .values tensors.
# We detect which API is available and extract/build caches accordingly.
# ---------------------------------------------------------------------------

def _extract_kv_per_layer(cache, num_layers):
    """Extract list of (key_tensor, value_tensor) for each layer from a cache.

    Returns list of (k, v) tuples, or None if extraction fails.
    Each tensor has shape [batch, heads, seq_len, head_dim].
    """
    # Strategy 1: Old-style .key_cache / .value_cache lists
    if hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
        try:
            kc = cache.key_cache
            vc = cache.value_cache
            if isinstance(kc, list) and len(kc) >= num_layers:
                return [(kc[i], vc[i]) for i in range(num_layers)]
            # Maybe it's a subscriptable non-list
            _ = kc[0]
            return [(kc[i], vc[i]) for i in range(num_layers)]
        except (IndexError, TypeError, KeyError):
            pass

    # Strategy 2: New-style .layers list with .keys/.values per layer
    if hasattr(cache, 'layers'):
        layers = cache.layers
        try:
            n = len(layers)
        except TypeError:
            n = 0
        if n >= num_layers:
            result = []
            for i in range(num_layers):
                layer = layers[i]
                k, v = None, None
                # New transformers: DynamicLayer has .keys and .values
                if hasattr(layer, 'keys') and hasattr(layer, 'values'):
                    k_candidate = layer.keys
                    v_candidate = layer.values
                    # Make sure they're actual tensors (not None or empty)
                    if isinstance(k_candidate, torch.Tensor) and k_candidate.numel() > 0:
                        k, v = k_candidate, v_candidate
                if k is None and hasattr(layer, 'key_cache') and hasattr(layer, 'value_cache'):
                    k, v = layer.key_cache, layer.value_cache
                if k is None and hasattr(layer, 'key_states') and hasattr(layer, 'value_states'):
                    k, v = layer.key_states, layer.value_states
                if k is None:
                    # Try tuple unpacking
                    try:
                        k, v = layer[0], layer[1]
                    except (TypeError, IndexError):
                        break
                if k is not None and isinstance(k, torch.Tensor):
                    result.append((k, v))
                else:
                    break
            if len(result) == num_layers:
                return result

    # Strategy 3: to_legacy_cache()
    if hasattr(cache, 'to_legacy_cache'):
        try:
            legacy = cache.to_legacy_cache()
            return [(legacy[i][0], legacy[i][1]) for i in range(num_layers)]
        except Exception:
            pass

    # Strategy 4: Indexing the cache directly
    try:
        result = []
        for i in range(num_layers):
            k, v = cache[i]
            result.append((k, v))
        return result
    except (TypeError, IndexError, KeyError, ValueError):
        pass

    return None


def _build_cache_from_kv_list(kv_list):
    """Build a DynamicCache with .key_cache/.value_cache from a list of (k,v) tuples.

    This creates a cache object that works with the assignment skeleton code
    which accesses cache.key_cache[layer_idx] and cache.value_cache[layer_idx].
    """
    cache = DynamicCache()
    # Force-create the lists regardless of transformers version
    if not hasattr(cache, 'key_cache') or not isinstance(getattr(cache, 'key_cache', None), list):
        cache.key_cache = []
    if not hasattr(cache, 'value_cache') or not isinstance(getattr(cache, 'value_cache', None), list):
        cache.value_cache = []
    for k, v in kv_list:
        cache.key_cache.append(k)
        cache.value_cache.append(v)
    return cache


def _ensure_extractable(cache, num_layers):
    """Convert cache to one where _extract_kv_per_layer will succeed.

    If extraction already works, returns the cache as-is.
    Otherwise, builds a new DynamicCache with explicit key_cache/value_cache lists.
    """
    kv = _extract_kv_per_layer(cache, num_layers)
    if kv is not None:
        return _build_cache_from_kv_list(kv)

    # Last resort: dump debug info
    print(f"[CACHE_COMPAT] FAILED to extract KV. type={type(cache)}", file=sys.stderr)
    print(f"[CACHE_COMPAT] dir={[a for a in dir(cache) if not a.startswith('__')]}", file=sys.stderr)
    if hasattr(cache, 'layers'):
        layers = cache.layers
        try:
            n = len(layers)
            print(f"[CACHE_COMPAT] len(layers)={n}, num_layers={num_layers}", file=sys.stderr)
            if n > 0:
                l0 = layers[0]
                print(f"[CACHE_COMPAT] layer[0] type={type(l0)}", file=sys.stderr)
                print(f"[CACHE_COMPAT] layer[0] dir={[a for a in dir(l0) if not a.startswith('__')]}", file=sys.stderr)
                if hasattr(l0, '__dict__'):
                    print(f"[CACHE_COMPAT] layer[0].__dict__ keys={list(l0.__dict__.keys())}", file=sys.stderr)
                    for dk, dv in l0.__dict__.items():
                        if isinstance(dv, torch.Tensor):
                            print(f"[CACHE_COMPAT]   {dk}: Tensor shape={dv.shape}", file=sys.stderr)
                        else:
                            print(f"[CACHE_COMPAT]   {dk}: {type(dv).__name__} = {dv}", file=sys.stderr)
        except Exception as e:
            print(f"[CACHE_COMPAT] Error inspecting layers: {e}", file=sys.stderr)
    return cache


class Engine:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = Model(model_path, device=device)
        self.tokenizer = Tokenizer(model_path)
        self.device = device

    def prefill(self, seq: Sequence, sampling_params: SamplingParams) -> int:
        """Process all prompt tokens in one forward pass, return first generated token.
        Should also store past_key_values in seq and set status to DECODING."""
        input_ids = torch.tensor([seq.prompt_token_ids], device=self.device)
        logits, past_key_values = self.model.forward(input_ids)
        first_token = sample_token(logits[:, -1, :], sampling_params).item()

        seq.past_key_values = past_key_values
        seq.status = SequenceStatus.DECODING
        return first_token

    def decode_step(self, seq: Sequence, sampling_params: SamplingParams) -> int:
        """Generate one token for a single sequence using cached KV."""
        last_token = seq.output_token_ids[-1]
        input_ids = torch.tensor([[last_token]], device=self.device)
        logits, past_key_values = self.model.forward(input_ids, past_key_values=seq.past_key_values)
        next_token = sample_token(logits[:, -1, :], sampling_params).item()
        seq.past_key_values = past_key_values
        return next_token

    def decode_batch(self, sequences: list[Sequence], sampling_params: SamplingParams) -> list[int]:
        """Generate one token for multiple sequences in a single GPU forward pass."""
        if not sequences:
            return []
        if len(sequences) == 1:
            return [self.decode_step(sequences[0], sampling_params)]

        n = len(sequences)
        num_layers = self.model.num_layers
        input_ids = torch.tensor(
            [[seq.output_token_ids[-1]] for seq in sequences], device=self.device,
        )

        # Extract KV tensors from each sequence's cache
        all_seq_kvs = []
        for seq in sequences:
            kv = _extract_kv_per_layer(seq.past_key_values, num_layers)
            if kv is None:
                # Fallback: convert cache and retry
                seq.past_key_values = _ensure_extractable(seq.past_key_values, num_layers)
                kv = _extract_kv_per_layer(seq.past_key_values, num_layers)
            if kv is None:
                # Total fallback: decode one at a time
                print("[DECODE_BATCH] Cache extraction failed, falling back to sequential decode", file=sys.stderr)
                return [self.decode_step(seq, sampling_params) for seq in sequences]
            all_seq_kvs.append(kv)

        cache_lens = [all_seq_kvs[i][0][0].shape[2] for i in range(n)]  # seq_len from first layer's key
        max_len = max(cache_lens)

        # Build batched cache with explicit key_cache/value_cache lists
        batched_cache = _build_cache_from_kv_list([])
        for layer_idx in range(num_layers):
            padded_keys, padded_values = [], []
            for seq_idx in range(n):
                k, v = all_seq_kvs[seq_idx][layer_idx]
                pad = max_len - k.shape[2]
                if pad > 0:
                    k = F.pad(k, (0, 0, pad, 0))
                    v = F.pad(v, (0, 0, pad, 0))
                padded_keys.append(k)
                padded_values.append(v)
            batched_cache.key_cache.append(torch.cat(padded_keys, dim=0))
            batched_cache.value_cache.append(torch.cat(padded_values, dim=0))

        attn_mask = torch.zeros(n, max_len + 1, device=self.device, dtype=torch.long)
        for i, cl in enumerate(cache_lens):
            attn_mask[i, max_len - cl:] = 1

        position_ids = torch.tensor([[cl] for cl in cache_lens], device=self.device)

        logits, new_cache = self.model.forward(
            input_ids, past_key_values=batched_cache,
            position_ids=position_ids, attention_mask=attn_mask,
        )

        # Extract KV from the new batched cache
        new_kv = _extract_kv_per_layer(new_cache, num_layers)
        if new_kv is None:
            new_cache = _ensure_extractable(new_cache, num_layers)
            new_kv = _extract_kv_per_layer(new_cache, num_layers)

        tokens = sample_token(logits[:, -1, :], sampling_params)

        if new_kv is not None:
            # Split batched cache back into per-sequence caches
            for i, seq in enumerate(sequences):
                real_len = cache_lens[i] + 1
                pad = max_len - cache_lens[i]
                per_seq_kv = []
                for layer_idx in range(num_layers):
                    k_all, v_all = new_kv[layer_idx]
                    k = k_all[i:i+1, :, pad:pad + real_len, :]
                    v = v_all[i:i+1, :, pad:pad + real_len, :]
                    per_seq_kv.append((k.clone(), v.clone()))
                seq.past_key_values = _build_cache_from_kv_list(per_seq_kv)
        else:
            # Fallback: re-prefill each sequence (expensive but correct)
            print("[DECODE_BATCH] Could not extract new_cache, re-prefilling", file=sys.stderr)
            for i, seq in enumerate(sequences):
                all_tokens = seq.prompt_token_ids + seq.output_token_ids
                input_all = torch.tensor([all_tokens], device=self.device)
                _, past_kv = self.model.forward(input_all)
                seq.past_key_values = past_kv

        return [t.item() for t in tokens]

    def generate(self, prompt: str, sampling_params: SamplingParams = None) -> str:
        """Generate text for a single prompt. Wire prefill + decode loop together."""
        if sampling_params is None:
            sampling_params = SamplingParams()

        prompt_tokens = self.tokenizer.encode(prompt)
        seq = Sequence(
            seq_id=0,
            prompt_token_ids=prompt_tokens,
            max_tokens=sampling_params.max_tokens,
        )

        first_token = self.prefill(seq, sampling_params)
        seq.output_token_ids.append(first_token)

        if first_token == self.tokenizer.eos_token_id:
            seq.status = SequenceStatus.FINISHED
            return self.tokenizer.decode(seq.output_token_ids)

        while seq.num_generated < sampling_params.max_tokens:
            next_token = self.decode_step(seq, sampling_params)
            seq.output_token_ids.append(next_token)
            if next_token == self.tokenizer.eos_token_id:
                break

        seq.status = SequenceStatus.FINISHED
        return self.tokenizer.decode(seq.output_token_ids)
