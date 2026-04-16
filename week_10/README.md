# Week 10: Inference Systems — nano-sglang

A minimal LLM inference engine implementing KV caching, prefill/decode phases, continuous batching, and paged memory management, built on top of Qwen3-0.6B.

## How to Run

```bash
# Run all tests on GPU via Modal
modal run modal_run.py::test

# Run the generation example
modal run modal_run.py::run

# Run the Part 4 benchmark
modal run modal_run.py::benchmark
```

---

## Part 1: KV Cache — `nano_sglang/kv_cache.py`

**What was implemented:** `update()` and `get()` methods on the `KVCache` class.

The KV cache pre-allocates tensors of shape `[max_batch_size, num_heads, max_seq_len, head_dim]` per layer—one for keys and one for values. This avoids recomputing attention over already-seen tokens during decoding.

**`update(layer_idx, batch_idx, key, value, start_pos)`:**
Writes incoming key/value tensors (shape `[1, num_heads, seq_len, head_dim]`) into the pre-allocated cache at the correct position. During prefill, `start_pos=0` and `seq_len` equals the prompt length, filling in the initial cache. During each decode step, `start_pos` advances by one and `seq_len=1`, appending a single token's KV pair.

```python
seq_len = key.shape[2]
self.keys[layer_idx][batch_idx, :, start_pos:start_pos + seq_len, :] = key[0]
self.values[layer_idx][batch_idx, :, start_pos:start_pos + seq_len, :] = value[0]
```

**`get(layer_idx, batch_idx, seq_len)`:**
Returns a slice of the cached tensors up to `seq_len`, maintaining the `[1, num_heads, seq_len, head_dim]` shape expected by the model.

```python
k = self.keys[layer_idx][batch_idx:batch_idx + 1, :, :seq_len, :]
v = self.values[layer_idx][batch_idx:batch_idx + 1, :, :seq_len, :]
```

Without caching, generating 100 tokens from a 1000-token prompt would require processing the entire growing sequence at every step (~100K token computations). With KV caching, we process the prompt once and compute only the new token at each step (~1,100 total computations).

---

## Part 2: Engine — `nano_sglang/engine.py`

**What was implemented:** `prefill()` and `generate()` methods on the `Engine` class.

**`prefill(seq, sampling_params)`:**
This is the compute-bound phase. All prompt tokens are fed through the model in a single forward pass. The method:
1. Converts `seq.prompt_token_ids` into a `[1, prompt_len]` tensor.
2. Runs one forward pass through the model, producing logits and a `DynamicCache` of past key/values.
3. Samples the first output token from the last position's logits.
4. Stores `past_key_values` on the `Sequence` object for reuse during decoding.
5. Sets `seq.status = DECODING`.

**`generate(prompt, sampling_params)`:**
Wires prefill and decode into a complete generation loop:
1. Tokenizes the prompt and creates a `Sequence`.
2. Calls `prefill()` to get the first token.
3. Enters a decode loop calling `decode_step()` (which feeds only the last token + cached KV to the model).
4. Stops when EOS is generated or `max_tokens` is reached.
5. Decodes output token IDs back to text.

The prefill phase is compute-bound (processing many tokens in parallel via matrix multiplications), while each decode step is memory-bound (fetching large KV caches from GPU memory to attend over just one new token).

---

## Part 3: Scheduler — `nano_sglang/scheduler.py`

**What was implemented:** `_decode_running()`, `step()`, and `run_to_completion()` methods.

**`_decode_running(sampling_params)`:**
The core of continuous batching. Instead of decoding each sequence independently (wasting GPU cycles), all running sequences are batched into a single `decode_batch()` call. After receiving new tokens:
- Each token is appended to its sequence's output.
- Sequences hitting EOS or `max_tokens` are moved to the `finished` list.
- Only still-active sequences remain in `running`.

**`step(sampling_params)`:**
One scheduling iteration that:
1. Prefills one waiting request (if room in the batch), moving it to `running`.
2. Decodes all running sequences together in one batched forward pass.

This interleaving is the key idea from the Orca paper: new requests enter the batch as soon as there's capacity, rather than waiting for all current requests to finish (static batching).

**`run_to_completion(sampling_params)`:**
Repeatedly calls `step()` until both `waiting_queue` and `running` are empty. Returns results sorted by `seq_id` to preserve original request order.

The continuous batching approach keeps the GPU saturated: when a short request finishes, its slot is immediately filled by a waiting request. Static batching would leave that slot idle until the longest request in the batch completes.

---

## Part 4: Benchmark

**Script:** `benchmark.py` (run via `modal run modal_run.py::benchmark`)

The benchmark measures throughput (tokens/sec) for increasing numbers of concurrent requests (1, 2, 4, 8, 16) and compares two strategies:

1. **Sequential**: `Engine.generate()` called one request at a time in a loop.
2. **Batched**: `Scheduler.run_to_completion()` processing all requests with continuous batching.

**Expected results:**

With 1 request, both approaches are equivalent (the scheduler just calls `decode_step` directly). As concurrency increases, the batched scheduler should show significantly higher throughput because:

- Multiple decode steps share one GPU kernel launch (amortized overhead).
- The GPU's parallel compute units are better utilized when processing multiple sequences simultaneously.
- The memory bandwidth cost of loading model weights is paid once per batch, not once per sequence.

The speedup is most pronounced for decode-heavy workloads (long generation with short prompts), since prefill still happens one-at-a-time while decodes are fully batched.

---

## Part 5 (Stretch): Paged KV Cache — `nano_sglang/block_manager.py`

**What was implemented:** `allocate()` and `free()` methods on the `BlockManager` class.

This implements the core idea from the PagedAttention / vLLM paper: manage KV cache memory in fixed-size blocks (analogous to OS virtual memory pages) rather than contiguous allocations.

**`allocate(seq_id, num_tokens)`:**
1. Computes the number of blocks needed: `ceil(num_tokens / block_size)`.
2. Checks if enough free blocks are available (raises `RuntimeError` if not).
3. Pops block IDs from `free_blocks` and records the mapping in `seq_to_blocks`.

**`free(seq_id)`:**
1. Retrieves all block IDs for the sequence.
2. Zeros out the K/V data in each block across all layers.
3. Returns blocks to the `free_blocks` pool.

**Why paging matters:**
Without paging, each sequence pre-allocates a contiguous chunk for `max_seq_len` tokens, leading to:
- **Internal fragmentation**: most sequences don't use all `max_seq_len` tokens.
- **External fragmentation**: freed chunks leave gaps that new sequences may not fit into.

With paged allocation, blocks are allocated on demand and can be scattered across the pool. This eliminates external fragmentation and greatly reduces internal fragmentation (waste is at most `block_size - 1` tokens per sequence). The vLLM paper reports near-zero waste and up to 2–4× throughput improvement from paging alone.

---

## References

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/pdf/2309.06180) — Kwon et al., 2023
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/system/files/osdi22-yu.pdf) — Yu et al., OSDI 2022
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm) — Gordić
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and [nano-vllm walkthrough](https://neutree.ai/blog/nano-vllm-part-1)
