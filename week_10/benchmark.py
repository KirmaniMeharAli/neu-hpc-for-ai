"""Part 4: Benchmark — throughput (tokens/sec) vs. number of concurrent requests.

Compares:
  1. Batched scheduler (continuous batching via Scheduler.run_to_completion)
  2. Sequential generation (Engine.generate called one request at a time)

Usage (on Modal):
    modal run modal_run.py::benchmark
"""

import time
import torch


MODEL_NAME = "Qwen/Qwen3-0.6B"


def run_benchmark():
    from nano_sglang.engine import Engine
    from nano_sglang.scheduler import Scheduler
    from nano_sglang.sampling import SamplingParams

    params = SamplingParams(temperature=0, max_tokens=50)

    prompts = [
        "The capital of France is",
        "Machine learning is",
        "The weather today is",
        "Python programming language",
        "Once upon a time there was",
        "The history of computing began",
        "Deep learning models are",
        "Artificial intelligence will",
        "The stock market today",
        "In the year 2050 we will",
        "The fastest animal on earth",
        "Quantum computing promises to",
        "The ocean covers most of",
        "Space exploration has led to",
        "Renewable energy sources include",
        "The human brain contains about",
    ]

    # Test with increasing numbers of concurrent requests
    request_counts = [1, 2, 4, 8, 16]

    print("=" * 70)
    print("BENCHMARK: Batched Scheduler vs. Sequential Generation")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Max tokens per request: {params.max_tokens}")
    print(f"Temperature: {params.temperature}")
    print()

    # ---- Sequential baseline (one at a time) ----
    print("-" * 70)
    print("Sequential Generation (Engine.generate, one at a time)")
    print("-" * 70)

    engine = Engine(MODEL_NAME)
    seq_results = []

    for n in request_counts:
        batch_prompts = prompts[:n]

        # Warmup for the first run
        if n == request_counts[0]:
            _ = engine.generate(batch_prompts[0], params)

        torch.cuda.synchronize()
        start = time.time()
        outputs = []
        for p in batch_prompts:
            out = engine.generate(p, params)
            outputs.append(out)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        total_tokens = sum(len(engine.tokenizer.encode(o)) for o in outputs)
        throughput = total_tokens / elapsed
        seq_results.append((n, elapsed, total_tokens, throughput))
        print(f"  {n:2d} requests | {elapsed:6.2f}s | {total_tokens:4d} tokens | {throughput:7.1f} tok/s")

    del engine
    torch.cuda.empty_cache()

    # ---- Batched scheduler ----
    print()
    print("-" * 70)
    print("Batched Scheduler (continuous batching via decode_batch)")
    print("-" * 70)

    batch_results = []

    for n in request_counts:
        batch_prompts = prompts[:n]
        scheduler = Scheduler(MODEL_NAME)

        for p in batch_prompts:
            scheduler.add_request(p)

        torch.cuda.synchronize()
        start = time.time()
        outputs = scheduler.run_to_completion(params)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        total_tokens = sum(len(scheduler.tokenizer.encode(o)) for o in outputs)
        throughput = total_tokens / elapsed
        batch_results.append((n, elapsed, total_tokens, throughput))
        print(f"  {n:2d} requests | {elapsed:6.2f}s | {total_tokens:4d} tokens | {throughput:7.1f} tok/s")

        del scheduler
        torch.cuda.empty_cache()

    # ---- Summary comparison ----
    print()
    print("=" * 70)
    print("SPEEDUP SUMMARY (Batched / Sequential)")
    print("=" * 70)
    print(f"{'Requests':>10} | {'Seq tok/s':>10} | {'Batch tok/s':>11} | {'Speedup':>8}")
    print("-" * 50)
    for (n, _, _, seq_tp), (_, _, _, bat_tp) in zip(seq_results, batch_results):
        speedup = bat_tp / seq_tp if seq_tp > 0 else float('inf')
        print(f"{n:>10} | {seq_tp:>10.1f} | {bat_tp:>11.1f} | {speedup:>7.2f}x")


if __name__ == "__main__":
    run_benchmark()
