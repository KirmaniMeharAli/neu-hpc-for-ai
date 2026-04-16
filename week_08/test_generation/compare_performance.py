"""
Week 8: Performance Comparison - CUDA+NCCL Implementation vs HuggingFace Transformers

Compares inference throughput of our multi-GPU CUDA+NCCL MoE implementation
against the HuggingFace Transformers DeepSeekV3 MoE reference implementation.

Metrics:
  - Latency (ms) per forward pass
  - Throughput (tokens/sec)
  - Scaling efficiency across GPUs

Usage:
  python compare_performance.py
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


# ============================================================
# Transformers reference MoE (same as test generation)
# ============================================================
class DeepseekV3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepseekV3TopkRouter(nn.Module):
    def __init__(self, hidden_size, n_routed_experts):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.weight = nn.Parameter(torch.empty(n_routed_experts, hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(n_routed_experts))

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        return router_logits


class DeepseekV3NaiveMoe(nn.Module):
    def __init__(self, num_experts, hidden_dim, intermediate_dim):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_dim, hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_dim, intermediate_dim)
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = F.silu(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states *= top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


class DeepseekV3MoE(nn.Module):
    def __init__(self, hidden_size, moe_intermediate_size, n_routed_experts,
                 n_shared_experts, n_group, topk_group, top_k,
                 norm_topk_prob, routed_scaling_factor):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor

        self.gate = DeepseekV3TopkRouter(hidden_size, n_routed_experts)
        self.experts = DeepseekV3NaiveMoe(n_routed_experts, hidden_size, moe_intermediate_size)
        self.shared_experts = DeepseekV3MLP(hidden_size, moe_intermediate_size * n_shared_experts)

    def route_tokens_to_experts(self, router_logits):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice
            .view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0].sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


# ============================================================
# Benchmark configurations
# ============================================================
@dataclass
class BenchConfig:
    name: str
    num_tokens: int
    n_routed_experts: int
    hidden_size: int
    moe_intermediate_size: int
    top_k: int
    n_group: int
    topk_group: int


BENCH_CONFIGS = [
    BenchConfig("Small",    num_tokens=64,  n_routed_experts=8,   hidden_size=64,  moe_intermediate_size=128, top_k=2, n_group=2, topk_group=1),
    BenchConfig("Medium",   num_tokens=256, n_routed_experts=16,  hidden_size=128, moe_intermediate_size=256, top_k=4, n_group=4, topk_group=2),
    BenchConfig("Large",    num_tokens=512, n_routed_experts=32,  hidden_size=256, moe_intermediate_size=512, top_k=4, n_group=4, topk_group=2),
    BenchConfig("XLarge",   num_tokens=1024, n_routed_experts=64, hidden_size=512, moe_intermediate_size=1024, top_k=8, n_group=8, topk_group=4),
]


def benchmark_pytorch(config: BenchConfig, device: str = "cpu", num_iters: int = 10) -> float:
    """Benchmark PyTorch Transformers reference implementation."""
    torch.manual_seed(42)

    model = DeepseekV3MoE(
        hidden_size=config.hidden_size,
        moe_intermediate_size=config.moe_intermediate_size,
        n_routed_experts=config.n_routed_experts,
        n_shared_experts=1,
        n_group=config.n_group,
        topk_group=config.topk_group,
        top_k=config.top_k,
        norm_topk_prob=True,
        routed_scaling_factor=2.0
    ).to(device)
    model.eval()

    x = torch.randn(1, config.num_tokens, config.hidden_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.time() - start) / num_iters * 1000  # ms

    return elapsed


def main():
    print("=" * 80)
    print("Week 8: Performance Comparison")
    print("PyTorch Transformers Reference vs CUDA+NCCL Implementation")
    print("=" * 80)

    # Determine available device
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    if has_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("No GPU available, running CPU-only benchmarks")

    print()
    print(f"{'Config':<10} {'Tokens':<8} {'Experts':<10} {'Hidden':<8} "
          f"{'TopK':<6} {'PyTorch-CPU (ms)':<18} ", end="")
    if has_cuda:
        print(f"{'PyTorch-GPU (ms)':<18}", end="")
    print()
    print("-" * 80)

    results = []

    for cfg in BENCH_CONFIGS:
        # CPU benchmark
        cpu_ms = benchmark_pytorch(cfg, device="cpu", num_iters=5)

        row = {
            "config": cfg.name,
            "num_tokens": cfg.num_tokens,
            "n_experts": cfg.n_routed_experts,
            "hidden_size": cfg.hidden_size,
            "top_k": cfg.top_k,
            "pytorch_cpu_ms": cpu_ms,
        }

        print(f"{cfg.name:<10} {cfg.num_tokens:<8} {cfg.n_routed_experts:<10} "
              f"{cfg.hidden_size:<8} {cfg.top_k:<6} {cpu_ms:<18.3f} ", end="")

        if has_cuda:
            gpu_ms = benchmark_pytorch(cfg, device="cuda", num_iters=10)
            row["pytorch_gpu_ms"] = gpu_ms
            print(f"{gpu_ms:<18.3f}", end="")

        print()
        results.append(row)

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), "..", "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: {results_path}")
    print()
    print("Note: To compare with our CUDA+NCCL implementation,")
    print("run: ./moe_parallel_test ./test_cases [num_gpus]")
    print("The 'Part C: Performance Benchmark' section shows CUDA timing.")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
