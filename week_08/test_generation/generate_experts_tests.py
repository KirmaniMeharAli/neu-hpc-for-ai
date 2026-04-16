"""
Week 7: Test Case Generation for DeepseekV3NaiveMoe Block

The NaiveMoe block holds all expert weights as 3D tensors and dispatches
tokens to selected experts based on router-provided indices and weights.

Architecture (from modeling_deepseek_v3.py):
    - gate_up_proj: Parameter(num_experts, 2 * intermediate_dim, hidden_dim)
    - down_proj:    Parameter(num_experts, hidden_dim, intermediate_dim)
    - act_fn:       SiLU

Forward pass:
    1. Create one-hot expert mask from top_k_index
    2. For each active expert:
       a. Find which tokens are assigned to this expert
       b. gate, up = linear(token_states, gate_up_proj[expert]).chunk(2)
       c. current = act_fn(gate) * up
       d. current = linear(current, down_proj[expert])
       e. current *= top_k_weights for this token-expert pair
       f. Accumulate into output

We shrink dimensions for testing:
    - hidden_size: 4 (actual: 7168)
    - moe_intermediate_size: 8 (actual: 2048)
    - num_experts: 4 (actual: 256)
    - top_k: 2 (actual: 8)
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# Standalone NaiveMoe Block (copied from modeling_deepseek_v3.py)
# ============================================================
class DeepseekV3NaiveMoe(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

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
        """
        Args:
            hidden_states: (num_tokens, hidden_dim)
            top_k_index:   (num_tokens, top_k) - expert indices per token
            top_k_weights: (num_tokens, top_k) - gating weights per token
        Returns:
            output: (num_tokens, hidden_dim)
        """
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # (num_experts, top_k, num_tokens)
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

            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


def set_deterministic(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_test_case(num_experts, hidden_dim, intermediate_dim,
                        num_tokens, top_k,
                        weight_seed, input_seed, test_name):
    """Generate a single NaiveMoe test case."""

    # --- Generate weights ---
    set_deterministic(weight_seed)
    model = DeepseekV3NaiveMoe(num_experts, hidden_dim, intermediate_dim)
    nn.init.normal_(model.gate_up_proj, mean=0.0, std=0.02)
    nn.init.normal_(model.down_proj, mean=0.0, std=0.02)
    model.eval()

    # --- Generate inputs ---
    set_deterministic(input_seed)
    hidden_states = torch.randn(num_tokens, hidden_dim)

    # Generate deterministic expert assignments
    # Each token picks top_k experts from [0, num_experts)
    top_k_index = torch.zeros(num_tokens, top_k, dtype=torch.long)
    for i in range(num_tokens):
        perm = torch.randperm(num_experts)[:top_k]
        top_k_index[i] = perm

    # Generate gating weights (positive, sum-to-1 per token)
    raw_weights = torch.rand(num_tokens, top_k)
    top_k_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)

    # --- Forward pass ---
    with torch.no_grad():
        output = model(hidden_states, top_k_index, top_k_weights)

    # --- Save test case ---
    test_case = {
        "test_name": test_name,
        "num_experts": num_experts,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "num_tokens": num_tokens,
        "top_k": top_k,
        "weight_seed": weight_seed,
        "input_seed": input_seed,
        # Weights: gate_up_proj is (num_experts, 2*intermediate_dim, hidden_dim)
        "gate_up_proj": model.gate_up_proj.data.numpy().tolist(),
        # down_proj is (num_experts, hidden_dim, intermediate_dim)
        "down_proj": model.down_proj.data.numpy().tolist(),
        # Inputs
        "hidden_states": hidden_states.numpy().tolist(),
        "top_k_index": top_k_index.numpy().tolist(),
        "top_k_weights": top_k_weights.numpy().tolist(),
        # Expected output
        "expected_output": output.numpy().tolist(),
    }

    output_dir = os.path.join(os.path.dirname(__file__), "..", "test_cases")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{test_name}.json")

    with open(filepath, "w") as f:
        json.dump(test_case, f, indent=2)

    print(f"  Saved: {filepath}")
    print(f"    Hidden states: ({num_tokens}, {hidden_dim})")
    print(f"    Top-k indices: ({num_tokens}, {top_k})")
    print(f"    Output shape:  ({num_tokens}, {hidden_dim})")
    print(f"    Output sample: {output[0, :min(4, hidden_dim)].tolist()}")

    return output


def main():
    print("=" * 60)
    print("DeepseekV3NaiveMoe Test Case Generation")
    print("=" * 60)

    # Test Case 1: Minimal
    print("\n--- Test Case 1: (experts=4, hidden=4, inter=8, tokens=2, topk=2) ---")
    generate_test_case(
        num_experts=4, hidden_dim=4, intermediate_dim=8,
        num_tokens=2, top_k=2,
        weight_seed=500, input_seed=600,
        test_name="experts_test_1"
    )

    # Test Case 2: More tokens
    print("\n--- Test Case 2: (experts=4, hidden=4, inter=8, tokens=4, topk=2) ---")
    generate_test_case(
        num_experts=4, hidden_dim=4, intermediate_dim=8,
        num_tokens=4, top_k=2,
        weight_seed=501, input_seed=601,
        test_name="experts_test_2"
    )

    # Test Case 3: Larger
    print("\n--- Test Case 3: (experts=8, hidden=8, inter=16, tokens=3, topk=2) ---")
    generate_test_case(
        num_experts=8, hidden_dim=8, intermediate_dim=16,
        num_tokens=3, top_k=2,
        weight_seed=502, input_seed=602,
        test_name="experts_test_3"
    )

    print("\n" + "=" * 60)
    print("All NaiveMoe test cases generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
