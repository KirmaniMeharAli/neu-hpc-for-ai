"""
Week 7: Test Case Generation for DeepseekV3TopkRouter Block

The TopkRouter computes router logits for expert selection:
    router_logits = linear(hidden_states_float32, weight_float32)

Architecture (from modeling_deepseek_v3.py):
    - weight: Parameter of shape (n_routed_experts, hidden_size)
    - e_score_correction_bias: Buffer of shape (n_routed_experts,), initialized to zeros
    - Forward: F.linear(hidden_states.float(), weight.float()) -> router_logits

Note: The router casts BOTH inputs and weights to float32 before computing logits.
      It does NOT apply softmax or sigmoid - that happens in DeepseekV3MoE.route_tokens_to_experts().

We shrink dimensions for testing:
    - hidden_size: 4 (actual: 7168)
    - n_routed_experts: 8 (actual: 256)
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# Standalone Router Block (copied from modeling_deepseek_v3.py)
# ============================================================
class DeepseekV3TopkRouter(nn.Module):
    """
    Router that computes logits for each expert.
    Forward: F.linear(hidden_states.float32, weight.float32)
    """
    def __init__(self, hidden_size, n_routed_experts):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.weight = nn.Parameter(torch.empty(n_routed_experts, hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(n_routed_experts))

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32),
                                  self.weight.type(torch.float32))
        return router_logits


def set_deterministic(seed):
    """Set all seeds and enable deterministic mode."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_test_case(hidden_size, n_routed_experts, batch_size, seq_len,
                        weight_seed, input_seed, test_name):
    """Generate a single Router test case."""

    # --- Generate weights with weight_seed ---
    set_deterministic(weight_seed)
    model = DeepseekV3TopkRouter(hidden_size, n_routed_experts)
    nn.init.normal_(model.weight, mean=0.0, std=0.02)
    model.eval()

    # --- Generate inputs with input_seed ---
    set_deterministic(input_seed)
    x = torch.randn(batch_size, seq_len, hidden_size)

    # --- Forward pass ---
    with torch.no_grad():
        output = model(x)

    # --- Save test case ---
    test_case = {
        "test_name": test_name,
        "hidden_size": hidden_size,
        "n_routed_experts": n_routed_experts,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "weight_seed": weight_seed,
        "input_seed": input_seed,
        # Weights
        "router_weight": model.weight.data.numpy().tolist(),
        "e_score_correction_bias": model.e_score_correction_bias.numpy().tolist(),
        # Input (note: router flattens to 2D internally)
        "input": x.numpy().tolist(),
        # Expected output: shape (batch_size * seq_len, n_routed_experts)
        "expected_output": output.numpy().tolist(),
    }

    output_dir = os.path.join(os.path.dirname(__file__), "..", "test_cases")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{test_name}.json")

    with open(filepath, "w") as f:
        json.dump(test_case, f, indent=2)

    print(f"  Saved: {filepath}")
    print(f"    Input shape:  ({batch_size}, {seq_len}, {hidden_size})")
    print(f"    Output shape: ({batch_size * seq_len}, {n_routed_experts})")
    print(f"    Output sample: {output[0, :min(4, n_routed_experts)].tolist()}")

    return output


def main():
    print("=" * 60)
    print("DeepseekV3TopkRouter Test Case Generation")
    print("=" * 60)

    # Test Case 1: Small
    print("\n--- Test Case 1: (hidden=4, experts=8, batch=1, seq=1) ---")
    generate_test_case(
        hidden_size=4, n_routed_experts=8,
        batch_size=1, seq_len=1,
        weight_seed=300, input_seed=400,
        test_name="router_test_1"
    )

    # Test Case 2: Multiple tokens
    print("\n--- Test Case 2: (hidden=4, experts=8, batch=1, seq=3) ---")
    generate_test_case(
        hidden_size=4, n_routed_experts=8,
        batch_size=1, seq_len=3,
        weight_seed=301, input_seed=401,
        test_name="router_test_2"
    )

    # Test Case 3: Larger
    print("\n--- Test Case 3: (hidden=8, experts=16, batch=2, seq=2) ---")
    generate_test_case(
        hidden_size=8, n_routed_experts=16,
        batch_size=2, seq_len=2,
        weight_seed=302, input_seed=402,
        test_name="router_test_3"
    )

    print("\n" + "=" * 60)
    print("All Router test cases generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
