"""
Week 7: Test Case Generation for DeepseekV3MLP Block

The DeepseekV3MLP block implements a gated MLP:
    output = down_proj(act_fn(gate_proj(x)) * up_proj(x))

Where act_fn is SiLU (Sigmoid Linear Unit).

Architecture (from modeling_deepseek_v3.py):
    - gate_proj: Linear(hidden_size -> intermediate_size, no bias)
    - up_proj:   Linear(hidden_size -> intermediate_size, no bias)
    - down_proj:  Linear(intermediate_size -> hidden_size, no bias)
    - act_fn:     SiLU

We shrink dimensions for testing:
    - hidden_size: 4 (actual: 7168)
    - intermediate_size: 8 (actual: 18432)
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================
# Standalone MLP Block (copied from modeling_deepseek_v3.py)
# ============================================================
class DeepseekV3MLP(nn.Module):
    """
    Gated MLP with SiLU activation.
    Forward: down_proj(silu(gate_proj(x)) * up_proj(x))
    """
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def set_deterministic(seed):
    """Set all seeds and enable deterministic mode."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_test_case(hidden_size, intermediate_size, batch_size, seq_len,
                        weight_seed, input_seed, test_name):
    """
    Generate a single MLP test case.
    
    Uses DIFFERENT seeds for weights and inputs so that changing
    weight configuration doesn't affect input generation.
    """
    # --- Generate weights with weight_seed ---
    set_deterministic(weight_seed)
    model = DeepseekV3MLP(hidden_size, intermediate_size)
    # Re-initialize weights with controlled random values
    nn.init.normal_(model.gate_proj.weight, mean=0.0, std=0.02)
    nn.init.normal_(model.up_proj.weight, mean=0.0, std=0.02)
    nn.init.normal_(model.down_proj.weight, mean=0.0, std=0.02)
    model.eval()

    # --- Generate inputs with input_seed ---
    set_deterministic(input_seed)
    x = torch.randn(batch_size, seq_len, hidden_size)

    # --- Forward pass ---
    with torch.no_grad():
        output = model(x)

    # --- Save test case as text file ---
    test_case = {
        "test_name": test_name,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "weight_seed": weight_seed,
        "input_seed": input_seed,
        # Weights
        "gate_proj_weight": model.gate_proj.weight.data.numpy().tolist(),
        "up_proj_weight": model.up_proj.weight.data.numpy().tolist(),
        "down_proj_weight": model.down_proj.weight.data.numpy().tolist(),
        # Input
        "input": x.numpy().tolist(),
        # Expected output
        "expected_output": output.numpy().tolist(),
    }

    output_dir = os.path.join(os.path.dirname(__file__), "..", "test_cases")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{test_name}.json")
    
    with open(filepath, "w") as f:
        json.dump(test_case, f, indent=2)

    print(f"  Saved: {filepath}")
    print(f"    Input shape:  ({batch_size}, {seq_len}, {hidden_size})")
    print(f"    Output shape: ({batch_size}, {seq_len}, {hidden_size})")
    print(f"    Output sample: {output[0, 0, :min(4, hidden_size)].tolist()}")

    return output


def main():
    print("=" * 60)
    print("DeepseekV3MLP Test Case Generation")
    print("=" * 60)

    # Test Case 1: Small dimensions, single token
    print("\n--- Test Case 1: Small (hidden=4, inter=8, batch=1, seq=1) ---")
    out1 = generate_test_case(
        hidden_size=4, intermediate_size=8,
        batch_size=1, seq_len=1,
        weight_seed=100, input_seed=200,
        test_name="mlp_test_1"
    )

    # Test Case 2: Small dimensions, multiple tokens
    print("\n--- Test Case 2: Small (hidden=4, inter=8, batch=1, seq=3) ---")
    out2 = generate_test_case(
        hidden_size=4, intermediate_size=8,
        batch_size=1, seq_len=3,
        weight_seed=101, input_seed=201,
        test_name="mlp_test_2"
    )

    # Test Case 3: Slightly larger dimensions
    print("\n--- Test Case 3: Medium (hidden=8, inter=16, batch=2, seq=2) ---")
    out3 = generate_test_case(
        hidden_size=8, intermediate_size=16,
        batch_size=2, seq_len=2,
        weight_seed=102, input_seed=202,
        test_name="mlp_test_3"
    )

    print("\n" + "=" * 60)
    print("All MLP test cases generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
