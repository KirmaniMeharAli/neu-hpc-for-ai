"""
Week 7: Test Case Generation for DeepseekV3MoE Block (Full MoE)

The full MoE block combines:
    1. TopkRouter  - compute router logits
    2. route_tokens_to_experts() - sigmoid + bias correction + group scoring + top-k selection
    3. NaiveMoe    - dispatch tokens to selected experts
    4. SharedExperts (MLP) - applied to ALL tokens
    5. Residual addition: output = expert_output + shared_expert_output

The routing logic (route_tokens_to_experts) does:
    a. Apply sigmoid to router logits
    b. Add e_score_correction_bias
    c. Group experts into n_group groups
    d. Score each group by top-2-in-group sum
    e. Select topk_group groups
    f. Mask out non-selected groups
    g. Pick top_k experts from remaining
    h. Gather weights from sigmoid scores (NOT corrected)
    i. Normalize weights if norm_topk_prob
    j. Scale by routed_scaling_factor

We shrink dimensions:
    - hidden_size: 4 (actual: 7168)
    - moe_intermediate_size: 8 (actual: 2048)
    - n_routed_experts: 8 (actual: 256)
    - n_shared_experts: 1 (actual: 1)
    - n_group: 2 (actual: 8)
    - topk_group: 1 (actual: 4)
    - num_experts_per_tok: 2 (actual: 8)
    - routed_scaling_factor: 2.0 (actual: 2.5)
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# Standalone MLP (shared experts use this)
# ============================================================
class DeepseekV3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================
# Standalone Router
# ============================================================
class DeepseekV3TopkRouter(nn.Module):
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


# ============================================================
# Standalone NaiveMoe
# ============================================================
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
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


# ============================================================
# Full MoE Block
# ============================================================
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
        self.shared_experts = DeepseekV3MLP(
            hidden_size, moe_intermediate_size * n_shared_experts
        )

    def route_tokens_to_experts(self, router_logits):
        """Token-to-expert routing with group-level selection."""
        # Step 1: Sigmoid (NOT softmax)
        router_logits = router_logits.sigmoid()

        # Step 2: Add bias correction
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias

        # Step 3: Group scoring - reshape into groups, take top-2 per group, sum
        group_scores = (
            router_logits_for_choice
            .view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )

        # Step 4: Select top groups
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        # Step 5: Create score mask from group selection
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)

        # Step 6: Select top-k experts
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]

        # Step 7: Gather weights from ORIGINAL sigmoid scores (not bias-corrected)
        topk_weights = router_logits.gather(1, topk_indices)

        # Step 8: Normalize if configured
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator

        # Step 9: Scale
        topk_weights = topk_weights * self.routed_scaling_factor

        return topk_indices, topk_weights

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape

        # Router
        router_logits = self.gate(hidden_states)

        # Route tokens
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)

        # Expert dispatch
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)

        # Add shared expert output
        hidden_states = hidden_states + self.shared_experts(residuals)

        return hidden_states, router_logits, topk_indices, topk_weights


def set_deterministic(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_test_case(hidden_size, moe_intermediate_size, n_routed_experts,
                        n_shared_experts, n_group, topk_group, top_k,
                        norm_topk_prob, routed_scaling_factor,
                        batch_size, seq_len,
                        weight_seed, input_seed, test_name):
    """Generate a single full MoE test case."""

    # --- Generate weights ---
    set_deterministic(weight_seed)
    model = DeepseekV3MoE(
        hidden_size, moe_intermediate_size, n_routed_experts,
        n_shared_experts, n_group, topk_group, top_k,
        norm_topk_prob, routed_scaling_factor
    )
    # Initialize all weights
    nn.init.normal_(model.gate.weight, mean=0.0, std=0.02)
    nn.init.normal_(model.experts.gate_up_proj, mean=0.0, std=0.02)
    nn.init.normal_(model.experts.down_proj, mean=0.0, std=0.02)
    nn.init.normal_(model.shared_experts.gate_proj.weight, mean=0.0, std=0.02)
    nn.init.normal_(model.shared_experts.up_proj.weight, mean=0.0, std=0.02)
    nn.init.normal_(model.shared_experts.down_proj.weight, mean=0.0, std=0.02)
    model.eval()

    # --- Generate inputs ---
    set_deterministic(input_seed)
    x = torch.randn(batch_size, seq_len, hidden_size)

    # --- Forward pass ---
    with torch.no_grad():
        output, router_logits, topk_indices, topk_weights = model(x)

    # --- Save test case ---
    test_case = {
        "test_name": test_name,
        "hidden_size": hidden_size,
        "moe_intermediate_size": moe_intermediate_size,
        "n_routed_experts": n_routed_experts,
        "n_shared_experts": n_shared_experts,
        "n_group": n_group,
        "topk_group": topk_group,
        "top_k": top_k,
        "norm_topk_prob": norm_topk_prob,
        "routed_scaling_factor": routed_scaling_factor,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "weight_seed": weight_seed,
        "input_seed": input_seed,
        # Router weights
        "router_weight": model.gate.weight.data.numpy().tolist(),
        "e_score_correction_bias": model.gate.e_score_correction_bias.numpy().tolist(),
        # Expert weights
        "expert_gate_up_proj": model.experts.gate_up_proj.data.numpy().tolist(),
        "expert_down_proj": model.experts.down_proj.data.numpy().tolist(),
        # Shared expert weights
        "shared_gate_proj_weight": model.shared_experts.gate_proj.weight.data.numpy().tolist(),
        "shared_up_proj_weight": model.shared_experts.up_proj.weight.data.numpy().tolist(),
        "shared_down_proj_weight": model.shared_experts.down_proj.weight.data.numpy().tolist(),
        # Input
        "input": x.numpy().tolist(),
        # Intermediate outputs (for debugging)
        "router_logits": router_logits.numpy().tolist(),
        "topk_indices": topk_indices.numpy().tolist(),
        "topk_weights": topk_weights.numpy().tolist(),
        # Expected final output
        "expected_output": output.numpy().tolist(),
    }

    output_dir = os.path.join(os.path.dirname(__file__), "..", "test_cases")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{test_name}.json")

    with open(filepath, "w") as f:
        json.dump(test_case, f, indent=2)

    print(f"  Saved: {filepath}")
    print(f"    Input shape:   ({batch_size}, {seq_len}, {hidden_size})")
    print(f"    Output shape:  ({batch_size}, {seq_len}, {hidden_size})")
    print(f"    Router logits: ({batch_size * seq_len}, {n_routed_experts})")
    print(f"    TopK indices:  {topk_indices.numpy().tolist()}")
    print(f"    TopK weights:  {topk_weights.numpy().tolist()}")
    print(f"    Output sample: {output[0, 0, :min(4, hidden_size)].tolist()}")

    return output


def main():
    print("=" * 60)
    print("DeepseekV3MoE (Full) Test Case Generation")
    print("=" * 60)

    # Test Case 1: Minimal MoE
    print("\n--- Test Case 1: Minimal MoE ---")
    generate_test_case(
        hidden_size=4, moe_intermediate_size=8,
        n_routed_experts=8, n_shared_experts=1,
        n_group=2, topk_group=1, top_k=2,
        norm_topk_prob=True, routed_scaling_factor=2.0,
        batch_size=1, seq_len=1,
        weight_seed=700, input_seed=800,
        test_name="moe_full_test_1"
    )

    # Test Case 2: Multiple tokens
    print("\n--- Test Case 2: Multiple tokens ---")
    generate_test_case(
        hidden_size=4, moe_intermediate_size=8,
        n_routed_experts=8, n_shared_experts=1,
        n_group=2, topk_group=1, top_k=2,
        norm_topk_prob=True, routed_scaling_factor=2.0,
        batch_size=1, seq_len=3,
        weight_seed=701, input_seed=801,
        test_name="moe_full_test_2"
    )

    # Test Case 3: No normalization
    print("\n--- Test Case 3: Without normalization ---")
    generate_test_case(
        hidden_size=4, moe_intermediate_size=8,
        n_routed_experts=8, n_shared_experts=1,
        n_group=2, topk_group=1, top_k=2,
        norm_topk_prob=False, routed_scaling_factor=2.0,
        batch_size=1, seq_len=2,
        weight_seed=702, input_seed=802,
        test_name="moe_full_test_3"
    )

    print("\n" + "=" * 60)
    print("All full MoE test cases generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
