/*
 * moe_operator.h - DeepSeekV3 MoE Operator Implementation
 *
 * Week 7: Communication - Pure C Implementation
 *
 * Implements all blocks of the DeepSeekV3 MoE operator:
 *   1. DeepseekV3MLP       - Gated MLP with SiLU
 *   2. DeepseekV3TopkRouter - Router linear projection
 *   3. DeepseekV3NaiveMoe  - Expert dispatch
 *   4. DeepseekV3MoE       - Full MoE (router + routing + experts + shared)
 */

#ifndef MOE_OPERATOR_H
#define MOE_OPERATOR_H

/* ============================================================
 * Block 1: DeepseekV3MLP
 * ============================================================
 * Gated MLP: output = down_proj(silu(gate_proj(x)) * up_proj(x))
 *
 * Parameters:
 *   gate_proj_w: (intermediate_size x hidden_size)
 *   up_proj_w:   (intermediate_size x hidden_size)
 *   down_proj_w: (hidden_size x intermediate_size)
 *   input:       (num_tokens x hidden_size)
 *   output:      (num_tokens x hidden_size)
 */
void deepseek_mlp_forward(
    const float *input,          /* (num_tokens, hidden_size) */
    const float *gate_proj_w,    /* (intermediate_size, hidden_size) */
    const float *up_proj_w,      /* (intermediate_size, hidden_size) */
    const float *down_proj_w,    /* (hidden_size, intermediate_size) */
    float *output,               /* (num_tokens, hidden_size) */
    int num_tokens,
    int hidden_size,
    int intermediate_size
);


/* ============================================================
 * Block 2: DeepseekV3TopkRouter
 * ============================================================
 * Router: logits = F.linear(hidden_states.float32, weight.float32)
 *
 * Note: Both input and weight are cast to float32 before computation.
 *       We're already in float32 in C, so this is a simple matmul_bt.
 */
void deepseek_router_forward(
    const float *input,          /* (num_tokens, hidden_size) */
    const float *router_weight,  /* (n_routed_experts, hidden_size) */
    float *router_logits,        /* (num_tokens, n_routed_experts) */
    int num_tokens,
    int hidden_size,
    int n_routed_experts
);


/* ============================================================
 * Block 3: DeepseekV3NaiveMoe
 * ============================================================
 * Expert dispatch: For each expert, find assigned tokens, compute
 * gate-up MLP, multiply by routing weight, accumulate.
 */
void deepseek_naive_moe_forward(
    const float *hidden_states,  /* (num_tokens, hidden_dim) */
    const int *top_k_index,      /* (num_tokens, top_k) */
    const float *top_k_weights,  /* (num_tokens, top_k) */
    const float *gate_up_proj,   /* (num_experts, 2*intermediate_dim, hidden_dim) */
    const float *down_proj,      /* (num_experts, hidden_dim, intermediate_dim) */
    float *output,               /* (num_tokens, hidden_dim) */
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int num_experts,
    int top_k
);


/* ============================================================
 * Block 4: Token Routing Logic
 * ============================================================
 * Implements route_tokens_to_experts():
 *   1. sigmoid on router logits
 *   2. Add e_score_correction_bias
 *   3. Group scoring (top-2 per group, sum)
 *   4. Select topk_group groups
 *   5. Mask non-selected groups
 *   6. Top-k expert selection
 *   7. Gather weights from original sigmoid scores
 *   8. Normalize if needed
 *   9. Scale by routed_scaling_factor
 */
void deepseek_route_tokens(
    const float *router_logits,          /* (num_tokens, n_routed_experts) */
    const float *e_score_correction_bias, /* (n_routed_experts,) */
    int *topk_indices,                    /* output: (num_tokens, top_k) */
    float *topk_weights,                  /* output: (num_tokens, top_k) */
    int num_tokens,
    int n_routed_experts,
    int n_group,
    int topk_group,
    int top_k,
    int norm_topk_prob,
    float routed_scaling_factor
);


/* ============================================================
 * Block 5: Full DeepseekV3MoE Forward
 * ============================================================
 * Combines all blocks:
 *   1. Router forward
 *   2. Route tokens to experts
 *   3. Expert dispatch (NaiveMoe)
 *   4. Shared expert MLP
 *   5. output = expert_output + shared_expert_output
 */
void deepseek_moe_forward(
    const float *input,                   /* (num_tokens, hidden_size) */
    /* Router weights */
    const float *router_weight,           /* (n_routed_experts, hidden_size) */
    const float *e_score_correction_bias, /* (n_routed_experts,) */
    /* Expert weights */
    const float *expert_gate_up_proj,     /* (n_experts, 2*moe_inter, hidden) */
    const float *expert_down_proj,        /* (n_experts, hidden, moe_inter) */
    /* Shared expert weights */
    const float *shared_gate_proj_w,      /* (shared_inter, hidden) */
    const float *shared_up_proj_w,        /* (shared_inter, hidden) */
    const float *shared_down_proj_w,      /* (hidden, shared_inter) */
    /* Output */
    float *output,                        /* (num_tokens, hidden_size) */
    /* Also output intermediates for debugging */
    float *out_router_logits,             /* (num_tokens, n_routed_experts) or NULL */
    int *out_topk_indices,                /* (num_tokens, top_k) or NULL */
    float *out_topk_weights,              /* (num_tokens, top_k) or NULL */
    /* Config */
    int num_tokens,
    int hidden_size,
    int moe_intermediate_size,
    int n_routed_experts,
    int n_shared_experts,
    int n_group,
    int topk_group,
    int top_k,
    int norm_topk_prob,
    float routed_scaling_factor
);

#endif /* MOE_OPERATOR_H */
