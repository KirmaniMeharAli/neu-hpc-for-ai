/*
 * moe_operator.c - DeepSeekV3 MoE Operator Implementation
 *
 * Week 7: Communication - Pure C Implementation
 * No CUDA, no parallelism, no pthreads. Standard C library only.
 *
 * Reference: HuggingFace transformers/models/deepseek_v3/modeling_deepseek_v3.py
 * Paper: DeepSeekMoE: Towards Ultimate Expert Specialization (arXiv:2401.06066)
 */

#include "moe_operator.h"
#include "matrix_ops.h"
#include "activations.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>

/* ============================================================
 * Block 1: DeepseekV3MLP
 * ============================================================
 *
 * Implements the gated MLP used both as individual experts and
 * as shared experts in DeepSeekV3.
 *
 * PyTorch equivalent:
 *   gate = gate_proj(x)          # Linear(hidden -> intermediate, no bias)
 *   up   = up_proj(x)            # Linear(hidden -> intermediate, no bias)
 *   out  = down_proj(silu(gate) * up)  # Linear(intermediate -> hidden, no bias)
 *
 * F.linear(x, W) computes x @ W^T, so we use matmul_bt.
 */
void deepseek_mlp_forward(
    const float *input,
    const float *gate_proj_w,
    const float *up_proj_w,
    const float *down_proj_w,
    float *output,
    int num_tokens,
    int hidden_size,
    int intermediate_size)
{
    /* Allocate temporaries */
    float *gate = alloc_floats(num_tokens * intermediate_size);
    float *up   = alloc_floats(num_tokens * intermediate_size);

    /* gate = input @ gate_proj_w^T */
    matmul_bt(input, gate_proj_w, gate, num_tokens, hidden_size, intermediate_size);

    /* up = input @ up_proj_w^T */
    matmul_bt(input, up_proj_w, up, num_tokens, hidden_size, intermediate_size);

    /* gate = silu(gate) */
    silu_inplace(gate, num_tokens * intermediate_size);

    /* gate = gate * up (element-wise) */
    elementwise_mul(gate, up, gate, num_tokens * intermediate_size);

    /* output = gate @ down_proj_w^T */
    matmul_bt(gate, down_proj_w, output, num_tokens, intermediate_size, hidden_size);

    free(gate);
    free(up);
}


/* ============================================================
 * Block 2: DeepseekV3TopkRouter
 * ============================================================
 *
 * Computes router logits via linear projection.
 *
 * PyTorch equivalent:
 *   hidden_states = hidden_states.view(-1, hidden_size)
 *   router_logits = F.linear(hidden_states.float(), weight.float())
 *
 * Since F.linear(x, W) = x @ W^T, this is matmul_bt.
 */
void deepseek_router_forward(
    const float *input,
    const float *router_weight,
    float *router_logits,
    int num_tokens,
    int hidden_size,
    int n_routed_experts)
{
    matmul_bt(input, router_weight, router_logits,
              num_tokens, hidden_size, n_routed_experts);
}


/* ============================================================
 * Helper: Find top-k values and indices in an array
 * ============================================================
 * Simple O(n*k) selection - fine for small k.
 */
static void topk(const float *arr, int n, int k, float *values, int *indices) {
    /* Track which indices are already selected */
    int *used = (int *)calloc(n, sizeof(int));

    for (int i = 0; i < k; i++) {
        float best_val = -FLT_MAX;
        int best_idx = 0;
        for (int j = 0; j < n; j++) {
            if (!used[j] && arr[j] > best_val) {
                best_val = arr[j];
                best_idx = j;
            }
        }
        values[i] = best_val;
        indices[i] = best_idx;
        used[best_idx] = 1;
    }

    free(used);
}


/* ============================================================
 * Block 3: Token Routing Logic (route_tokens_to_experts)
 * ============================================================
 *
 * This is the core routing algorithm from DeepSeekV3MoE.
 *
 * Steps (matching PyTorch exactly):
 *   1. router_logits = sigmoid(router_logits)
 *   2. router_logits_for_choice = router_logits + e_score_correction_bias
 *   3. Group experts into n_group groups of size (n_experts / n_group)
 *   4. For each group, take top-2 scores and sum them -> group_score
 *   5. Select topk_group groups with highest group_scores
 *   6. Mask out experts in non-selected groups
 *   7. From remaining, select top_k experts per token
 *   8. Gather weights from ORIGINAL sigmoid scores (not bias-corrected)
 *   9. Normalize weights if norm_topk_prob
 *  10. Scale by routed_scaling_factor
 */
void deepseek_route_tokens(
    const float *router_logits_in,
    const float *e_score_correction_bias,
    int *topk_indices_out,
    float *topk_weights_out,
    int num_tokens,
    int n_routed_experts,
    int n_group,
    int topk_group,
    int top_k,
    int norm_topk_prob,
    float routed_scaling_factor)
{
    int experts_per_group = n_routed_experts / n_group;

    /* Allocate working arrays */
    float *sigmoid_logits  = alloc_floats(num_tokens * n_routed_experts);
    float *logits_for_choice = alloc_floats(num_tokens * n_routed_experts);
    float *group_scores    = alloc_floats(num_tokens * n_group);
    float *scores_for_choice = alloc_floats(num_tokens * n_routed_experts);

    /* Step 1: Apply sigmoid to router logits */
    copy_array(router_logits_in, sigmoid_logits, num_tokens * n_routed_experts);
    sigmoid_inplace(sigmoid_logits, num_tokens * n_routed_experts);

    /* Step 2: Add bias correction */
    for (int t = 0; t < num_tokens; t++) {
        for (int e = 0; e < n_routed_experts; e++) {
            logits_for_choice[t * n_routed_experts + e] =
                sigmoid_logits[t * n_routed_experts + e] + e_score_correction_bias[e];
        }
    }

    /* Step 3-4: Group scoring
     * For each token, for each group:
     *   - Take the top-2 expert scores within that group
     *   - Sum them to get the group score
     */
    float top2_vals[2];
    int top2_idxs[2];

    for (int t = 0; t < num_tokens; t++) {
        for (int g = 0; g < n_group; g++) {
            float *group_start = &logits_for_choice[t * n_routed_experts + g * experts_per_group];
            topk(group_start, experts_per_group, 2, top2_vals, top2_idxs);
            group_scores[t * n_group + g] = top2_vals[0] + top2_vals[1];
        }
    }

    /* Step 5: Select topk_group groups */
    float *group_topk_vals = alloc_floats(topk_group);
    int   *group_topk_idxs = (int *)malloc(topk_group * sizeof(int));

    for (int t = 0; t < num_tokens; t++) {
        topk(&group_scores[t * n_group], n_group, topk_group,
             group_topk_vals, group_topk_idxs);

        /* Step 6: Create score mask - zero out experts in non-selected groups */
        for (int e = 0; e < n_routed_experts; e++) {
            scores_for_choice[t * n_routed_experts + e] = 0.0f;
        }
        for (int gi = 0; gi < topk_group; gi++) {
            int g = group_topk_idxs[gi];
            for (int e = 0; e < experts_per_group; e++) {
                int expert_idx = g * experts_per_group + e;
                scores_for_choice[t * n_routed_experts + expert_idx] =
                    logits_for_choice[t * n_routed_experts + expert_idx];
            }
        }

        /* Step 7: Select top_k experts from masked scores */
        float *tk_vals = alloc_floats(top_k);
        int   *tk_idxs = (int *)malloc(top_k * sizeof(int));

        topk(&scores_for_choice[t * n_routed_experts], n_routed_experts, top_k,
             tk_vals, tk_idxs);

        /* Store indices */
        for (int i = 0; i < top_k; i++) {
            topk_indices_out[t * top_k + i] = tk_idxs[i];
        }

        /* Step 8: Gather weights from ORIGINAL sigmoid scores */
        for (int i = 0; i < top_k; i++) {
            topk_weights_out[t * top_k + i] = sigmoid_logits[t * n_routed_experts + tk_idxs[i]];
        }

        /* Step 9: Normalize if configured */
        if (norm_topk_prob) {
            float denom = 1e-20f;
            for (int i = 0; i < top_k; i++) {
                denom += topk_weights_out[t * top_k + i];
            }
            for (int i = 0; i < top_k; i++) {
                topk_weights_out[t * top_k + i] /= denom;
            }
        }

        /* Step 10: Scale */
        for (int i = 0; i < top_k; i++) {
            topk_weights_out[t * top_k + i] *= routed_scaling_factor;
        }

        free(tk_vals);
        free(tk_idxs);
    }

    free(group_topk_vals);
    free(group_topk_idxs);
    free(sigmoid_logits);
    free(logits_for_choice);
    free(group_scores);
    free(scores_for_choice);
}


/* ============================================================
 * Block 4: DeepseekV3NaiveMoe
 * ============================================================
 *
 * Dispatches tokens to their assigned experts and accumulates results.
 *
 * For each active expert:
 *   1. Find tokens assigned to this expert
 *   2. gate_up = F.linear(token, gate_up_proj[expert])  -- shape: 2*intermediate
 *   3. Split into gate and up halves
 *   4. hidden = silu(gate) * up
 *   5. hidden = F.linear(hidden, down_proj[expert])
 *   6. hidden *= routing_weight
 *   7. Accumulate into output
 *
 * gate_up_proj layout: (num_experts, 2*intermediate_dim, hidden_dim)
 * down_proj layout:    (num_experts, hidden_dim, intermediate_dim)
 */
void deepseek_naive_moe_forward(
    const float *hidden_states,
    const int *top_k_index,
    const float *top_k_weights,
    const float *gate_up_proj,
    const float *down_proj,
    float *output,
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int num_experts,
    int top_k)
{
    /* Zero output */
    zero_array(output, num_tokens * hidden_dim);

    /* For each expert, find which tokens are assigned to it */
    for (int expert = 0; expert < num_experts; expert++) {
        /* Find tokens and their top_k positions for this expert */
        for (int t = 0; t < num_tokens; t++) {
            for (int ki = 0; ki < top_k; ki++) {
                if (top_k_index[t * top_k + ki] != expert) continue;

                /* This token t is assigned to this expert at position ki */
                const float *token = &hidden_states[t * hidden_dim];
                float weight = top_k_weights[t * top_k + ki];

                /* gate_up_proj for this expert: (2*intermediate_dim, hidden_dim)
                 * Offset: expert * (2*intermediate_dim * hidden_dim) */
                const float *expert_gate_up =
                    &gate_up_proj[expert * (2 * intermediate_dim) * hidden_dim];

                /* Compute gate_up = token @ expert_gate_up^T
                 * token: (1, hidden_dim)
                 * expert_gate_up: (2*intermediate_dim, hidden_dim)
                 * result: (1, 2*intermediate_dim) */
                float *gate_up_result = alloc_floats(2 * intermediate_dim);
                matmul_bt(token, expert_gate_up, gate_up_result,
                          1, hidden_dim, 2 * intermediate_dim);

                /* Split into gate (first half) and up (second half) */
                float *gate = gate_up_result;                    /* first intermediate_dim */
                float *up   = &gate_up_result[intermediate_dim]; /* second intermediate_dim */

                /* Apply SiLU to gate */
                silu_inplace(gate, intermediate_dim);

                /* gate = gate * up */
                elementwise_mul(gate, up, gate, intermediate_dim);

                /* down_proj for this expert: (hidden_dim, intermediate_dim)
                 * Offset: expert * (hidden_dim * intermediate_dim) */
                const float *expert_down =
                    &down_proj[expert * hidden_dim * intermediate_dim];

                /* hidden = gate @ expert_down^T
                 * gate: (1, intermediate_dim)
                 * expert_down: (hidden_dim, intermediate_dim)
                 * result: (1, hidden_dim) */
                float *expert_output = alloc_floats(hidden_dim);
                matmul_bt(gate, expert_down, expert_output,
                          1, intermediate_dim, hidden_dim);

                /* Scale by routing weight */
                scale(expert_output, weight, hidden_dim);

                /* Accumulate into output */
                for (int d = 0; d < hidden_dim; d++) {
                    output[t * hidden_dim + d] += expert_output[d];
                }

                free(gate_up_result);
                free(expert_output);
            }
        }
    }
}


/* ============================================================
 * Block 5: Full DeepseekV3MoE Forward
 * ============================================================
 *
 * Combines all blocks:
 *   1. router_logits = gate(hidden_states)     [Router]
 *   2. topk_idx, topk_w = route_tokens(logits) [Routing]
 *   3. expert_out = experts(hidden, idx, w)     [NaiveMoe]
 *   4. shared_out = shared_experts(residuals)   [MLP]
 *   5. output = expert_out + shared_out         [Residual]
 *
 * Note: The input (residuals) is used for shared experts,
 *       and the expert output is added to it.
 */
void deepseek_moe_forward(
    const float *input,
    const float *router_weight,
    const float *e_score_correction_bias,
    const float *expert_gate_up_proj,
    const float *expert_down_proj,
    const float *shared_gate_proj_w,
    const float *shared_up_proj_w,
    const float *shared_down_proj_w,
    float *output,
    float *out_router_logits,
    int *out_topk_indices,
    float *out_topk_weights,
    int num_tokens,
    int hidden_size,
    int moe_intermediate_size,
    int n_routed_experts,
    int n_shared_experts,
    int n_group,
    int topk_group,
    int top_k,
    int norm_topk_prob,
    float routed_scaling_factor)
{
    int shared_intermediate = moe_intermediate_size * n_shared_experts;

    /* Step 1: Router forward - compute logits */
    float *router_logits = alloc_floats(num_tokens * n_routed_experts);
    deepseek_router_forward(input, router_weight, router_logits,
                             num_tokens, hidden_size, n_routed_experts);

    /* Save router logits if requested */
    if (out_router_logits) {
        copy_array(router_logits, out_router_logits, num_tokens * n_routed_experts);
    }

    /* Step 2: Route tokens to experts */
    int   *topk_indices = (int *)malloc(num_tokens * top_k * sizeof(int));
    float *topk_weights = alloc_floats(num_tokens * top_k);

    deepseek_route_tokens(router_logits, e_score_correction_bias,
                           topk_indices, topk_weights,
                           num_tokens, n_routed_experts,
                           n_group, topk_group, top_k,
                           norm_topk_prob, routed_scaling_factor);

    /* Save routing results if requested */
    if (out_topk_indices) {
        memcpy(out_topk_indices, topk_indices, num_tokens * top_k * sizeof(int));
    }
    if (out_topk_weights) {
        copy_array(topk_weights, out_topk_weights, num_tokens * top_k);
    }

    /* Step 3: Expert dispatch */
    float *expert_output = alloc_floats(num_tokens * hidden_size);
    deepseek_naive_moe_forward(input, topk_indices, topk_weights,
                                expert_gate_up_proj, expert_down_proj,
                                expert_output,
                                num_tokens, hidden_size, moe_intermediate_size,
                                n_routed_experts, top_k);

    /* Step 4: Shared expert MLP */
    float *shared_output = alloc_floats(num_tokens * hidden_size);
    deepseek_mlp_forward(input, shared_gate_proj_w, shared_up_proj_w,
                          shared_down_proj_w, shared_output,
                          num_tokens, hidden_size, shared_intermediate);

    /* Step 5: output = expert_output + shared_output */
    elementwise_add(expert_output, shared_output, output,
                     num_tokens * hidden_size);

    /* Cleanup */
    free(router_logits);
    free(topk_indices);
    free(topk_weights);
    free(expert_output);
    free(shared_output);
}
