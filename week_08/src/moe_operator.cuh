/*
 * moe_operator.cuh - DeepSeekV3 MoE Operator - Single GPU CUDA Implementation
 *
 * Week 8: Data and Expert Parallelism
 *
 * This file implements the MoE operator blocks on a single GPU using CUDA kernels.
 * The multi-GPU parallelism layer (moe_parallel.cu) calls these functions.
 *
 * Blocks:
 *   1. DeepseekV3MLP       - Gated MLP with SiLU
 *   2. DeepseekV3TopkRouter - Router linear projection
 *   3. Token routing logic  - Sigmoid + group scoring + top-k selection
 *   4. DeepseekV3NaiveMoe  - Expert dispatch
 *   5. DeepseekV3MoE       - Full MoE forward
 */

#ifndef MOE_OPERATOR_CUH
#define MOE_OPERATOR_CUH

#include "moe_kernels.cuh"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ============================================================
 * Block 1: DeepseekV3MLP (GPU)
 * ============================================================
 * output = down_proj(silu(gate_proj(x)) * up_proj(x))
 *
 * All weight/input/output pointers must be device pointers.
 */
static inline void gpu_deepseek_mlp_forward(
    const float *d_input,       /* (num_tokens, hidden_size) */
    const float *d_gate_proj_w, /* (intermediate_size, hidden_size) */
    const float *d_up_proj_w,   /* (intermediate_size, hidden_size) */
    const float *d_down_proj_w, /* (hidden_size, intermediate_size) */
    float *d_output,            /* (num_tokens, hidden_size) */
    int num_tokens,
    int hidden_size,
    int intermediate_size,
    cudaStream_t stream = 0)
{
    float *d_gate, *d_up;
    CUDA_CHECK(cudaMalloc(&d_gate, num_tokens * intermediate_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up, num_tokens * intermediate_size * sizeof(float)));

    /* gate = input @ gate_proj_w^T */
    gpu_matmul_bt(d_input, d_gate_proj_w, d_gate,
                  num_tokens, hidden_size, intermediate_size, stream);

    /* up = input @ up_proj_w^T */
    gpu_matmul_bt(d_input, d_up_proj_w, d_up,
                  num_tokens, hidden_size, intermediate_size, stream);

    /* gate = silu(gate) */
    gpu_silu_inplace(d_gate, num_tokens * intermediate_size, stream);

    /* gate = gate * up */
    gpu_elementwise_mul(d_gate, d_up, d_gate, num_tokens * intermediate_size, stream);

    /* output = gate @ down_proj_w^T */
    gpu_matmul_bt(d_gate, d_down_proj_w, d_output,
                  num_tokens, intermediate_size, hidden_size, stream);

    CUDA_CHECK(cudaFree(d_gate));
    CUDA_CHECK(cudaFree(d_up));
}


/* ============================================================
 * Block 2: DeepseekV3TopkRouter (GPU)
 * ============================================================
 * router_logits = F.linear(hidden_states, weight) = input @ weight^T
 */
static inline void gpu_deepseek_router_forward(
    const float *d_input,         /* (num_tokens, hidden_size) */
    const float *d_router_weight, /* (n_routed_experts, hidden_size) */
    float *d_router_logits,       /* (num_tokens, n_routed_experts) */
    int num_tokens,
    int hidden_size,
    int n_routed_experts,
    cudaStream_t stream = 0)
{
    gpu_matmul_bt(d_input, d_router_weight, d_router_logits,
                  num_tokens, hidden_size, n_routed_experts, stream);
}


/* ============================================================
 * Helper: CPU-side top-k selection
 * ============================================================
 * The routing logic involves complex control flow (group scoring,
 * masking, top-k selection) that is most cleanly implemented on CPU.
 * For small token counts this is efficient.
 */
static void cpu_topk(const float *arr, int n, int k, float *values, int *indices) {
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
 * Block 3: Token Routing Logic (CPU with GPU data transfers)
 * ============================================================
 * Routing is performed on CPU because:
 *   - It involves complex control flow (group scoring, masking)
 *   - Token counts are typically small
 *   - The compute is negligible vs expert computation
 *
 * Input d_router_logits is on GPU; outputs are written to GPU.
 */
static inline void gpu_deepseek_route_tokens(
    const float *d_router_logits,          /* (num_tokens, n_routed_experts) on GPU */
    const float *d_e_score_correction_bias, /* (n_routed_experts,) on GPU */
    int *d_topk_indices,                    /* output: (num_tokens, top_k) on GPU */
    float *d_topk_weights,                  /* output: (num_tokens, top_k) on GPU */
    int num_tokens,
    int n_routed_experts,
    int n_group,
    int topk_group,
    int top_k,
    int norm_topk_prob,
    float routed_scaling_factor)
{
    int experts_per_group = n_routed_experts / n_group;
    int total_logits = num_tokens * n_routed_experts;

    /* Copy to CPU for routing logic */
    float *h_logits = (float *)malloc(total_logits * sizeof(float));
    float *h_bias = (float *)malloc(n_routed_experts * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_logits, d_router_logits, total_logits * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bias, d_e_score_correction_bias,
                           n_routed_experts * sizeof(float), cudaMemcpyDeviceToHost));

    /* Allocate CPU working arrays */
    float *sigmoid_logits = (float *)malloc(total_logits * sizeof(float));
    float *logits_for_choice = (float *)malloc(total_logits * sizeof(float));
    float *group_scores = (float *)malloc(num_tokens * n_group * sizeof(float));
    float *scores_for_choice = (float *)malloc(total_logits * sizeof(float));
    int *h_topk_indices = (int *)malloc(num_tokens * top_k * sizeof(int));
    float *h_topk_weights = (float *)malloc(num_tokens * top_k * sizeof(float));

    /* Step 1: Sigmoid */
    for (int i = 0; i < total_logits; i++) {
        sigmoid_logits[i] = 1.0f / (1.0f + expf(-h_logits[i]));
    }

    /* Step 2: Add bias */
    for (int t = 0; t < num_tokens; t++) {
        for (int e = 0; e < n_routed_experts; e++) {
            logits_for_choice[t * n_routed_experts + e] =
                sigmoid_logits[t * n_routed_experts + e] + h_bias[e];
        }
    }

    /* Step 3-4: Group scoring (top-2 per group, sum) */
    float top2_vals[2];
    int top2_idxs[2];
    for (int t = 0; t < num_tokens; t++) {
        for (int g = 0; g < n_group; g++) {
            float *group_start = &logits_for_choice[t * n_routed_experts + g * experts_per_group];
            cpu_topk(group_start, experts_per_group, 2, top2_vals, top2_idxs);
            group_scores[t * n_group + g] = top2_vals[0] + top2_vals[1];
        }
    }

    /* Step 5-6: Select top groups, mask non-selected */
    float *group_topk_vals = (float *)malloc(topk_group * sizeof(float));
    int *group_topk_idxs = (int *)malloc(topk_group * sizeof(int));

    for (int t = 0; t < num_tokens; t++) {
        cpu_topk(&group_scores[t * n_group], n_group, topk_group,
                 group_topk_vals, group_topk_idxs);

        /* Zero out non-selected groups */
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

        /* Step 7: Top-k expert selection */
        float *tk_vals = (float *)malloc(top_k * sizeof(float));
        int *tk_idxs = (int *)malloc(top_k * sizeof(int));
        cpu_topk(&scores_for_choice[t * n_routed_experts], n_routed_experts, top_k,
                 tk_vals, tk_idxs);

        for (int i = 0; i < top_k; i++) {
            h_topk_indices[t * top_k + i] = tk_idxs[i];
        }

        /* Step 8: Gather weights from original sigmoid scores */
        for (int i = 0; i < top_k; i++) {
            h_topk_weights[t * top_k + i] = sigmoid_logits[t * n_routed_experts + tk_idxs[i]];
        }

        /* Step 9: Normalize */
        if (norm_topk_prob) {
            float denom = 1e-20f;
            for (int i = 0; i < top_k; i++) {
                denom += h_topk_weights[t * top_k + i];
            }
            for (int i = 0; i < top_k; i++) {
                h_topk_weights[t * top_k + i] /= denom;
            }
        }

        /* Step 10: Scale */
        for (int i = 0; i < top_k; i++) {
            h_topk_weights[t * top_k + i] *= routed_scaling_factor;
        }

        free(tk_vals);
        free(tk_idxs);
    }

    /* Copy results back to GPU */
    CUDA_CHECK(cudaMemcpy(d_topk_indices, h_topk_indices,
                           num_tokens * top_k * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_topk_weights, h_topk_weights,
                           num_tokens * top_k * sizeof(float), cudaMemcpyHostToDevice));

    /* Cleanup */
    free(h_logits); free(h_bias);
    free(sigmoid_logits); free(logits_for_choice);
    free(group_scores); free(scores_for_choice);
    free(h_topk_indices); free(h_topk_weights);
    free(group_topk_vals); free(group_topk_idxs);
}


/* ============================================================
 * Block 4: DeepseekV3NaiveMoe - Expert Dispatch (GPU)
 * ============================================================
 * For each expert, compute gated MLP on assigned tokens.
 * Expert dispatch uses CPU for control flow, GPU for compute.
 */
static inline void gpu_deepseek_naive_moe_forward(
    const float *d_hidden_states, /* (num_tokens, hidden_dim) */
    const int *d_top_k_index,     /* (num_tokens, top_k) */
    const float *d_top_k_weights, /* (num_tokens, top_k) */
    const float *d_gate_up_proj,  /* (num_experts, 2*intermediate_dim, hidden_dim) */
    const float *d_down_proj,     /* (num_experts, hidden_dim, intermediate_dim) */
    float *d_output,              /* (num_tokens, hidden_dim) */
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int num_experts,
    int top_k,
    cudaStream_t stream = 0)
{
    /* Zero output */
    CUDA_CHECK(cudaMemset(d_output, 0, num_tokens * hidden_dim * sizeof(float)));

    /* Copy routing info to CPU for dispatch logic */
    int *h_top_k_index = (int *)malloc(num_tokens * top_k * sizeof(int));
    float *h_top_k_weights = (float *)malloc(num_tokens * top_k * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_top_k_index, d_top_k_index,
                           num_tokens * top_k * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_top_k_weights, d_top_k_weights,
                           num_tokens * top_k * sizeof(float), cudaMemcpyDeviceToHost));

    /* Temporary GPU buffers for single-token expert computation */
    float *d_gate_up_result, *d_expert_output;
    CUDA_CHECK(cudaMalloc(&d_gate_up_result, 2 * intermediate_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_expert_output, hidden_dim * sizeof(float)));

    for (int expert = 0; expert < num_experts; expert++) {
        for (int t = 0; t < num_tokens; t++) {
            for (int ki = 0; ki < top_k; ki++) {
                if (h_top_k_index[t * top_k + ki] != expert) continue;

                float weight = h_top_k_weights[t * top_k + ki];

                const float *d_token = &d_hidden_states[t * hidden_dim];
                const float *d_expert_gate_up =
                    &d_gate_up_proj[expert * (2 * intermediate_dim) * hidden_dim];
                const float *d_expert_down =
                    &d_down_proj[expert * hidden_dim * intermediate_dim];

                /* gate_up = token @ expert_gate_up^T  -> (1, 2*intermediate_dim) */
                gpu_matmul_bt(d_token, d_expert_gate_up, d_gate_up_result,
                              1, hidden_dim, 2 * intermediate_dim, stream);

                /* gate = first half, up = second half */
                float *d_gate = d_gate_up_result;
                float *d_up = &d_gate_up_result[intermediate_dim];

                /* gate = silu(gate) */
                gpu_silu_inplace(d_gate, intermediate_dim, stream);

                /* gate = gate * up */
                gpu_elementwise_mul(d_gate, d_up, d_gate, intermediate_dim, stream);

                /* expert_output = gate @ expert_down^T  -> (1, hidden_dim) */
                gpu_matmul_bt(d_gate, d_expert_down, d_expert_output,
                              1, intermediate_dim, hidden_dim, stream);

                /* Scale by routing weight */
                gpu_scale(d_expert_output, weight, hidden_dim, stream);

                /* Accumulate into output[t] */
                gpu_accumulate(&d_output[t * hidden_dim], d_expert_output,
                               hidden_dim, stream);

                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
        }
    }

    CUDA_CHECK(cudaFree(d_gate_up_result));
    CUDA_CHECK(cudaFree(d_expert_output));
    free(h_top_k_index);
    free(h_top_k_weights);
}


/* ============================================================
 * Block 5: Full DeepseekV3MoE Forward (Single GPU)
 * ============================================================
 * Combines: Router -> Routing -> Expert Dispatch -> Shared Expert -> Add
 */
static inline void gpu_deepseek_moe_forward(
    const float *d_input,
    const float *d_router_weight,
    const float *d_e_score_correction_bias,
    const float *d_expert_gate_up_proj,
    const float *d_expert_down_proj,
    const float *d_shared_gate_proj_w,
    const float *d_shared_up_proj_w,
    const float *d_shared_down_proj_w,
    float *d_output,
    float *d_out_router_logits,  /* NULL if not needed */
    int *d_out_topk_indices,     /* NULL if not needed */
    float *d_out_topk_weights,   /* NULL if not needed */
    int num_tokens,
    int hidden_size,
    int moe_intermediate_size,
    int n_routed_experts,
    int n_shared_experts,
    int n_group,
    int topk_group,
    int top_k,
    int norm_topk_prob,
    float routed_scaling_factor,
    cudaStream_t stream = 0)
{
    int shared_intermediate = moe_intermediate_size * n_shared_experts;

    /* Step 1: Router forward */
    float *d_router_logits;
    CUDA_CHECK(cudaMalloc(&d_router_logits,
                           num_tokens * n_routed_experts * sizeof(float)));
    gpu_deepseek_router_forward(d_input, d_router_weight, d_router_logits,
                                 num_tokens, hidden_size, n_routed_experts, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (d_out_router_logits) {
        CUDA_CHECK(cudaMemcpy(d_out_router_logits, d_router_logits,
                               num_tokens * n_routed_experts * sizeof(float),
                               cudaMemcpyDeviceToDevice));
    }

    /* Step 2: Route tokens */
    int *d_topk_indices;
    float *d_topk_weights;
    CUDA_CHECK(cudaMalloc(&d_topk_indices, num_tokens * top_k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_topk_weights, num_tokens * top_k * sizeof(float)));

    gpu_deepseek_route_tokens(d_router_logits, d_e_score_correction_bias,
                               d_topk_indices, d_topk_weights,
                               num_tokens, n_routed_experts,
                               n_group, topk_group, top_k,
                               norm_topk_prob, routed_scaling_factor);

    if (d_out_topk_indices) {
        CUDA_CHECK(cudaMemcpy(d_out_topk_indices, d_topk_indices,
                               num_tokens * top_k * sizeof(int),
                               cudaMemcpyDeviceToDevice));
    }
    if (d_out_topk_weights) {
        CUDA_CHECK(cudaMemcpy(d_out_topk_weights, d_topk_weights,
                               num_tokens * top_k * sizeof(float),
                               cudaMemcpyDeviceToDevice));
    }

    /* Step 3: Expert dispatch */
    float *d_expert_output;
    CUDA_CHECK(cudaMalloc(&d_expert_output, num_tokens * hidden_size * sizeof(float)));
    gpu_deepseek_naive_moe_forward(d_input, d_topk_indices, d_topk_weights,
                                    d_expert_gate_up_proj, d_expert_down_proj,
                                    d_expert_output,
                                    num_tokens, hidden_size, moe_intermediate_size,
                                    n_routed_experts, top_k, stream);

    /* Step 4: Shared expert MLP */
    float *d_shared_output;
    CUDA_CHECK(cudaMalloc(&d_shared_output, num_tokens * hidden_size * sizeof(float)));
    gpu_deepseek_mlp_forward(d_input, d_shared_gate_proj_w, d_shared_up_proj_w,
                              d_shared_down_proj_w, d_shared_output,
                              num_tokens, hidden_size, shared_intermediate, stream);

    /* Step 5: output = expert_output + shared_output */
    gpu_elementwise_add(d_expert_output, d_shared_output, d_output,
                         num_tokens * hidden_size, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* Cleanup */
    CUDA_CHECK(cudaFree(d_router_logits));
    CUDA_CHECK(cudaFree(d_topk_indices));
    CUDA_CHECK(cudaFree(d_topk_weights));
    CUDA_CHECK(cudaFree(d_expert_output));
    CUDA_CHECK(cudaFree(d_shared_output));
}

#endif /* MOE_OPERATOR_CUH */
