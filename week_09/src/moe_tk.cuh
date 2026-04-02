/*
 * moe_tk.cuh - DeepSeekV3 MoE Operator with ThunderKittens
 *
 * Week 9: Sequence Parallelism
 *
 * High-level MoE blocks built on TK GEMM kernels:
 *   1. MLP:    down_proj(silu(gate_proj(x)) * up_proj(x))
 *   2. Router: F.linear(hidden_states, weight)
 *   3. Expert dispatch with weighted accumulation
 *   4. Full MoE: router + routing + experts + shared experts
 */

#pragma once

#include "moe_tk_gemm.cuh"
#include "moe_tk_ops.cuh"

/* ============================================================
 * Block 1: DeepseekV3MLP (TK-accelerated)
 * ============================================================
 * output = down_proj(silu(gate_proj(x)) * up_proj(x))
 *
 * All three linear layers use WMMA+TMA GEMM.
 */
inline void moe_mlp_tk(
    __nv_bfloat16* input,       // (num_tokens, hidden_size)
    __nv_bfloat16* gate_proj_w, // (intermediate_size, hidden_size)
    __nv_bfloat16* up_proj_w,   // (intermediate_size, hidden_size)
    __nv_bfloat16* down_proj_w, // (hidden_size, intermediate_size)
    __nv_bfloat16* output,      // (num_tokens, hidden_size)
    __nv_bfloat16* gate_buf,    // temp (num_tokens, intermediate_size)
    __nv_bfloat16* up_buf,      // temp (num_tokens, intermediate_size)
    int num_tokens,
    int hidden_size,
    int intermediate_size,
    bool use_tma)               // true = WMMA+TMA, false = WMMA only
{
    if (use_tma) {
        // gate = input @ gate_proj_w^T  via WMMA+TMA
        launch_gemm_wmma_tma(input, gate_proj_w, gate_buf,
                              num_tokens, intermediate_size, hidden_size);
        // up = input @ up_proj_w^T
        launch_gemm_wmma_tma(input, up_proj_w, up_buf,
                              num_tokens, intermediate_size, hidden_size);
    } else {
        launch_gemm_wmma(input, gate_proj_w, gate_buf,
                          num_tokens, intermediate_size, hidden_size);
        launch_gemm_wmma(input, up_proj_w, up_buf,
                          num_tokens, intermediate_size, hidden_size);
    }

    // silu(gate)
    apply_silu_bf16(gate_buf, num_tokens * intermediate_size);

    // gate *= up
    apply_emul_bf16(gate_buf, up_buf, num_tokens * intermediate_size);

    // output = gate @ down_proj_w^T
    if (use_tma) {
        launch_gemm_wmma_tma(gate_buf, down_proj_w, output,
                              num_tokens, hidden_size, intermediate_size);
    } else {
        launch_gemm_wmma(gate_buf, down_proj_w, output,
                          num_tokens, hidden_size, intermediate_size);
    }
}


/* ============================================================
 * Block 2: DeepseekV3TopkRouter (TK-accelerated)
 * ============================================================
 * router_logits = F.linear(hidden_states, weight) = input @ weight^T
 */
inline void moe_router_tk(
    __nv_bfloat16* input,         // (num_tokens, hidden_size)
    __nv_bfloat16* router_weight, // (n_experts, hidden_size)
    __nv_bfloat16* router_logits, // (num_tokens, n_experts)
    int num_tokens,
    int hidden_size,
    int n_experts,
    bool use_tma)
{
    if (use_tma) {
        launch_gemm_wmma_tma(input, router_weight, router_logits,
                              num_tokens, n_experts, hidden_size);
    } else {
        launch_gemm_wmma(input, router_weight, router_logits,
                          num_tokens, n_experts, hidden_size);
    }
}
