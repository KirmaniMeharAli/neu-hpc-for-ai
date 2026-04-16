/*
 * moe_parallel.cu - Multi-GPU DeepSeekV3 MoE with Data + Expert Parallelism
 *
 * Week 8: Data and Expert Parallelism using NCCL
 *
 * Architecture:
 *   - Data Parallelism: Input tokens are partitioned across GPUs
 *   - Expert Parallelism: Experts are partitioned across GPUs
 *   - Router is replicated on all GPUs
 *   - Shared experts are replicated on all GPUs
 *
 * Communication Pattern (from lecture):
 *   1. Routing:     Each GPU runs router on its local tokens
 *   2. Permutation: NCCL All-to-All sends tokens to the GPU that owns the target expert
 *   3. Computation: Each GPU computes its local experts on received tokens
 *   4. Un-permute:  NCCL All-to-All sends expert outputs back to originating GPUs
 *   5. Scaling:     Each GPU scales and sums the expert outputs for its local tokens
 *
 * References:
 *   - "A Hybrid Tensor-Expert-Data Parallelism Approach to Optimize MoE Training"
 *   - "MegaScale-MoE: Large-Scale Communication-Efficient Training of MoE Models"
 *   - NCCL Developer Guide: https://docs.nvidia.com/deeplearning/nccl
 */

#ifndef MOE_PARALLEL_CU_INCLUDED
#define MOE_PARALLEL_CU_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <pthread.h>

#include "moe_kernels.cuh"
#include "moe_operator.cuh"

/* ============================================================
 * NCCL Error Checking
 * ============================================================ */
#define NCCL_CHECK(call)                                                       \
    do {                                                                        \
        ncclResult_t res = (call);                                             \
        if (res != ncclSuccess) {                                              \
            fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    ncclGetErrorString(res));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)


/* ============================================================
 * MoE Configuration
 * ============================================================ */
typedef struct {
    int hidden_size;
    int moe_intermediate_size;
    int n_routed_experts;
    int n_shared_experts;
    int n_group;
    int topk_group;
    int top_k;
    int norm_topk_prob;
    float routed_scaling_factor;
} MoEConfig;


/* ============================================================
 * Per-GPU State for Parallel MoE
 * ============================================================
 * Each GPU holds:
 *   - Its partition of the input tokens (data parallelism)
 *   - Its partition of the experts (expert parallelism)
 *   - A full copy of the router and shared expert weights (replicated)
 */
typedef struct {
    int gpu_id;
    int num_gpus;
    ncclComm_t comm;
    cudaStream_t stream;

    /* Configuration */
    MoEConfig config;

    /* Number of local tokens (data parallelism) */
    int local_num_tokens;

    /* Expert partition: experts [expert_start, expert_end) are on this GPU */
    int expert_start;
    int expert_end;
    int local_num_experts;

    /* Device pointers - Replicated weights */
    float *d_router_weight;           /* (n_routed_experts, hidden_size) */
    float *d_e_score_correction_bias; /* (n_routed_experts,) */
    float *d_shared_gate_proj_w;      /* (shared_inter, hidden) */
    float *d_shared_up_proj_w;        /* (shared_inter, hidden) */
    float *d_shared_down_proj_w;      /* (hidden, shared_inter) */

    /* Device pointers - Partitioned expert weights (only local experts) */
    float *d_local_gate_up_proj;      /* (local_num_experts, 2*moe_inter, hidden) */
    float *d_local_down_proj;         /* (local_num_experts, hidden, moe_inter) */

    /* Device pointers - Local input/output tokens */
    float *d_local_input;             /* (local_num_tokens, hidden_size) */
    float *d_local_output;            /* (local_num_tokens, hidden_size) */

    /* Intermediate device buffers */
    float *d_router_logits;           /* (local_num_tokens, n_routed_experts) */
    int   *d_topk_indices;            /* (local_num_tokens, top_k) */
    float *d_topk_weights;            /* (local_num_tokens, top_k) */
} GPUState;


/* ============================================================
 * Initialize GPU State
 * ============================================================ */
void init_gpu_state(GPUState *state, int gpu_id, int num_gpus,
                    ncclComm_t comm, const MoEConfig *config,
                    int total_num_tokens) {
    state->gpu_id = gpu_id;
    state->num_gpus = num_gpus;
    state->comm = comm;
    state->config = *config;

    CUDA_CHECK(cudaSetDevice(gpu_id));
    CUDA_CHECK(cudaStreamCreate(&state->stream));

    /* Data parallelism: partition tokens evenly */
    int base_tokens = total_num_tokens / num_gpus;
    int remainder = total_num_tokens % num_gpus;
    state->local_num_tokens = base_tokens + (gpu_id < remainder ? 1 : 0);

    /* Expert parallelism: partition experts evenly */
    int base_experts = config->n_routed_experts / num_gpus;
    int expert_remainder = config->n_routed_experts % num_gpus;
    state->expert_start = gpu_id * base_experts + (gpu_id < expert_remainder ? gpu_id : expert_remainder);
    state->local_num_experts = base_experts + (gpu_id < expert_remainder ? 1 : 0);
    state->expert_end = state->expert_start + state->local_num_experts;
}


/* ============================================================
 * Allocate and Upload Weights to GPU
 * ============================================================ */
void upload_weights_to_gpu(GPUState *state,
                            const float *h_router_weight,
                            const float *h_e_score_bias,
                            const float *h_expert_gate_up_proj,
                            const float *h_expert_down_proj,
                            const float *h_shared_gate_w,
                            const float *h_shared_up_w,
                            const float *h_shared_down_w) {
    CUDA_CHECK(cudaSetDevice(state->gpu_id));

    int hs = state->config.hidden_size;
    int mi = state->config.moe_intermediate_size;
    int ne = state->config.n_routed_experts;
    int ns = state->config.n_shared_experts;
    int shared_inter = mi * ns;

    /* Replicated: router weights */
    CUDA_CHECK(cudaMalloc(&state->d_router_weight, ne * hs * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state->d_router_weight, h_router_weight,
                           ne * hs * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state->d_e_score_correction_bias, ne * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state->d_e_score_correction_bias, h_e_score_bias,
                           ne * sizeof(float), cudaMemcpyHostToDevice));

    /* Replicated: shared expert weights */
    CUDA_CHECK(cudaMalloc(&state->d_shared_gate_proj_w, shared_inter * hs * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state->d_shared_gate_proj_w, h_shared_gate_w,
                           shared_inter * hs * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state->d_shared_up_proj_w, shared_inter * hs * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state->d_shared_up_proj_w, h_shared_up_w,
                           shared_inter * hs * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state->d_shared_down_proj_w, hs * shared_inter * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state->d_shared_down_proj_w, h_shared_down_w,
                           hs * shared_inter * sizeof(float), cudaMemcpyHostToDevice));

    /* Partitioned: local expert weights */
    int lne = state->local_num_experts;
    int gate_up_size_per_expert = 2 * mi * hs;
    int down_size_per_expert = hs * mi;

    CUDA_CHECK(cudaMalloc(&state->d_local_gate_up_proj,
                           lne * gate_up_size_per_expert * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state->d_local_gate_up_proj,
                           &h_expert_gate_up_proj[state->expert_start * gate_up_size_per_expert],
                           lne * gate_up_size_per_expert * sizeof(float),
                           cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state->d_local_down_proj,
                           lne * down_size_per_expert * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state->d_local_down_proj,
                           &h_expert_down_proj[state->expert_start * down_size_per_expert],
                           lne * down_size_per_expert * sizeof(float),
                           cudaMemcpyHostToDevice));

    /* Allocate token buffers */
    CUDA_CHECK(cudaMalloc(&state->d_local_input,
                           state->local_num_tokens * hs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_local_output,
                           state->local_num_tokens * hs * sizeof(float)));

    /* Allocate intermediate buffers */
    CUDA_CHECK(cudaMalloc(&state->d_router_logits,
                           state->local_num_tokens * ne * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_topk_indices,
                           state->local_num_tokens * state->config.top_k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->d_topk_weights,
                           state->local_num_tokens * state->config.top_k * sizeof(float)));
}


/* ============================================================
 * Upload Local Input Tokens
 * ============================================================ */
void upload_local_input(GPUState *state, const float *h_input, int total_tokens) {
    CUDA_CHECK(cudaSetDevice(state->gpu_id));
    int hs = state->config.hidden_size;

    /* Compute this GPU's token offset */
    int offset = 0;
    for (int g = 0; g < state->gpu_id; g++) {
        int base = total_tokens / state->num_gpus;
        int rem = total_tokens % state->num_gpus;
        offset += base + (g < rem ? 1 : 0);
    }

    CUDA_CHECK(cudaMemcpy(state->d_local_input,
                           &h_input[offset * hs],
                           state->local_num_tokens * hs * sizeof(float),
                           cudaMemcpyHostToDevice));
}


/* ============================================================
 * NCCL All-to-All Communication
 * ============================================================
 * Emulates all-to-all using NCCL point-to-point ncclSend/ncclRecv.
 *
 * send_bufs[i] = data this GPU sends TO GPU i,  send_counts[i] floats
 * recv_bufs[i] = data this GPU recvs FROM GPU i, recv_counts[i] floats
 */
void nccl_all_to_all(float **d_send_bufs, int *send_counts,
                      float **d_recv_bufs, int *recv_counts,
                      int num_gpus, ncclComm_t comm, cudaStream_t stream) {
    NCCL_CHECK(ncclGroupStart());
    for (int peer = 0; peer < num_gpus; peer++) {
        if (send_counts[peer] > 0) {
            NCCL_CHECK(ncclSend(d_send_bufs[peer], send_counts[peer],
                                 ncclFloat, peer, comm, stream));
        }
        if (recv_counts[peer] > 0) {
            NCCL_CHECK(ncclRecv(d_recv_bufs[peer], recv_counts[peer],
                                 ncclFloat, peer, comm, stream));
        }
    }
    NCCL_CHECK(ncclGroupEnd());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}


/* ============================================================
 * Parallel MoE Forward Pass
 * ============================================================
 *
 * This implements the full data + expert parallel MoE forward:
 *
 * Phase 1 - Routing (local):
 *   Each GPU computes router logits on its local tokens and
 *   determines top-k expert assignments.
 *
 * Phase 2 - Permutation (NCCL All-to-All):
 *   Tokens are sent to the GPU that owns the assigned expert.
 *   GPU g owns experts [expert_start_g, expert_end_g).
 *
 * Phase 3 - Expert Computation (local):
 *   Each GPU runs its local experts on the tokens it received.
 *
 * Phase 4 - Un-permutation (NCCL All-to-All):
 *   Expert outputs are sent back to the originating GPUs.
 *
 * Phase 5 - Scaling and Accumulation (local):
 *   Each GPU scales expert outputs by routing weights and sums.
 *   Then adds the shared expert output.
 */
void parallel_moe_forward(GPUState *state) {
    CUDA_CHECK(cudaSetDevice(state->gpu_id));

    int hs = state->config.hidden_size;
    int mi = state->config.moe_intermediate_size;
    int ne = state->config.n_routed_experts;
    int ns = state->config.n_shared_experts;
    int top_k = state->config.top_k;
    int local_tokens = state->local_num_tokens;
    int num_gpus = state->num_gpus;
    cudaStream_t stream = state->stream;

    /* ================================================================
     * Phase 1: Routing (local on each GPU)
     * ================================================================
     * Each GPU runs the router on its local tokens to determine
     * which experts each token should be sent to.
     */
    gpu_deepseek_router_forward(state->d_local_input, state->d_router_weight,
                                 state->d_router_logits,
                                 local_tokens, hs, ne, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    gpu_deepseek_route_tokens(state->d_router_logits,
                               state->d_e_score_correction_bias,
                               state->d_topk_indices, state->d_topk_weights,
                               local_tokens, ne,
                               state->config.n_group, state->config.topk_group,
                               top_k, state->config.norm_topk_prob,
                               state->config.routed_scaling_factor);

    /* Copy routing results to host for dispatch planning */
    int *h_topk_indices = (int *)malloc(local_tokens * top_k * sizeof(int));
    float *h_topk_weights = (float *)malloc(local_tokens * top_k * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_topk_indices, state->d_topk_indices,
                           local_tokens * top_k * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_topk_weights, state->d_topk_weights,
                           local_tokens * top_k * sizeof(float), cudaMemcpyDeviceToHost));

    /* ================================================================
     * Phase 2: Permutation (NCCL All-to-All)
     * ================================================================
     * For each (token, top_k_slot), determine which GPU owns the target
     * expert and prepare send buffers.
     *
     * We build a dispatch table:
     *   For each GPU peer, collect the tokens to send.
     */

    /* Count tokens to send to each GPU */
    int *send_token_counts = (int *)calloc(num_gpus, sizeof(int));
    for (int t = 0; t < local_tokens; t++) {
        for (int ki = 0; ki < top_k; ki++) {
            int expert_id = h_topk_indices[t * top_k + ki];
            /* Find which GPU owns this expert */
            for (int g = 0; g < num_gpus; g++) {
                int base_e = ne / num_gpus;
                int rem_e = ne % num_gpus;
                int start_e = g * base_e + (g < rem_e ? g : rem_e);
                int count_e = base_e + (g < rem_e ? 1 : 0);
                if (expert_id >= start_e && expert_id < start_e + count_e) {
                    send_token_counts[g]++;
                    break;
                }
            }
        }
    }

    /* Build send lists: for each GPU, list of (token_data, expert_id, weight, source_token_idx, source_ki) */
    /* We'll pack: [hidden_state(hs floats), expert_id(1 float), weight(1 float), src_token(1 float), src_ki(1 float)] */
    int pack_size = hs + 4; /* hidden_state + expert_id + weight + src_token + src_ki */

    float **h_send_bufs = (float **)malloc(num_gpus * sizeof(float *));
    int *send_float_counts = (int *)malloc(num_gpus * sizeof(int));
    int *send_offsets = (int *)calloc(num_gpus, sizeof(int));

    for (int g = 0; g < num_gpus; g++) {
        send_float_counts[g] = send_token_counts[g] * pack_size;
        h_send_bufs[g] = (float *)calloc(send_float_counts[g] > 0 ? send_float_counts[g] : 1, sizeof(float));
    }

    /* Copy local input to host for packing */
    float *h_local_input = (float *)malloc(local_tokens * hs * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_local_input, state->d_local_input,
                           local_tokens * hs * sizeof(float), cudaMemcpyDeviceToHost));

    /* Pack send buffers */
    for (int t = 0; t < local_tokens; t++) {
        for (int ki = 0; ki < top_k; ki++) {
            int expert_id = h_topk_indices[t * top_k + ki];
            float weight = h_topk_weights[t * top_k + ki];

            /* Find owning GPU */
            int target_gpu = -1;
            for (int g = 0; g < num_gpus; g++) {
                int base_e = ne / num_gpus;
                int rem_e = ne % num_gpus;
                int start_e = g * base_e + (g < rem_e ? g : rem_e);
                int count_e = base_e + (g < rem_e ? 1 : 0);
                if (expert_id >= start_e && expert_id < start_e + count_e) {
                    target_gpu = g;
                    break;
                }
            }

            int off = send_offsets[target_gpu];
            float *buf = &h_send_bufs[target_gpu][off * pack_size];
            memcpy(buf, &h_local_input[t * hs], hs * sizeof(float));
            buf[hs + 0] = (float)expert_id;
            buf[hs + 1] = weight;
            buf[hs + 2] = (float)t;  /* source token index */
            buf[hs + 3] = (float)ki; /* source top_k slot */
            send_offsets[target_gpu]++;
        }
    }

    /* Upload send buffers to GPU */
    float **d_send_bufs = (float **)malloc(num_gpus * sizeof(float *));
    for (int g = 0; g < num_gpus; g++) {
        int nbytes = send_float_counts[g] > 0 ? send_float_counts[g] * sizeof(float) : sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_send_bufs[g], nbytes));
        if (send_float_counts[g] > 0) {
            CUDA_CHECK(cudaMemcpy(d_send_bufs[g], h_send_bufs[g],
                                   send_float_counts[g] * sizeof(float),
                                   cudaMemcpyHostToDevice));
        }
    }

    /* Exchange counts via NCCL AllGather so each GPU knows recv counts */
    /* For simplicity, we use a host-side gather since num_gpus is small */
    /* In production, this would use NCCL AllGather */
    /* We pre-compute recv counts: each GPU's send_token_counts[state->gpu_id]
       tells us how many tokens GPU g is sending to us */
    /* But since each GPU only knows its own send counts, we need to
       communicate. We'll use a simple approach: each GPU computes
       recv counts from the global routing info. Since all GPUs have
       the same router weights and each GPU knows which tokens it has,
       we can determine this deterministically. */

    /* Actually, for the all-to-all, we need to know how many items
       we'll RECEIVE from each peer. We use ncclSend/ncclRecv with
       pre-exchanged counts. For this implementation, we first exchange
       counts using NCCL, then exchange data. */

    /* Exchange counts: send our send_token_counts to all peers as their recv counts */
    /* Use NCCL point-to-point to exchange count info */
    int *recv_token_counts = (int *)calloc(num_gpus, sizeof(int));
    int *recv_float_counts = (int *)malloc(num_gpus * sizeof(int));

    /* Exchange counts using NCCL */
    int *d_send_count_buf, *d_recv_count_buf;
    CUDA_CHECK(cudaMalloc(&d_send_count_buf, num_gpus * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_recv_count_buf, num_gpus * sizeof(int)));

    /* Pack: d_send_count_buf[g] = number of tokens this GPU sends to GPU g */
    CUDA_CHECK(cudaMemcpy(d_send_count_buf, send_token_counts,
                           num_gpus * sizeof(int), cudaMemcpyHostToDevice));

    /* All-to-all count exchange: each GPU sends its count to every other GPU */
    NCCL_CHECK(ncclGroupStart());
    for (int peer = 0; peer < num_gpus; peer++) {
        /* Send our send_count[peer] to peer */
        NCCL_CHECK(ncclSend(&d_send_count_buf[peer], 1, ncclInt, peer,
                             state->comm, stream));
        /* Recv peer's send_count[gpu_id] into our recv_token_counts[peer] */
        NCCL_CHECK(ncclRecv(&d_recv_count_buf[peer], 1, ncclInt, peer,
                             state->comm, stream));
    }
    NCCL_CHECK(ncclGroupEnd());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(recv_token_counts, d_recv_count_buf,
                           num_gpus * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_send_count_buf));
    CUDA_CHECK(cudaFree(d_recv_count_buf));

    for (int g = 0; g < num_gpus; g++) {
        recv_float_counts[g] = recv_token_counts[g] * pack_size;
    }

    /* Allocate recv buffers */
    float **d_recv_bufs = (float **)malloc(num_gpus * sizeof(float *));
    for (int g = 0; g < num_gpus; g++) {
        int nbytes = recv_float_counts[g] > 0 ? recv_float_counts[g] * sizeof(float) : sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_recv_bufs[g], nbytes));
    }

    /* All-to-All data exchange */
    nccl_all_to_all(d_send_bufs, send_float_counts,
                     d_recv_bufs, recv_float_counts,
                     num_gpus, state->comm, stream);

    /* ================================================================
     * Phase 3: Expert Computation (local)
     * ================================================================
     * Unpack received tokens, run through local experts, produce outputs.
     */

    int total_recv_tokens = 0;
    for (int g = 0; g < num_gpus; g++) {
        total_recv_tokens += recv_token_counts[g];
    }

    /* Copy all recv buffers to host for unpacking */
    float **h_recv_bufs = (float **)malloc(num_gpus * sizeof(float *));
    for (int g = 0; g < num_gpus; g++) {
        h_recv_bufs[g] = (float *)malloc(
            (recv_float_counts[g] > 0 ? recv_float_counts[g] : 1) * sizeof(float));
        if (recv_float_counts[g] > 0) {
            CUDA_CHECK(cudaMemcpy(h_recv_bufs[g], d_recv_bufs[g],
                                   recv_float_counts[g] * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        }
    }

    /* Allocate output for expert computation: one output per received token */
    float *h_expert_outputs = (float *)calloc(
        (total_recv_tokens > 0 ? total_recv_tokens : 1) * hs, sizeof(float));

    /* Process received tokens through local experts */
    /* We need GPU buffers for expert computation */
    float *d_token_buf, *d_gate_up_result, *d_expert_out;
    CUDA_CHECK(cudaMalloc(&d_token_buf, hs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gate_up_result, 2 * mi * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_expert_out, hs * sizeof(float)));

    int recv_idx = 0;
    for (int g = 0; g < num_gpus; g++) {
        for (int r = 0; r < recv_token_counts[g]; r++) {
            float *packed = &h_recv_bufs[g][r * pack_size];
            float *token_data = packed;          /* hs floats */
            int expert_id = (int)packed[hs + 0];
            /* weight is packed[hs+1] but we don't use it here;
               scaling is done after un-permutation */

            /* Map global expert_id to local expert index */
            int local_expert_idx = expert_id - state->expert_start;
            if (local_expert_idx < 0 || local_expert_idx >= state->local_num_experts) {
                /* This shouldn't happen if routing is correct */
                fprintf(stderr, "GPU %d: Expert %d not local! (range [%d,%d))\n",
                        state->gpu_id, expert_id, state->expert_start, state->expert_end);
                recv_idx++;
                continue;
            }

            /* Upload token to GPU */
            CUDA_CHECK(cudaMemcpy(d_token_buf, token_data, hs * sizeof(float),
                                   cudaMemcpyHostToDevice));

            /* Get local expert weights */
            const float *d_expert_gate_up =
                &state->d_local_gate_up_proj[local_expert_idx * 2 * mi * hs];
            const float *d_expert_down =
                &state->d_local_down_proj[local_expert_idx * hs * mi];

            /* Compute: gate_up = token @ gate_up^T */
            gpu_matmul_bt(d_token_buf, d_expert_gate_up, d_gate_up_result,
                          1, hs, 2 * mi, stream);

            /* SiLU on gate half, multiply with up half */
            float *d_gate = d_gate_up_result;
            float *d_up = &d_gate_up_result[mi];
            gpu_silu_inplace(d_gate, mi, stream);
            gpu_elementwise_mul(d_gate, d_up, d_gate, mi, stream);

            /* down_proj: expert_out = gate @ down^T */
            gpu_matmul_bt(d_gate, d_expert_down, d_expert_out,
                          1, mi, hs, stream);

            /* Copy result back to host */
            CUDA_CHECK(cudaMemcpy(&h_expert_outputs[recv_idx * hs], d_expert_out,
                                   hs * sizeof(float), cudaMemcpyDeviceToHost));
            recv_idx++;
        }
    }

    CUDA_CHECK(cudaFree(d_token_buf));
    CUDA_CHECK(cudaFree(d_gate_up_result));
    CUDA_CHECK(cudaFree(d_expert_out));

    /* ================================================================
     * Phase 4: Un-permutation (NCCL All-to-All)
     * ================================================================
     * Send expert outputs back to originating GPUs.
     * Pack: [expert_output(hs floats), src_token(1 float), src_ki(1 float), weight(1 float)]
     */
    int unpack_size = hs + 3; /* output + src_token + src_ki + weight */

    /* Build reverse send buffers: for each source GPU, pack the expert outputs */
    float **h_rev_send_bufs = (float **)malloc(num_gpus * sizeof(float *));
    int *rev_send_float_counts = (int *)malloc(num_gpus * sizeof(int));
    int *rev_send_offsets = (int *)calloc(num_gpus, sizeof(int));

    for (int g = 0; g < num_gpus; g++) {
        rev_send_float_counts[g] = recv_token_counts[g] * unpack_size;
        h_rev_send_bufs[g] = (float *)calloc(
            rev_send_float_counts[g] > 0 ? rev_send_float_counts[g] : 1, sizeof(float));
    }

    recv_idx = 0;
    for (int g = 0; g < num_gpus; g++) {
        for (int r = 0; r < recv_token_counts[g]; r++) {
            float *packed_in = &h_recv_bufs[g][r * pack_size];
            float weight = packed_in[hs + 1];
            float src_token = packed_in[hs + 2];
            float src_ki = packed_in[hs + 3];

            float *buf = &h_rev_send_bufs[g][rev_send_offsets[g] * unpack_size];
            memcpy(buf, &h_expert_outputs[recv_idx * hs], hs * sizeof(float));
            buf[hs + 0] = src_token;
            buf[hs + 1] = src_ki;
            buf[hs + 2] = weight;
            rev_send_offsets[g]++;
            recv_idx++;
        }
    }

    /* Upload reverse send buffers to GPU */
    float **d_rev_send_bufs = (float **)malloc(num_gpus * sizeof(float *));
    for (int g = 0; g < num_gpus; g++) {
        int nbytes = rev_send_float_counts[g] > 0 ? rev_send_float_counts[g] * sizeof(float) : sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_rev_send_bufs[g], nbytes));
        if (rev_send_float_counts[g] > 0) {
            CUDA_CHECK(cudaMemcpy(d_rev_send_bufs[g], h_rev_send_bufs[g],
                                   rev_send_float_counts[g] * sizeof(float),
                                   cudaMemcpyHostToDevice));
        }
    }

    /* Reverse recv counts = original send counts */
    int *rev_recv_float_counts = (int *)malloc(num_gpus * sizeof(int));
    for (int g = 0; g < num_gpus; g++) {
        rev_recv_float_counts[g] = send_token_counts[g] * unpack_size;
    }

    float **d_rev_recv_bufs = (float **)malloc(num_gpus * sizeof(float *));
    for (int g = 0; g < num_gpus; g++) {
        int nbytes = rev_recv_float_counts[g] > 0 ? rev_recv_float_counts[g] * sizeof(float) : sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_rev_recv_bufs[g], nbytes));
    }

    /* Reverse All-to-All */
    nccl_all_to_all(d_rev_send_bufs, rev_send_float_counts,
                     d_rev_recv_bufs, rev_recv_float_counts,
                     num_gpus, state->comm, stream);

    /* ================================================================
     * Phase 5: Scaling and Accumulation (local)
     * ================================================================
     * Unpack expert outputs, scale by routing weight, accumulate per token.
     */

    /* Zero local output on GPU */
    CUDA_CHECK(cudaMemset(state->d_local_output, 0,
                           local_tokens * hs * sizeof(float)));

    /* Accumulate expert outputs on host, then upload */
    float *h_expert_accum = (float *)calloc(local_tokens * hs, sizeof(float));

    for (int g = 0; g < num_gpus; g++) {
        if (rev_recv_float_counts[g] == 0) continue;
        float *h_rev_recv = (float *)malloc(rev_recv_float_counts[g] * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_rev_recv, d_rev_recv_bufs[g],
                               rev_recv_float_counts[g] * sizeof(float),
                               cudaMemcpyDeviceToHost));

        int num_items = rev_recv_float_counts[g] / unpack_size;
        for (int r = 0; r < num_items; r++) {
            float *buf = &h_rev_recv[r * unpack_size];
            int src_token = (int)buf[hs + 0];
            /* src_ki = (int)buf[hs + 1]; -- not needed for accumulation */
            float weight = buf[hs + 2];

            /* Scale and accumulate */
            for (int d = 0; d < hs; d++) {
                h_expert_accum[src_token * hs + d] += buf[d] * weight;
            }
        }
        free(h_rev_recv);
    }

    /* Upload accumulated expert output to GPU */
    float *d_expert_accum;
    CUDA_CHECK(cudaMalloc(&d_expert_accum, local_tokens * hs * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_expert_accum, h_expert_accum,
                           local_tokens * hs * sizeof(float), cudaMemcpyHostToDevice));

    /* Compute shared experts on local tokens */
    int shared_inter = mi * ns;
    float *d_shared_output;
    CUDA_CHECK(cudaMalloc(&d_shared_output, local_tokens * hs * sizeof(float)));
    gpu_deepseek_mlp_forward(state->d_local_input,
                              state->d_shared_gate_proj_w,
                              state->d_shared_up_proj_w,
                              state->d_shared_down_proj_w,
                              d_shared_output,
                              local_tokens, hs, shared_inter, stream);

    /* Final output = expert_accum + shared_output */
    gpu_elementwise_add(d_expert_accum, d_shared_output, state->d_local_output,
                         local_tokens * hs, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* ================================================================
     * Cleanup
     * ================================================================ */
    CUDA_CHECK(cudaFree(d_expert_accum));
    CUDA_CHECK(cudaFree(d_shared_output));

    for (int g = 0; g < num_gpus; g++) {
        CUDA_CHECK(cudaFree(d_send_bufs[g]));
        CUDA_CHECK(cudaFree(d_recv_bufs[g]));
        CUDA_CHECK(cudaFree(d_rev_send_bufs[g]));
        CUDA_CHECK(cudaFree(d_rev_recv_bufs[g]));
        free(h_send_bufs[g]);
        free(h_recv_bufs[g]);
        free(h_rev_send_bufs[g]);
    }
    free(d_send_bufs); free(d_recv_bufs);
    free(d_rev_send_bufs); free(d_rev_recv_bufs);
    free(h_send_bufs); free(h_recv_bufs);
    free(h_rev_send_bufs);
    free(send_token_counts); free(send_float_counts); free(send_offsets);
    free(recv_token_counts); free(recv_float_counts);
    free(rev_send_float_counts); free(rev_recv_float_counts);
    free(rev_send_offsets);
    free(h_topk_indices); free(h_topk_weights);
    free(h_local_input);
    free(h_expert_outputs);
    free(h_expert_accum);
}


/* ============================================================
 * Download Output
 * ============================================================ */
void download_local_output(GPUState *state, float *h_output, int total_tokens) {
    CUDA_CHECK(cudaSetDevice(state->gpu_id));
    int hs = state->config.hidden_size;

    int offset = 0;
    for (int g = 0; g < state->gpu_id; g++) {
        int base = total_tokens / state->num_gpus;
        int rem = total_tokens % state->num_gpus;
        offset += base + (g < rem ? 1 : 0);
    }

    CUDA_CHECK(cudaMemcpy(&h_output[offset * hs], state->d_local_output,
                           state->local_num_tokens * hs * sizeof(float),
                           cudaMemcpyDeviceToHost));
}


/* ============================================================
 * Cleanup GPU State
 * ============================================================ */
void cleanup_gpu_state(GPUState *state) {
    CUDA_CHECK(cudaSetDevice(state->gpu_id));
    CUDA_CHECK(cudaFree(state->d_router_weight));
    CUDA_CHECK(cudaFree(state->d_e_score_correction_bias));
    CUDA_CHECK(cudaFree(state->d_shared_gate_proj_w));
    CUDA_CHECK(cudaFree(state->d_shared_up_proj_w));
    CUDA_CHECK(cudaFree(state->d_shared_down_proj_w));
    CUDA_CHECK(cudaFree(state->d_local_gate_up_proj));
    CUDA_CHECK(cudaFree(state->d_local_down_proj));
    CUDA_CHECK(cudaFree(state->d_local_input));
    CUDA_CHECK(cudaFree(state->d_local_output));
    CUDA_CHECK(cudaFree(state->d_router_logits));
    CUDA_CHECK(cudaFree(state->d_topk_indices));
    CUDA_CHECK(cudaFree(state->d_topk_weights));
    CUDA_CHECK(cudaStreamDestroy(state->stream));
}

#endif /* MOE_PARALLEL_CU_INCLUDED */
