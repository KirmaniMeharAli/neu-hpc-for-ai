/*
 * baseline_gemm.cuh - Naive CUDA GEMM baseline (no tensor cores)
 *
 * Week 9: Sequence Parallelism
 * Used as the baseline for performance comparison against TK.
 * This represents the pure-CUDA equivalent of Week 7's C implementation.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

/* Naive GEMM: C = A @ B^T, all fp32, no tensor cores */
__global__ void naive_gemm_f32_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[col * K + k]; // B^T
        }
        C[row * N + col] = sum;
    }
}

inline void launch_naive_gemm_f32(
    float* A, float* B, float* C, int M, int N, int K)
{
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    naive_gemm_f32_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
}

/* Naive GEMM with bf16 inputs but fp32 accumulation */
__global__ void naive_gemm_bf16_kernel(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = __bfloat162float(A[row * K + k]);
            float b = __bfloat162float(B[col * K + k]);
            sum += a * b;
        }
        C[row * N + col] = __float2bfloat16(sum);
    }
}

inline void launch_naive_gemm_bf16(
    __nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K)
{
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    naive_gemm_bf16_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
}

/* Naive MLP baseline (bf16, no tensor cores) */
inline void baseline_mlp(
    __nv_bfloat16* input,
    __nv_bfloat16* gate_proj_w,
    __nv_bfloat16* up_proj_w,
    __nv_bfloat16* down_proj_w,
    __nv_bfloat16* output,
    __nv_bfloat16* gate_buf,
    __nv_bfloat16* up_buf,
    int num_tokens,
    int hidden_size,
    int intermediate_size)
{
    launch_naive_gemm_bf16(input, gate_proj_w, gate_buf,
                            num_tokens, intermediate_size, hidden_size);
    launch_naive_gemm_bf16(input, up_proj_w, up_buf,
                            num_tokens, intermediate_size, hidden_size);

    // SiLU and multiply use the same kernels from moe_tk_ops.cuh
    // (element-wise ops aren't the bottleneck)
    extern void apply_silu_bf16(__nv_bfloat16*, int);
    extern void apply_emul_bf16(__nv_bfloat16*, const __nv_bfloat16*, int);

    apply_silu_bf16(gate_buf, num_tokens * intermediate_size);
    apply_emul_bf16(gate_buf, up_buf, num_tokens * intermediate_size);

    launch_naive_gemm_bf16(gate_buf, down_proj_w, output,
                            num_tokens, hidden_size, intermediate_size);
}
