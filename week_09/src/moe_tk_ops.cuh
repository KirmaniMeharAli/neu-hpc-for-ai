/*
 * moe_tk_ops.cuh - Element-wise operations for DeepSeekV3 MoE
 *
 * Week 9: Sequence Parallelism
 * Standard CUDA kernels for activation functions and element-wise ops.
 * These are memory-bound, not compute-bound, so tensor cores don't help.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

/* SiLU activation: silu(x) = x * sigmoid(x) */
__global__ void silu_bf16_kernel(__nv_bfloat16* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(data[idx]);
        float result = x / (1.0f + expf(-x));
        data[idx] = __float2bfloat16(result);
    }
}

inline void apply_silu_bf16(__nv_bfloat16* data, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_bf16_kernel<<<blocks, threads>>>(data, n);
}

/* Sigmoid: sigmoid(x) = 1 / (1 + exp(-x)) */
__global__ void sigmoid_f32_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
    }
}

inline void apply_sigmoid_f32(float* data, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sigmoid_f32_kernel<<<blocks, threads>>>(data, n);
}

/* Element-wise multiply: a[i] *= b[i] */
__global__ void emul_bf16_kernel(__nv_bfloat16* a, const __nv_bfloat16* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __bfloat162float(a[idx]);
        float vb = __bfloat162float(b[idx]);
        a[idx] = __float2bfloat16(va * vb);
    }
}

inline void apply_emul_bf16(__nv_bfloat16* a, const __nv_bfloat16* b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    emul_bf16_kernel<<<blocks, threads>>>(a, b, n);
}

/* Element-wise add: a[i] += b[i] */
__global__ void eadd_bf16_kernel(__nv_bfloat16* a, const __nv_bfloat16* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __bfloat162float(a[idx]);
        float vb = __bfloat162float(b[idx]);
        a[idx] = __float2bfloat16(va + vb);
    }
}

inline void apply_eadd_bf16(__nv_bfloat16* a, const __nv_bfloat16* b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    eadd_bf16_kernel<<<blocks, threads>>>(a, b, n);
}

/* Scale: a[i] *= scalar */
__global__ void scale_bf16_kernel(__nv_bfloat16* a, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = __bfloat162float(a[idx]);
        a[idx] = __float2bfloat16(v * scalar);
    }
}

inline void apply_scale_bf16(__nv_bfloat16* a, float scalar, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scale_bf16_kernel<<<blocks, threads>>>(a, scalar, n);
}

/* bf16 -> float conversion */
__global__ void bf16_to_f32_kernel(const __nv_bfloat16* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __bfloat162float(in[idx]);
    }
}

inline void bf16_to_f32(const __nv_bfloat16* in, float* out, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    bf16_to_f32_kernel<<<blocks, threads>>>(in, out, n);
}

/* float -> bf16 conversion */
__global__ void f32_to_bf16_kernel(const float* in, __nv_bfloat16* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(in[idx]);
    }
}

inline void f32_to_bf16(const float* in, __nv_bfloat16* out, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    f32_to_bf16_kernel<<<blocks, threads>>>(in, out, n);
}
