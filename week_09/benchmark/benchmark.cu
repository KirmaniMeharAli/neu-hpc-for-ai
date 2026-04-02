/*
 * benchmark.cu - Performance Benchmark for DeepSeekV3 MoE
 *
 * Week 9: Sequence Parallelism
 *
 * Compares three implementations:
 *   1. Naive CUDA GEMM (no tensor cores) - Week 7 equivalent
 *   2. TK WMMA Tensor Cores (Level 04)
 *   3. TK WMMA + TMA (Level 05)
 *
 * Measures: latency (ms), throughput (TFLOPs), speedup vs baseline
 */

#include "moe_tk_gemm.cuh"
#include "moe_tk_ops.cuh"
#include "moe_tk.cuh"
#include "../baseline/baseline_gemm.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>
#include <curand.h>

/* ============================================================
 * Timing utility
 * ============================================================ */
template<typename Func>
double time_kernel(Func fn, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) fn();
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) fn();
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / iters;
}

/* ============================================================
 * Random initialization
 * ============================================================ */
void rand_init_bf16(__nv_bfloat16* d_ptr, int n) {
    // Allocate fp32 temp, fill with curand, convert to bf16
    float* d_tmp;
    cudaMalloc(&d_tmp, n * sizeof(float));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateNormal(gen, d_tmp, n, 0.0f, 0.02f);
    curandDestroyGenerator(gen);

    f32_to_bf16(d_tmp, d_ptr, n);
    cudaFree(d_tmp);
}

void rand_init_f32(float* d_ptr, int n) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateNormal(gen, d_ptr, n, 0.0f, 0.02f);
    curandDestroyGenerator(gen);
}

/* ============================================================
 * Main
 * ============================================================ */
int main() {
    printf("============================================================\n");
    printf("Week 9: DeepSeekV3 MoE Performance Benchmark\n");
    printf("ThunderKittens WMMA + TMA vs Baseline\n");
    printf("============================================================\n\n");

    // GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Shared Memory/SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Global Memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("\n");

    int warmup = 5;
    int iters = 20;

    // ============================================================
    // Benchmark 1: GEMM at various sizes
    // ============================================================
    printf("============================================================\n");
    printf("Benchmark 1: GEMM Performance (C = A @ B^T)\n");
    printf("============================================================\n");
    printf("%-8s  %-20s  %-12s  %-12s  %-10s\n",
           "N", "Implementation", "Time (ms)", "TFLOPs", "Speedup");
    printf("--------  --------------------  ------------  ------------  ----------\n");

    int gemm_sizes[] = {128, 256, 512, 1024, 2048};
    int num_gemm = 5;

    for (int s = 0; s < num_gemm; s++) {
        int N = gemm_sizes[s];
        long long flops = 2LL * N * N * N;

        // Allocate
        __nv_bfloat16 *d_A_bf, *d_B_bf, *d_C_bf;
        float *d_A_f32, *d_B_f32, *d_C_f32;

        cudaMalloc(&d_A_bf, N * N * sizeof(__nv_bfloat16));
        cudaMalloc(&d_B_bf, N * N * sizeof(__nv_bfloat16));
        cudaMalloc(&d_C_bf, N * N * sizeof(__nv_bfloat16));
        cudaMalloc(&d_A_f32, N * N * sizeof(float));
        cudaMalloc(&d_B_f32, N * N * sizeof(float));
        cudaMalloc(&d_C_f32, N * N * sizeof(float));

        rand_init_bf16(d_A_bf, N * N);
        rand_init_bf16(d_B_bf, N * N);
        rand_init_f32(d_A_f32, N * N);
        rand_init_f32(d_B_f32, N * N);

        // 1. Naive CUDA (fp32, no tensor cores)
        double t_naive = time_kernel(
            [&]() { launch_naive_gemm_f32(d_A_f32, d_B_f32, d_C_f32, N, N, N); },
            warmup, iters);
        double tflops_naive = flops / (t_naive * 1e9);

        // 2. TK WMMA (bf16, tensor cores)
        double t_wmma = time_kernel(
            [&]() { launch_gemm_wmma(d_A_bf, d_B_bf, d_C_bf, N, N, N); },
            warmup, iters);
        double tflops_wmma = flops / (t_wmma * 1e9);

        // 3. TK WMMA + TMA (bf16, tensor cores + TMA)
        double t_tma = -1, tflops_tma = 0;
        // TMA version requires dimensions to be multiples of tile sizes
        if (N % TILE_M == 0 && N % TILE_N == 0 && N % TILE_K == 0) {
            t_tma = time_kernel(
                [&]() { launch_gemm_wmma_tma(d_A_bf, d_B_bf, d_C_bf, N, N, N); },
                warmup, iters);
            tflops_tma = flops / (t_tma * 1e9);
        }

        printf("%-8d  %-20s  %-12.3f  %-12.2f  %-10s\n",
               N, "Naive CUDA (fp32)", t_naive, tflops_naive, "1.00x");
        printf("%-8d  %-20s  %-12.3f  %-12.2f  %-10.2fx\n",
               N, "TK WMMA (bf16)", t_wmma, tflops_wmma, t_naive / t_wmma);
        if (t_tma > 0) {
            printf("%-8d  %-20s  %-12.3f  %-12.2f  %-10.2fx\n",
                   N, "TK WMMA+TMA (bf16)", t_tma, tflops_tma, t_naive / t_tma);
        }
        printf("\n");

        cudaFree(d_A_bf); cudaFree(d_B_bf); cudaFree(d_C_bf);
        cudaFree(d_A_f32); cudaFree(d_B_f32); cudaFree(d_C_f32);
    }

    // ============================================================
    // Benchmark 2: MoE MLP at various sizes
    // ============================================================
    printf("============================================================\n");
    printf("Benchmark 2: MoE MLP Performance\n");
    printf("output = down_proj(silu(gate_proj(x)) * up_proj(x))\n");
    printf("============================================================\n");
    printf("%-10s  %-8s  %-8s  %-20s  %-12s  %-10s\n",
           "Tokens", "Hidden", "Inter", "Implementation", "Time (ms)", "Speedup");
    printf("----------  --------  --------  --------------------  ------------  ----------\n");

    struct MLP_Config { int tokens; int hidden; int inter; };
    MLP_Config mlp_configs[] = {
        {64,  64,  128},
        {64,  128, 256},
        {128, 128, 256},
        {128, 256, 512},
        {256, 256, 512},
        {256, 512, 1024},
    };
    int num_mlp = 6;

    for (int c = 0; c < num_mlp; c++) {
        int T = mlp_configs[c].tokens;
        int H = mlp_configs[c].hidden;
        int I = mlp_configs[c].inter;

        // Allocate
        __nv_bfloat16 *d_input, *d_gate_w, *d_up_w, *d_down_w;
        __nv_bfloat16 *d_output, *d_gate_buf, *d_up_buf;
        __nv_bfloat16 *d_output_baseline;

        cudaMalloc(&d_input,    T * H * sizeof(__nv_bfloat16));
        cudaMalloc(&d_gate_w,   I * H * sizeof(__nv_bfloat16));
        cudaMalloc(&d_up_w,     I * H * sizeof(__nv_bfloat16));
        cudaMalloc(&d_down_w,   H * I * sizeof(__nv_bfloat16));
        cudaMalloc(&d_output,   T * H * sizeof(__nv_bfloat16));
        cudaMalloc(&d_gate_buf, T * I * sizeof(__nv_bfloat16));
        cudaMalloc(&d_up_buf,   T * I * sizeof(__nv_bfloat16));
        cudaMalloc(&d_output_baseline, T * H * sizeof(__nv_bfloat16));

        rand_init_bf16(d_input, T * H);
        rand_init_bf16(d_gate_w, I * H);
        rand_init_bf16(d_up_w, I * H);
        rand_init_bf16(d_down_w, H * I);

        // Naive baseline MLP
        double t_base = time_kernel([&]() {
            launch_naive_gemm_bf16(d_input, d_gate_w, d_gate_buf, T, I, H);
            launch_naive_gemm_bf16(d_input, d_up_w, d_up_buf, T, I, H);
            apply_silu_bf16(d_gate_buf, T * I);
            apply_emul_bf16(d_gate_buf, d_up_buf, T * I);
            launch_naive_gemm_bf16(d_gate_buf, d_down_w, d_output_baseline, T, H, I);
        }, warmup, iters);

        // TK WMMA MLP
        double t_wmma = time_kernel([&]() {
            moe_mlp_tk(d_input, d_gate_w, d_up_w, d_down_w, d_output,
                       d_gate_buf, d_up_buf, T, H, I, false);
        }, warmup, iters);

        // TK WMMA+TMA MLP (only if dimensions align)
        double t_tma = -1;
        bool tma_ok = (T % TILE_M == 0) && (H % TILE_K == 0) &&
                      (I % TILE_N == 0) && (I % TILE_K == 0);
        if (tma_ok) {
            t_tma = time_kernel([&]() {
                moe_mlp_tk(d_input, d_gate_w, d_up_w, d_down_w, d_output,
                           d_gate_buf, d_up_buf, T, H, I, true);
            }, warmup, iters);
        }

        printf("%-10d  %-8d  %-8d  %-20s  %-12.3f  %-10s\n",
               T, H, I, "Naive CUDA (bf16)", t_base, "1.00x");
        printf("%-10d  %-8d  %-8d  %-20s  %-12.3f  %-10.2fx\n",
               T, H, I, "TK WMMA (bf16)", t_wmma, t_base / t_wmma);
        if (t_tma > 0) {
            printf("%-10d  %-8d  %-8d  %-20s  %-12.3f  %-10.2fx\n",
                   T, H, I, "TK WMMA+TMA (bf16)", t_tma, t_base / t_tma);
        }
        printf("\n");

        cudaFree(d_input); cudaFree(d_gate_w); cudaFree(d_up_w);
        cudaFree(d_down_w); cudaFree(d_output); cudaFree(d_gate_buf);
        cudaFree(d_up_buf); cudaFree(d_output_baseline);
    }

    printf("============================================================\n");
    printf("Benchmark complete.\n");
    printf("============================================================\n");

    return 0;
}
