/*
 * benchmark.cu - DeepSeekV3 MoE Performance Benchmark
 *
 * Week 9: Sequence Parallelism
 * 
 * Compares three GEMM implementations used in the MoE operator:
 *   1. Naive CUDA (fp32, no tensor cores) - Week 7 equivalent
 *   2. WMMA Tensor Cores (bf16, using nvcuda::wmma)
 *   3. WMMA + Shared Memory Tiling (bf16, optimized)
 *
 * Then benchmarks the full MoE MLP: down(silu(gate(x)) * up(x))
 *
 * Build:
 *   nvcc -std=c++17 -O3 -arch=sm_90a -lcurand -o benchmark_moe benchmark.cu
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>

using namespace nvcuda;

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

/* ============================================================
 * Implementation 1: Naive CUDA GEMM (no tensor cores)
 * ============================================================
 * C = A @ B^T, all fp32. This is the Week 7 baseline equivalent.
 */
__global__ void naive_gemm_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = sum;
    }
}

void launch_naive_gemm(float* A, float* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    naive_gemm_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
}

/* ============================================================
 * Implementation 2: WMMA Tensor Core GEMM
 * ============================================================
 * Uses nvcuda::wmma for 16x16x16 matrix multiply-accumulate.
 * A: bf16 (M,K), B: bf16 (N,K) stored row-major, C: fp32 (M,N)
 *
 * Each warp computes one 16x16 output tile.
 * This directly uses the WMMA API that ThunderKittens wraps.
 */

// WMMA tile dimensions
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__global__ void wmma_gemm_kernel(
    const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
    int M, int N, int K)
{
    // Each warp handles one 16x16 output tile
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warpN = blockIdx.x;

    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Zero accumulator
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K dimension in chunks of WMMA_K
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int bRow = warpN * WMMA_N;

        // Load A tile (row-major): A[aRow:aRow+16, k:k+16]
        if (aRow < M && k < K) {
            wmma::load_matrix_sync(a_frag, A + aRow * K + k, K);
        }

        // Load B tile: B is (N,K) row-major, but we need it col-major for B^T
        // B^T[k:k+16, bRow:bRow+16] = B[bRow:bRow+16, k:k+16] loaded as col_major
        if (bRow < N && k < K) {
            wmma::load_matrix_sync(b_frag, B + bRow * K + k, K);
        }

        // Tensor core MMA: C += A * B^T
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

void launch_wmma_gemm(
    const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
    int M, int N, int K)
{
    // One warp per 16x16 output tile
    // blockDim.y = 32 means 1 warp per block row
    dim3 threads(1, 32);  // 1 warp
    dim3 blocks(
        (N + WMMA_N - 1) / WMMA_N,
        (M + WMMA_M - 1) / WMMA_M
    );
    wmma_gemm_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
}

/* ============================================================
 * Implementation 3: WMMA + Shared Memory Tiling
 * ============================================================
 * Adds shared memory to reduce global memory accesses.
 * Multiple warps per block, each handling different output tiles.
 * Tiles of A and B are loaded into shared memory cooperatively.
 */

constexpr int BLOCK_TILES_M = 4;  // tiles per block in M
constexpr int BLOCK_TILES_N = 4;  // tiles per block in N
constexpr int WARPS_PER_BLOCK = BLOCK_TILES_M * BLOCK_TILES_N; // 16

__global__ void wmma_tiled_gemm_kernel(
    const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
    int M, int N, int K)
{
    // Shared memory for tiles
    __shared__ __nv_bfloat16 sA[BLOCK_TILES_M * WMMA_M][WMMA_K];
    __shared__ __nv_bfloat16 sB[BLOCK_TILES_N * WMMA_N][WMMA_K];

    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int warpRow = warpId / BLOCK_TILES_N;
    int warpCol = warpId % BLOCK_TILES_N;

    int globalRowStart = blockIdx.y * BLOCK_TILES_M * WMMA_M;
    int globalColStart = blockIdx.x * BLOCK_TILES_N * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        // Cooperatively load A tile block into shared memory
        int totalA = BLOCK_TILES_M * WMMA_M * WMMA_K;
        for (int i = threadIdx.x; i < totalA; i += blockDim.x) {
            int r = i / WMMA_K;
            int c = i % WMMA_K;
            int gr = globalRowStart + r;
            int gc = k + c;
            sA[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2bfloat16(0.0f);
        }

        // Cooperatively load B tile block into shared memory
        int totalB = BLOCK_TILES_N * WMMA_N * WMMA_K;
        for (int i = threadIdx.x; i < totalB; i += blockDim.x) {
            int r = i / WMMA_K;
            int c = i % WMMA_K;
            int gr = globalColStart + r;
            int gc = k + c;
            sB[r][c] = (gr < N && gc < K) ? B[gr * K + gc] : __float2bfloat16(0.0f);
        }

        __syncthreads();

        // Each warp loads its tile from shared memory and does MMA
        int localRow = warpRow * WMMA_M;
        int localCol = warpCol * WMMA_N;

        wmma::load_matrix_sync(a_frag, &sA[localRow][0], WMMA_K);
        wmma::load_matrix_sync(b_frag, &sB[localCol][0], WMMA_K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // Store result
    int outRow = globalRowStart + warpRow * WMMA_M;
    int outCol = globalColStart + warpCol * WMMA_N;
    if (outRow < M && outCol < N) {
        wmma::store_matrix_sync(C + outRow * N + outCol, c_frag, N, wmma::mem_row_major);
    }
}

void launch_wmma_tiled_gemm(
    const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
    int M, int N, int K)
{
    dim3 threads(WARPS_PER_BLOCK * 32);
    dim3 blocks(
        (N + BLOCK_TILES_N * WMMA_N - 1) / (BLOCK_TILES_N * WMMA_N),
        (M + BLOCK_TILES_M * WMMA_M - 1) / (BLOCK_TILES_M * WMMA_M)
    );
    wmma_tiled_gemm_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
}


/* ============================================================
 * Element-wise kernels (SiLU, multiply)
 * ============================================================ */
__global__ void silu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        data[idx] = x / (1.0f + expf(-x));
    }
}

__global__ void emul_kernel(float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] *= b[idx];
    }
}

__global__ void eadd_kernel(float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] += b[idx];
    }
}

__global__ void f32_to_bf16_kernel(const float* in, __nv_bfloat16* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __float2bfloat16(in[idx]);
}

__global__ void bf16_to_f32_kernel(const __nv_bfloat16* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __bfloat162float(in[idx]);
}


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
 * Random init helpers
 * ============================================================ */
void rand_init_f32(float* d_ptr, int n) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateNormal(gen, d_ptr, n, 0.0f, 0.02f);
    curandDestroyGenerator(gen);
}

void rand_init_bf16(__nv_bfloat16* d_ptr, int n) {
    float* tmp;
    cudaMalloc(&tmp, n * sizeof(float));
    rand_init_f32(tmp, n);
    int blk = (n + 255) / 256;
    f32_to_bf16_kernel<<<blk, 256>>>(tmp, d_ptr, n);
    cudaFree(tmp);
}


/* ============================================================
 * Main Benchmark
 * ============================================================ */
int main() {
    printf("============================================================\n");
    printf("Week 9: DeepSeekV3 MoE Performance Benchmark\n");
    printf("WMMA Tensor Cores vs Naive CUDA Baseline\n");
    printf("============================================================\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Global Memory: %.1f GB\n\n", prop.totalGlobalMem / 1e9);

    int warmup = 3, iters = 20;

    // ============================================================
    // Benchmark 1: GEMM at various sizes
    // ============================================================
    printf("============================================================\n");
    printf("Benchmark 1: GEMM Performance (C = A @ B^T)\n");
    printf("============================================================\n");
    printf("%-6s  %-28s  %-10s  %-10s  %-8s\n",
           "N", "Implementation", "Time(ms)", "TFLOPs", "Speedup");
    printf("------  ----------------------------  ----------  ----------  --------\n");

    int sizes[] = {256, 512, 1024, 2048};

    for (int s = 0; s < 4; s++) {
        int N = sizes[s];
        long long flops = 2LL * N * N * N;

        float *d_A_f32, *d_B_f32, *d_C_f32;
        __nv_bfloat16 *d_A_bf, *d_B_bf;
        float *d_C_wmma;

        CHECK_CUDA(cudaMalloc(&d_A_f32, N*N*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B_f32, N*N*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C_f32, N*N*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_A_bf,  N*N*sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMalloc(&d_B_bf,  N*N*sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMalloc(&d_C_wmma, N*N*sizeof(float)));

        rand_init_f32(d_A_f32, N*N);
        rand_init_f32(d_B_f32, N*N);
        rand_init_bf16(d_A_bf, N*N);
        rand_init_bf16(d_B_bf, N*N);

        // Naive
        double t1 = time_kernel([&](){ launch_naive_gemm(d_A_f32, d_B_f32, d_C_f32, N, N, N); }, warmup, iters);
        double tf1 = flops / (t1 * 1e9);

        // WMMA
        double t2 = time_kernel([&](){ launch_wmma_gemm(d_A_bf, d_B_bf, d_C_wmma, N, N, N); }, warmup, iters);
        double tf2 = flops / (t2 * 1e9);

        // WMMA + Tiled shared memory
        double t3 = time_kernel([&](){ launch_wmma_tiled_gemm(d_A_bf, d_B_bf, d_C_wmma, N, N, N); }, warmup, iters);
        double tf3 = flops / (t3 * 1e9);

        printf("%-6d  %-28s  %-10.3f  %-10.2f  %-8s\n", N, "Naive CUDA (fp32)", t1, tf1, "1.00x");
        printf("%-6d  %-28s  %-10.3f  %-10.2f  %-8.2fx\n", N, "WMMA Tensor Cores (bf16)", t2, tf2, t1/t2);
        printf("%-6d  %-28s  %-10.3f  %-10.2f  %-8.2fx\n", N, "WMMA+Tiled Smem (bf16)", t3, tf3, t1/t3);
        printf("\n");

        cudaFree(d_A_f32); cudaFree(d_B_f32); cudaFree(d_C_f32);
        cudaFree(d_A_bf); cudaFree(d_B_bf); cudaFree(d_C_wmma);
    }

    // ============================================================
    // Benchmark 2: MoE MLP
    // ============================================================
    printf("============================================================\n");
    printf("Benchmark 2: MoE MLP: down(silu(gate(x)) * up(x))\n");
    printf("============================================================\n");
    printf("%-8s  %-6s  %-6s  %-28s  %-10s  %-8s\n",
           "Tokens", "H", "I", "Implementation", "Time(ms)", "Speedup");
    printf("--------  ------  ------  ----------------------------  ----------  --------\n");

    struct Config { int T, H, I; };
    Config cfgs[] = {{64,64,128}, {128,128,256}, {256,256,512}, {256,512,1024}};

    for (int c = 0; c < 4; c++) {
        int T = cfgs[c].T, H = cfgs[c].H, I = cfgs[c].I;

        // Allocate fp32 buffers for naive
        float *d_in_f, *d_gw_f, *d_uw_f, *d_dw_f, *d_out_f, *d_gate_f, *d_up_f;
        CHECK_CUDA(cudaMalloc(&d_in_f,   T*H*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_gw_f,   I*H*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_uw_f,   I*H*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dw_f,   H*I*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out_f,  T*H*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_gate_f, T*I*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_up_f,   T*I*sizeof(float)));

        rand_init_f32(d_in_f, T*H); rand_init_f32(d_gw_f, I*H);
        rand_init_f32(d_uw_f, I*H); rand_init_f32(d_dw_f, H*I);

        // Allocate bf16 + fp32 buffers for WMMA
        __nv_bfloat16 *d_in_bf, *d_gw_bf, *d_uw_bf, *d_dw_bf;
        float *d_out_w, *d_gate_w, *d_up_w;
        CHECK_CUDA(cudaMalloc(&d_in_bf,  T*H*sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMalloc(&d_gw_bf,  I*H*sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMalloc(&d_uw_bf,  I*H*sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMalloc(&d_dw_bf,  H*I*sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMalloc(&d_out_w,  T*H*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_gate_w, T*I*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_up_w,   T*I*sizeof(float)));

        rand_init_bf16(d_in_bf, T*H); rand_init_bf16(d_gw_bf, I*H);
        rand_init_bf16(d_uw_bf, I*H); rand_init_bf16(d_dw_bf, H*I);

        int blkTI = (T*I+255)/256, blkTH = (T*H+255)/256;

        // Naive MLP
        double t_naive = time_kernel([&](){
            launch_naive_gemm(d_in_f, d_gw_f, d_gate_f, T, I, H);
            launch_naive_gemm(d_in_f, d_uw_f, d_up_f, T, I, H);
            silu_kernel<<<blkTI, 256>>>(d_gate_f, T*I);
            emul_kernel<<<blkTI, 256>>>(d_gate_f, d_up_f, T*I);
            launch_naive_gemm(d_gate_f, d_dw_f, d_out_f, T, H, I);
        }, warmup, iters);

        // WMMA MLP
        double t_wmma = time_kernel([&](){
            launch_wmma_gemm(d_in_bf, d_gw_bf, d_gate_w, T, I, H);
            launch_wmma_gemm(d_in_bf, d_uw_bf, d_up_w, T, I, H);
            silu_kernel<<<blkTI, 256>>>(d_gate_w, T*I);
            emul_kernel<<<blkTI, 256>>>(d_gate_w, d_up_w, T*I);
            // For down_proj, need bf16 input - convert gate_w to bf16 temp
            __nv_bfloat16* tmp_bf;
            cudaMalloc(&tmp_bf, T*I*sizeof(__nv_bfloat16));
            f32_to_bf16_kernel<<<blkTI, 256>>>(d_gate_w, tmp_bf, T*I);
            launch_wmma_gemm(tmp_bf, d_dw_bf, d_out_w, T, H, I);
            cudaFree(tmp_bf);
        }, warmup, iters);

        // WMMA + Tiled MLP
        double t_tiled = time_kernel([&](){
            launch_wmma_tiled_gemm(d_in_bf, d_gw_bf, d_gate_w, T, I, H);
            launch_wmma_tiled_gemm(d_in_bf, d_uw_bf, d_up_w, T, I, H);
            silu_kernel<<<blkTI, 256>>>(d_gate_w, T*I);
            emul_kernel<<<blkTI, 256>>>(d_gate_w, d_up_w, T*I);
            __nv_bfloat16* tmp_bf;
            cudaMalloc(&tmp_bf, T*I*sizeof(__nv_bfloat16));
            f32_to_bf16_kernel<<<blkTI, 256>>>(d_gate_w, tmp_bf, T*I);
            launch_wmma_tiled_gemm(tmp_bf, d_dw_bf, d_out_w, T, H, I);
            cudaFree(tmp_bf);
        }, warmup, iters);

        printf("%-8d  %-6d  %-6d  %-28s  %-10.3f  %-8s\n", T, H, I, "Naive CUDA (fp32)", t_naive, "1.00x");
        printf("%-8d  %-6d  %-6d  %-28s  %-10.3f  %-8.2fx\n", T, H, I, "WMMA Tensor Cores (bf16)", t_wmma, t_naive/t_wmma);
        printf("%-8d  %-6d  %-6d  %-28s  %-10.3f  %-8.2fx\n", T, H, I, "WMMA+Tiled Smem (bf16)", t_tiled, t_naive/t_tiled);
        printf("\n");

        cudaFree(d_in_f); cudaFree(d_gw_f); cudaFree(d_uw_f);
        cudaFree(d_dw_f); cudaFree(d_out_f); cudaFree(d_gate_f); cudaFree(d_up_f);
        cudaFree(d_in_bf); cudaFree(d_gw_bf); cudaFree(d_uw_bf);
        cudaFree(d_dw_bf); cudaFree(d_out_w); cudaFree(d_gate_w); cudaFree(d_up_w);
    }

    printf("============================================================\n");
    printf("Benchmark complete.\n");
    printf("============================================================\n");

    return 0;
}
