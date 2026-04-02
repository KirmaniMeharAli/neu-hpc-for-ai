/*
 * benchmark.cu - DeepSeekV3 MoE Performance Benchmark
 *
 * Week 9: Sequence Parallelism
 *
 * Compares GEMM implementations used in the MoE operator:
 *   1. Naive CUDA (fp32, no tensor cores) - Week 7 baseline
 *   2. WMMA Tensor Cores (bf16, nvcuda::wmma)
 *   3. WMMA + TMA (bf16, async global->shared via cuda::memcpy_async + tensor cores)
 *
 * TMA (Tensor Memory Accelerator):
 *   Uses cuda::memcpy_async with cuda::pipeline to asynchronously copy data
 *   from global memory to shared memory. On Hopper (H100) and Blackwell (B200),
 *   this maps to the hardware TMA unit, handling address calculation and data
 *   movement without occupying compute threads.
 *
 * Build for H100:  nvcc -std=c++17 -O3 -arch=sm_90a -lcurand -o benchmark_moe benchmark.cu
 * Build for B200:  nvcc -std=c++17 -O3 -arch=sm_100a -lcurand -o benchmark_moe benchmark.cu
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>

using namespace nvcuda;
namespace cg = cooperative_groups;

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
 * ============================================================ */
__global__ void naive_gemm_kernel(
    const float* A, const float* B, float* C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[col * K + k];
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
 * ============================================================ */
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__global__ void wmma_gemm_kernel(
    const __nv_bfloat16* A, const __nv_bfloat16* B, float* C, int M, int N, int K)
{
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warpN = blockIdx.x;
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        if (warpM * WMMA_M < M && k < K)
            wmma::load_matrix_sync(a_frag, A + warpM * WMMA_M * K + k, K);
        if (warpN * WMMA_N < N && k < K)
            wmma::load_matrix_sync(b_frag, B + warpN * WMMA_N * K + k, K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    int cRow = warpM * WMMA_M, cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N)
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
}

void launch_wmma_gemm(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
                       int M, int N, int K) {
    dim3 threads(1, 32);
    dim3 blocks((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    wmma_gemm_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
}

/* ============================================================
 * Implementation 3: WMMA + TMA (cuda::memcpy_async + cuda::pipeline)
 * ============================================================
 * Uses cuda::memcpy_async with cuda::pipeline for async global->shared
 * memory transfers. On Hopper/Blackwell, this maps to hardware TMA.
 * Pattern: async prefetch tile into shared, barrier wait, then WMMA MMA.
 */
constexpr int TMA_BM = 64, TMA_BN = 64, TMA_BK = 16;
constexpr int TMA_WM = TMA_BM / WMMA_M;  // 4
constexpr int TMA_WN = TMA_BN / WMMA_N;  // 4
constexpr int TMA_NWARPS = TMA_WM * TMA_WN; // 16

__global__ void wmma_tma_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Double-buffered shared memory for pipelining
    __shared__ __nv_bfloat16 sA[2][TMA_BM][TMA_BK];
    __shared__ __nv_bfloat16 sB[2][TMA_BN][TMA_BK];

    int warpId = threadIdx.x / 32;
    int warpRow = warpId / TMA_WN;
    int warpCol = warpId % TMA_WN;
    int gRowStart = blockIdx.y * TMA_BM;
    int gColStart = blockIdx.x * TMA_BN;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // Create a pipeline with 2 stages for double buffering
    auto group = cg::this_thread_block();
    constexpr int stages = 2;
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    int numTiles = (K + TMA_BK - 1) / TMA_BK;

    for (int tk = 0; tk < numTiles; tk++) {
        int buf = tk % stages;
        int kOffset = tk * TMA_BK;

        // TMA Stage: async copy A tile from global to shared memory
        pipe.producer_acquire();
        for (int i = threadIdx.x; i < TMA_BM * TMA_BK; i += blockDim.x) {
            int r = i / TMA_BK, c = i % TMA_BK;
            int gr = gRowStart + r, gc = kOffset + c;
            if (gr < M && gc < K) {
                cuda::memcpy_async(&sA[buf][r][c], &A[gr * K + gc],
                                   sizeof(__nv_bfloat16), pipe);
            } else {
                sA[buf][r][c] = __float2bfloat16(0.0f);
            }
        }
        // TMA Stage: async copy B tile from global to shared memory
        for (int i = threadIdx.x; i < TMA_BN * TMA_BK; i += blockDim.x) {
            int r = i / TMA_BK, c = i % TMA_BK;
            int gr = gColStart + r, gc = kOffset + c;
            if (gr < N && gc < K) {
                cuda::memcpy_async(&sB[buf][r][c], &B[gr * K + gc],
                                   sizeof(__nv_bfloat16), pipe);
            } else {
                sB[buf][r][c] = __float2bfloat16(0.0f);
            }
        }
        pipe.producer_commit();

        // Wait for this stage's async copies to complete
        pipe.consumer_wait();
        __syncthreads();

        // WMMA: tensor core MMA on the prefetched tiles
        wmma::load_matrix_sync(a_frag, &sA[buf][warpRow * WMMA_M][0], TMA_BK);
        wmma::load_matrix_sync(b_frag, &sB[buf][warpCol * WMMA_N][0], TMA_BK);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        pipe.consumer_release();
        __syncthreads();
    }

    int outRow = gRowStart + warpRow * WMMA_M;
    int outCol = gColStart + warpCol * WMMA_N;
    if (outRow < M && outCol < N)
        wmma::store_matrix_sync(C + outRow * N + outCol, c_frag, N, wmma::mem_row_major);
}

void launch_wmma_tma_gemm(const __nv_bfloat16* A, const __nv_bfloat16* B,
                            float* C, int M, int N, int K) {
    dim3 threads(TMA_NWARPS * 32);
    dim3 blocks((N + TMA_BN - 1) / TMA_BN, (M + TMA_BM - 1) / TMA_BM);
    wmma_tma_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
}


/* ============================================================
 * Helpers
 * ============================================================ */
__global__ void silu_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = data[i]; data[i] = x / (1.0f + expf(-x)); }
}
__global__ void emul_kernel(float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= b[i];
}
__global__ void f32_to_bf16_kernel(const float* in, __nv_bfloat16* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in[i]);
}

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

void rand_init_f32(float* p, int n) {
    curandGenerator_t g; curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(g, 42); curandGenerateNormal(g, p, n, 0, .02f);
    curandDestroyGenerator(g);
}
void rand_init_bf16(__nv_bfloat16* p, int n) {
    float* t; cudaMalloc(&t, n*4); rand_init_f32(t, n);
    f32_to_bf16_kernel<<<(n+255)/256, 256>>>(t, p, n); cudaFree(t);
}

/* ============================================================
 * Main
 * ============================================================ */
int main() {
    printf("============================================================\n");
    printf("Week 9: DeepSeekV3 MoE Performance Benchmark\n");
    printf("WMMA Tensor Cores + TMA vs Naive CUDA Baseline\n");
    printf("============================================================\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Global Memory: %.1f GB\n\n", prop.totalGlobalMem / 1e9);

    int W = 3, R = 20;

    // ---- GEMM Benchmark ----
    printf("============================================================\n");
    printf("Benchmark 1: GEMM Performance (C = A @ B^T)\n");
    printf("============================================================\n");
    printf("%-6s  %-28s  %-10s  %-10s  %-8s\n", "N", "Implementation", "Time(ms)", "TFLOPs", "Speedup");
    printf("------  ----------------------------  ----------  ----------  --------\n");

    int sizes[] = {256, 512, 1024, 2048};
    for (int s = 0; s < 4; s++) {
        int N = sizes[s]; long long fl = 2LL*N*N*N;
        float *dAf, *dBf, *dCf; __nv_bfloat16 *dAb, *dBb; float *dCw, *dCt;
        cudaMalloc(&dAf,N*N*4); cudaMalloc(&dBf,N*N*4); cudaMalloc(&dCf,N*N*4);
        cudaMalloc(&dAb,N*N*2); cudaMalloc(&dBb,N*N*2); cudaMalloc(&dCw,N*N*4); cudaMalloc(&dCt,N*N*4);
        rand_init_f32(dAf,N*N); rand_init_f32(dBf,N*N); rand_init_bf16(dAb,N*N); rand_init_bf16(dBb,N*N);

        double t1=time_kernel([&](){launch_naive_gemm(dAf,dBf,dCf,N,N,N);},W,R);
        double t2=time_kernel([&](){launch_wmma_gemm(dAb,dBb,dCw,N,N,N);},W,R);
        double t3=time_kernel([&](){launch_wmma_tma_gemm(dAb,dBb,dCt,N,N,N);},W,R);

        printf("%-6d  %-28s  %-10.3f  %-10.2f  %-8s\n",   N,"Naive CUDA (fp32)",t1,fl/(t1*1e9),"1.00x");
        printf("%-6d  %-28s  %-10.3f  %-10.2f  %-8.2fx\n", N,"WMMA Tensor Cores (bf16)",t2,fl/(t2*1e9),t1/t2);
        printf("%-6d  %-28s  %-10.3f  %-10.2f  %-8.2fx\n", N,"WMMA + TMA (bf16)",t3,fl/(t3*1e9),t1/t3);
        printf("\n");
        cudaFree(dAf);cudaFree(dBf);cudaFree(dCf);cudaFree(dAb);cudaFree(dBb);cudaFree(dCw);cudaFree(dCt);
    }

    // ---- MoE MLP Benchmark ----
    printf("============================================================\n");
    printf("Benchmark 2: MoE MLP: down(silu(gate(x)) * up(x))\n");
    printf("============================================================\n");
    printf("%-8s  %-6s  %-6s  %-28s  %-10s  %-8s\n", "Tokens","H","I","Implementation","Time(ms)","Speedup");
    printf("--------  ------  ------  ----------------------------  ----------  --------\n");

    struct Cfg{int T,H,I;};
    Cfg cs[]={{64,64,128},{128,128,256},{256,256,512},{256,512,1024}};
    for(int c=0;c<4;c++){
        int T=cs[c].T,H=cs[c].H,I=cs[c].I;
        float*dif,*dgf,*duf,*ddf,*dof,*dgb,*dub;
        cudaMalloc(&dif,T*H*4);cudaMalloc(&dgf,I*H*4);cudaMalloc(&duf,I*H*4);
        cudaMalloc(&ddf,H*I*4);cudaMalloc(&dof,T*H*4);cudaMalloc(&dgb,T*I*4);cudaMalloc(&dub,T*I*4);
        rand_init_f32(dif,T*H);rand_init_f32(dgf,I*H);rand_init_f32(duf,I*H);rand_init_f32(ddf,H*I);

        __nv_bfloat16*dib,*dgwb,*duwb,*ddwb; float*dow,*dgw,*duw;
        cudaMalloc(&dib,T*H*2);cudaMalloc(&dgwb,I*H*2);cudaMalloc(&duwb,I*H*2);cudaMalloc(&ddwb,H*I*2);
        cudaMalloc(&dow,T*H*4);cudaMalloc(&dgw,T*I*4);cudaMalloc(&duw,T*I*4);
        rand_init_bf16(dib,T*H);rand_init_bf16(dgwb,I*H);rand_init_bf16(duwb,I*H);rand_init_bf16(ddwb,H*I);
        int bk=(T*I+255)/256;

        double tn=time_kernel([&](){
            launch_naive_gemm(dif,dgf,dgb,T,I,H);launch_naive_gemm(dif,duf,dub,T,I,H);
            silu_kernel<<<bk,256>>>(dgb,T*I);emul_kernel<<<bk,256>>>(dgb,dub,T*I);
            launch_naive_gemm(dgb,ddf,dof,T,H,I);},W,R);
        double tw=time_kernel([&](){
            launch_wmma_gemm(dib,dgwb,dgw,T,I,H);launch_wmma_gemm(dib,duwb,duw,T,I,H);
            silu_kernel<<<bk,256>>>(dgw,T*I);emul_kernel<<<bk,256>>>(dgw,duw,T*I);
            __nv_bfloat16*tb;cudaMalloc(&tb,T*I*2);f32_to_bf16_kernel<<<bk,256>>>(dgw,tb,T*I);
            launch_wmma_gemm(tb,ddwb,dow,T,H,I);cudaFree(tb);},W,R);
        double tt=time_kernel([&](){
            launch_wmma_tma_gemm(dib,dgwb,dgw,T,I,H);launch_wmma_tma_gemm(dib,duwb,duw,T,I,H);
            silu_kernel<<<bk,256>>>(dgw,T*I);emul_kernel<<<bk,256>>>(dgw,duw,T*I);
            __nv_bfloat16*tb;cudaMalloc(&tb,T*I*2);f32_to_bf16_kernel<<<bk,256>>>(dgw,tb,T*I);
            launch_wmma_tma_gemm(tb,ddwb,dow,T,H,I);cudaFree(tb);},W,R);

        printf("%-8d  %-6d  %-6d  %-28s  %-10.3f  %-8s\n",   T,H,I,"Naive CUDA (fp32)",tn,"1.00x");
        printf("%-8d  %-6d  %-6d  %-28s  %-10.3f  %-8.2fx\n", T,H,I,"WMMA Tensor Cores (bf16)",tw,tn/tw);
        printf("%-8d  %-6d  %-6d  %-28s  %-10.3f  %-8.2fx\n", T,H,I,"WMMA + TMA (bf16)",tt,tn/tt);
        printf("\n");
        cudaFree(dif);cudaFree(dgf);cudaFree(duf);cudaFree(ddf);cudaFree(dof);cudaFree(dgb);cudaFree(dub);
        cudaFree(dib);cudaFree(dgwb);cudaFree(duwb);cudaFree(ddwb);cudaFree(dow);cudaFree(dgw);cudaFree(duw);
    }

    printf("============================================================\n");
    printf("Benchmark complete.\n");
    printf("============================================================\n");
    return 0;
}
