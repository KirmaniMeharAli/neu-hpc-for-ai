/*
 * moe_tk_gemm.cuh - DeepSeekV3 MoE GEMM kernels with ThunderKittens
 *
 * Week 9: Sequence Parallelism
 * 
 * Implements matrix multiplication using:
 *   Level 04: WMMA tensor cores (Hopper/Blackwell)
 *   Level 05: WMMA + TMA (Tensor Memory Accelerator)
 *
 * These are used by all MoE blocks (MLP, Router, Expert dispatch)
 * since they all rely on F.linear(input, weight) = input @ weight^T.
 *
 * Reference: ThunderKittens kernels/gemm/educational_b200/
 */

#pragma once

#include "kittens.cuh"
using namespace kittens;

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

/* ============================================================
 * Level 04: GEMM with WMMA Tensor Cores
 * ============================================================
 * Uses warp-level matrix multiply-accumulate (MMA) via TK.
 * Loads tiles through shared memory using warp::load.
 *
 * Performance reference: ~26 TFLOPs on B200
 */

static constexpr int BLOCK_SIZE = 32;
static constexpr int NUM_WORKERS = 1;
static constexpr int NUM_THREADS = NUM_WORKERS * WARP_THREADS;

struct matmul_globals_wmma {
    using sub_tile = st_bf<BLOCK_SIZE, BLOCK_SIZE>;
    using tile_gl = gl<bf16, 1, 1, -1, -1, st_bf<BLOCK_SIZE, BLOCK_SIZE>>;
    
    tile_gl A;
    tile_gl B;
    tile_gl C;
    int K; // inner dimension
};

__global__ void gemm_wmma_kernel(const __grid_constant__ matmul_globals_wmma g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, BLOCK_SIZE> &As = al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE>>();
    st_bf<BLOCK_SIZE, BLOCK_SIZE> &Bs = al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE>>();

    rt_bf<BLOCK_SIZE, BLOCK_SIZE> A_reg;
    rt_bf<BLOCK_SIZE, BLOCK_SIZE> B_reg;
    rt_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::rt_layout::col> B_reg_col;
    rt_fl<BLOCK_SIZE, BLOCK_SIZE> C_accum;

    int col = blockIdx.x;
    int row = blockIdx.y;

    warp::zero(C_accum);
    int num_tiles = (g.K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile) {
        warp::load(As, g.A, {0, 0, row, tile});
        warp::load(Bs, g.B, {0, 0, tile, col});
        __syncthreads();
        warp::load(A_reg, As);
        warp::load(B_reg, Bs);
        warp::swap_layout(B_reg_col, B_reg);
        __syncthreads();
        warp::mma_AB(C_accum, A_reg, B_reg_col, C_accum);
        __syncthreads();
    }
    warp::store(g.C, C_accum, {0, 0, row, col});
}

/*
 * Launch WMMA-only GEMM: C = A @ B^T
 * A: (M, K) row-major, B: (N, K) row-major, C: (M, N) row-major
 * All bf16.
 */
inline void launch_gemm_wmma(
    __nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K)
{
    using tile_gl = matmul_globals_wmma::tile_gl;
    size_t sM = (size_t)M, sN = (size_t)N, sK = (size_t)K;
    
    // A is (M, K), indexed as tiles of (BLOCK_SIZE, BLOCK_SIZE)
    tile_gl a_arg{(bf16*)A, nullptr, nullptr, sM, sK};
    // B is transposed in the matmul: we store B as (N, K), and the inner loop
    // iterates over K dimension. For C = A @ B^T, B_layout should be (K, N)
    // so that tile (tile, col) gives the right K-slice × N-slice.
    tile_gl b_arg{(bf16*)B, nullptr, nullptr, sK, sN};
    tile_gl c_arg{(bf16*)C, nullptr, nullptr, sM, sN};
    
    matmul_globals_wmma g{a_arg, b_arg, c_arg, K};

    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    unsigned long mem_size = 100000;
    cudaFuncSetAttribute(gemm_wmma_kernel,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    gemm_wmma_kernel<<<blocks, NUM_THREADS, mem_size>>>(g);
    CHECK_CUDA(cudaGetLastError());
}


/* ============================================================
 * Level 05: GEMM with WMMA + TMA
 * ============================================================
 * Adds TMA (Tensor Memory Accelerator) for async global<->shared
 * memory transfers. The TMA hardware handles address calculation
 * and data movement without occupying any threads.
 *
 * Performance reference: ~55 TFLOPs on B200
 */

static constexpr int TILE_M = 64;
static constexpr int TILE_N = 64;
static constexpr int TILE_K = 32;

using a_tile_tma = st_bf<TILE_M, TILE_K>;
using b_tile_tma = st_bf<TILE_K, TILE_N>;
using d_tile_tma = st_bf<TILE_M, TILE_N>;

using a_gl_tma = gl<bf16, 1, 1, -1, -1, a_tile_tma>;
using b_gl_tma = gl<bf16, 1, 1, -1, -1, b_tile_tma>;
using d_gl_tma = gl<bf16, 1, 1, -1, -1, d_tile_tma>;

__global__
__launch_bounds__(WARP_THREADS, 1)
void gemm_wmma_tma_kernel(
    const __grid_constant__ a_gl_tma A_layout,
    const __grid_constant__ b_gl_tma B_layout,
    const __grid_constant__ d_gl_tma D_layout,
    int K_dim
) {
    int col = blockIdx.x;
    int row = blockIdx.y;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    a_tile_tma &As = al.allocate<a_tile_tma>();
    b_tile_tma &Bs = al.allocate<b_tile_tma>();
    d_tile_tma &Ds = al.allocate<d_tile_tma>();

    __shared__ semaphore smem_arrived;
    if (threadIdx.x == 0) {
        init_semaphore(smem_arrived, 0, 1);
    }
    __syncthreads();

    rt_bf<TILE_M, TILE_K> A_reg;
    rt_bf<TILE_K, TILE_N> B_reg;
    rt_bf<TILE_K, TILE_N, ducks::rt_layout::col> B_reg_col;
    rt_fl<TILE_M, TILE_N> C_accum;

    warp::zero(C_accum);
    int num_tiles = K_dim / TILE_K;
    int phase = 0;

    for (int tile = 0; tile < num_tiles; ++tile) {
        // TMA: async load A and B tiles from global to shared
        if (threadIdx.x == 0) {
            tma::expect_bytes(smem_arrived, sizeof(a_tile_tma) + sizeof(b_tile_tma));
            tma::load_async(As, A_layout, {row, tile}, smem_arrived);
            tma::load_async(Bs, B_layout, {tile, col}, smem_arrived);
        }

        // Wait for TMA to complete
        wait(smem_arrived, phase);
        phase ^= 1;

        // Shared -> registers -> tensor core MMA
        warp::load(A_reg, As);
        warp::load(B_reg, Bs);
        warp::swap_layout(B_reg_col, B_reg);
        warp::mma_AB(C_accum, A_reg, B_reg_col, C_accum);

        __syncthreads();
    }

    // Store result via TMA
    warp::store(Ds, C_accum);
    __syncthreads();

    if (threadIdx.x == 0) {
        tma::store_async(D_layout, Ds, {row, col});
        tma::store_async_read_wait();
    }
}

/*
 * Launch WMMA+TMA GEMM: C = A @ B^T
 * Requires dimensions to be multiples of tile sizes.
 */
inline void launch_gemm_wmma_tma(
    __nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K)
{
    a_gl_tma A_layout{reinterpret_cast<bf16*>(A), nullptr, nullptr,
                       (unsigned long)M, (unsigned long)K};
    b_gl_tma B_layout{reinterpret_cast<bf16*>(B), nullptr, nullptr,
                       (unsigned long)K, (unsigned long)N};
    d_gl_tma D_layout{reinterpret_cast<bf16*>(C), nullptr, nullptr,
                       (unsigned long)M, (unsigned long)N};

    dim3 blocks(N / TILE_N, M / TILE_M);
    int smem_size = MAX_SHARED_MEMORY - 1024;

    cudaFuncSetAttribute(gemm_wmma_tma_kernel,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    gemm_wmma_tma_kernel<<<blocks, WARP_THREADS, smem_size>>>(
        A_layout, B_layout, D_layout, K);
    CHECK_CUDA(cudaGetLastError());
}
