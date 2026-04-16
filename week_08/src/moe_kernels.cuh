/*
 * moe_kernels.cuh - CUDA Kernels for DeepSeekV3 MoE Operator
 *
 * Week 8: Data and Expert Parallelism
 *
 * CUDA kernels for:
 *   - Matrix multiply (matmul, matmul_bt)
 *   - Activation functions (SiLU, Sigmoid)
 *   - Element-wise operations
 *   - TopK selection
 *   - Routing logic
 */

#ifndef MOE_KERNELS_CUH
#define MOE_KERNELS_CUH

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

/* ============================================================
 * Matrix Multiply Kernels
 * ============================================================ */

/*
 * C = A * B^T
 * A: (M x K), B: (N x K), C: (M x N)
 * Tiled for shared memory performance.
 */
#define TILE_SIZE 16

__global__ void matmul_bt_kernel(const float *A, const float *B, float *C,
                                  int M, int K, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bCol = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        /* B is (N x K), we want B^T so B[col][bCol] */
        tileB[threadIdx.y][threadIdx.x] =
            (col < N && bCol < K) ? B[col * K + bCol] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/*
 * Standard C = A * B
 * A: (M x K), B: (K x N), C: (M x N)
 */
__global__ void matmul_kernel(const float *A, const float *B, float *C,
                               int M, int K, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


/* ============================================================
 * Activation Kernels
 * ============================================================ */

__global__ void silu_inplace_kernel(float *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = arr[idx];
        arr[idx] = x / (1.0f + expf(-x));
    }
}

__global__ void sigmoid_inplace_kernel(float *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = 1.0f / (1.0f + expf(-arr[idx]));
    }
}


/* ============================================================
 * Element-wise Kernels
 * ============================================================ */

__global__ void elementwise_mul_kernel(const float *A, const float *B,
                                        float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] * B[idx];
    }
}

__global__ void elementwise_add_kernel(const float *A, const float *B,
                                        float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void scale_kernel(float *A, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        A[idx] *= scalar;
    }
}

__global__ void add_bias_kernel(const float *input, const float *bias,
                                 float *output, int num_tokens,
                                 int n_experts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_tokens * n_experts) {
        int e = idx % n_experts;
        output[idx] = input[idx] + bias[e];
    }
}

__global__ void zero_kernel(float *A, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        A[idx] = 0.0f;
    }
}

__global__ void accumulate_kernel(float *dst, const float *src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += src[idx];
    }
}


/* ============================================================
 * Host-side wrappers
 * ============================================================ */

static inline void gpu_matmul_bt(const float *A, const float *B, float *C,
                                   int M, int K, int N, cudaStream_t stream = 0) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_bt_kernel<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
}

static inline void gpu_matmul(const float *A, const float *B, float *C,
                                int M, int K, int N, cudaStream_t stream = 0) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_kernel<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
}

static inline void gpu_silu_inplace(float *arr, int n, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_inplace_kernel<<<blocks, threads, 0, stream>>>(arr, n);
}

static inline void gpu_sigmoid_inplace(float *arr, int n, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sigmoid_inplace_kernel<<<blocks, threads, 0, stream>>>(arr, n);
}

static inline void gpu_elementwise_mul(const float *A, const float *B,
                                         float *C, int n, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    elementwise_mul_kernel<<<blocks, threads, 0, stream>>>(A, B, C, n);
}

static inline void gpu_elementwise_add(const float *A, const float *B,
                                         float *C, int n, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    elementwise_add_kernel<<<blocks, threads, 0, stream>>>(A, B, C, n);
}

static inline void gpu_scale(float *A, float scalar, int n, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scale_kernel<<<blocks, threads, 0, stream>>>(A, scalar, n);
}

static inline void gpu_add_bias(const float *input, const float *bias,
                                  float *output, int num_tokens, int n_experts,
                                  cudaStream_t stream = 0) {
    int n = num_tokens * n_experts;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(input, bias, output,
                                                      num_tokens, n_experts);
}

static inline void gpu_zero(float *A, int n, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    zero_kernel<<<blocks, threads, 0, stream>>>(A, n);
}

static inline void gpu_accumulate(float *dst, const float *src, int n,
                                    cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    accumulate_kernel<<<blocks, threads, 0, stream>>>(dst, src, n);
}

#endif /* MOE_KERNELS_CUH */
