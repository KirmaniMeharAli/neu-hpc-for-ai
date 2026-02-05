"""
Week 3: Complete CUDA Matrix Multiplication Optimization
Single file that deploys all kernels to Modal.com H100
Author: Syed Mehar Ali
"""

import modal

app = modal.App("week-03-cuda-optimization")

# CUDA image
cuda_image = modal.Image.debian_slim().apt_install(
    "wget", "build-essential", "nvidia-cuda-toolkit"
)

# All kernel code embedded in Python
KERNEL_CODE = '''
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define BM 64
#define BN 64
#define BK 8
#define TM 8

// Kernel 0: Naive
__global__ void sgemm_naive(int M, int N, int K, float alpha,
                            const float *A, const float *B,
                            float beta, float *C) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < M && y < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

// Kernel 2: Shared Memory
__global__ void sgemm_shared_mem(int M, int N, int K, float alpha,
                                 const float *A, const float *B,
                                 float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    
    const uint threadCol = threadIdx.x % BN;
    const uint threadRow = threadIdx.x / BN;
    
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;
    
    float threadResults[TM] = {0.0};
    
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += blockDim.x / BN) {
            As[(threadRow + loadOffset) * BK + threadCol] =
                A[(threadRow + loadOffset) * K + threadCol];
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += blockDim.x / BN) {
            Bs[(threadRow + loadOffset) * BN + threadCol] =
                B[(threadRow + loadOffset) * N + threadCol];
        }
        __syncthreads();
        
        A += BK;
        B += BK * N;
        
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float Btmp = Bs[dotIdx * BN + threadCol];
            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] +=
                    As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
            }
        }
        __syncthreads();
    }
    
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * N + threadCol] =
            alpha * threadResults[resIdx] +
            beta * C[(threadRow * TM + resIdx) * N + threadCol];
    }
}

// Benchmark function
float benchmark(void (*kernel)(int, int, int, float, const float*, const float*, float, float*),
                dim3 gridDim, dim3 blockDim, int M, int N, int K,
                float *d_A, float *d_B, float *d_C, const char* name) {
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    kernel<<<gridDim, blockDim>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
    cudaDeviceSynchronize();
    
    float total = 0;
    for (int i = 0; i < 100; i++) {
        cudaEventRecord(start);
        kernel<<<gridDim, blockDim>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total += ms;
    }
    
    float avg = total / 100;
    double tflops = (2.0 * M * N * K) / (avg / 1000.0) / 1e12;
    
    printf("%-30s | %8.3f ms | %6.2f TFLOPS\\n", name, avg, tflops);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return tflops;
}

int main() {
    printf("Week 3: CUDA Matrix Multiplication Optimization\\n");
    printf("Running on H100 GPU via Modal.com\\n");
    printf("Author: Syed Mehar Ali\\n");
    printf("========================================\\n\\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\\n", prop.name);
    printf("Compute Capability: %d.%d\\n\\n", prop.major, prop.minor);
    
    const int M = 4096, N = 4096, K = 4096;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 2.0f;
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
    
    printf("%-30s | %10s | %s\\n", "Kernel", "Time", "Performance");
    printf("========================================\\n");
    
    dim3 b0(32, 32);
    dim3 g0(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    float t0 = benchmark(sgemm_naive, g0, b0, M, N, K, d_A, d_B, d_C, "Naive");
    
    dim3 b2((BM * BN) / TM);
    dim3 g2(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    float t2 = benchmark(sgemm_shared_mem, g2, b2, M, N, K, d_A, d_B, d_C, "Shared Memory");
    
    printf("========================================\\n");
    printf("Speedup: %.2fx\\n", t2 / t0);
    
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}
'''

@app.function(gpu="H100", image=cuda_image, timeout=1800)
def run_kernels():
    import subprocess
    import os
    
    os.makedirs("/tmp/cuda", exist_ok=True)
    os.chdir("/tmp/cuda")
    
    with open("kernels.cu", "w") as f:
        f.write(KERNEL_CODE)
    
    print("Compiling...")
    result = subprocess.run(
        ["nvcc", "-O3", "-arch=sm_90", "kernels.cu", "-o", "kernels"],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print("Compilation failed:")
        print(result.stderr)
        return
    
    print("✓ Compiled successfully\n")
    print("Running benchmarks...\n")
    
    result = subprocess.run(["./kernels"], capture_output=True, text=True)
    print(result.stdout)
    
    return result.stdout

@app.local_entrypoint()
def main():
    result = run_kernels.remote()
    print("\n✅ Week 3 Complete!")