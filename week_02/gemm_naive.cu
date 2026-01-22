/*
 * Week 2 Assignment: Naive GEMM Implementation
 * Author: Syed Mehar Ali
 * Northeastern University - INFO 7390
 * 
 * Part 1: Implement naive GEMM: D = alpha * A * B + beta * C
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void gemm_naive(
    int m, int n, int k,
    float alpha, float *A, float *B,
    float beta, float *C, float *D)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        D[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}

void launch_gemm_naive(
    int m, int n, int k,
    float alpha, float *d_A, float *d_B,
    float beta, float *d_C, float *d_D)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                 (m + blockDim.y - 1) / blockDim.y);
    
    gemm_naive<<<gridDim, blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C, d_D);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void gemm_cpu(
    int m, int n, int k,
    float alpha, float *A, float *B,
    float beta, float *C, float *D)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            D[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
}

bool matrices_equal(float *A, float *B, int size, float epsilon = 1e-3) {
    for (int i = 0; i < size; i++) {
        if (fabs(A[i] - B[i]) > epsilon) {
            printf("Mismatch at index %d: GPU=%f, CPU=%f, diff=%f\n",
                   i, A[i], B[i], fabs(A[i] - B[i]));
            return false;
        }
    }
    return true;
}

void test_gemm() {
    int m = 4, n = 4, k = 4;
    float alpha = 2.0f, beta = 1.0f;
    
    int size_A = m * k * sizeof(float);
    int size_B = k * n * sizeof(float);
    int size_C = m * n * sizeof(float);
    int size_D = m * n * sizeof(float);
    
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    float *h_D = (float*)malloc(size_D);
    float *h_D_ref = (float*)malloc(size_D);
    
    for (int i = 0; i < m * k; i++) h_A[i] = 1.0f;
    for (int i = 0; i < k * n; i++) h_B[i] = 2.0f;
    for (int i = 0; i < m * n; i++) h_C[i] = 0.5f;
    
    float *d_A, *d_B, *d_C, *d_D;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMalloc(&d_D, size_D));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));
    
    launch_gemm_naive(m, n, k, alpha, d_A, d_B, beta, d_C, d_D);
    CUDA_CHECK(cudaMemcpy(h_D, d_D, size_D, cudaMemcpyDeviceToHost));
    
    gemm_cpu(m, n, k, alpha, h_A, h_B, beta, h_C, h_D_ref);
    
    if (matrices_equal(h_D, h_D_ref, m * n)) {
        printf("✓ Test PASSED: GPU and CPU results match!\n");
    } else {
        printf("✗ Test FAILED: GPU and CPU results differ!\n");
    }
    
    free(h_A); free(h_B); free(h_C); free(h_D); free(h_D_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
}

int main() {
    printf("Testing Naive GEMM Implementation\n");
    printf("==================================\n");
    test_gemm();
    return 0;
}
