/*
 * Week 2 Assignment: Extended GEMM with Transpose Support
 * Author: Syed Mehar Ali
 * Northeastern University - INFO 7390
 * 
 * Part 2: GEMM with optional transpose: C ← α·op(A)·op(B) + β·C
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

__global__ void gemm_extended(
    int m, int n, int k,
    float alpha, float *A, bool transposeA,
    float *B, bool transposeB,
    float beta, float *C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        
        if (!transposeA && !transposeB) {
            for (int i = 0; i < k; i++) {
                sum += A[row * k + i] * B[i * n + col];
            }
        }
        else if (transposeA && !transposeB) {
            for (int i = 0; i < k; i++) {
                sum += A[i * m + row] * B[i * n + col];
            }
        }
        else if (!transposeA && transposeB) {
            for (int i = 0; i < k; i++) {
                sum += A[row * k + i] * B[col * k + i];
            }
        }
        else {
            for (int i = 0; i < k; i++) {
                sum += A[i * m + row] * B[col * k + i];
            }
        }
        
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}

void launch_gemm_extended(
    int m, int n, int k,
    float alpha, float *d_A, bool transposeA,
    float *d_B, bool transposeB,
    float beta, float *d_C)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                 (m + blockDim.y - 1) / blockDim.y);
    
    gemm_extended<<<gridDim, blockDim>>>(
        m, n, k, alpha, d_A, transposeA, d_B, transposeB, beta, d_C);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void gemm_cpu_extended(
    int m, int n, int k,
    float alpha, float *A, bool transposeA,
    float *B, bool transposeB,
    float beta, float *C)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            
            for (int p = 0; p < k; p++) {
                float a_val, b_val;
                
                if (transposeA) {
                    a_val = A[p * m + i];
                } else {
                    a_val = A[i * k + p];
                }
                
                if (transposeB) {
                    b_val = B[j * k + p];
                } else {
                    b_val = B[p * n + j];
                }
                
                sum += a_val * b_val;
            }
            
            C[i * n + j] = alpha * sum + beta * C[i * n + j];
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

void test_all_cases() {
    int m = 3, n = 3, k = 3;
    float alpha = 2.0f, beta = 1.0f;
    
    struct {
        bool transposeA;
        bool transposeB;
        const char* name;
    } tests[] = {
        {false, false, "C ← α·AB + β·C"},
        {true,  false, "C ← α·A^T·B + β·C"},
        {false, true,  "C ← α·A·B^T + β·C"},
        {true,  true,  "C ← α·A^T·B^T + β·C"}
    };
    
    for (int test_idx = 0; test_idx < 4; test_idx++) {
        bool transposeA = tests[test_idx].transposeA;
        bool transposeB = tests[test_idx].transposeB;
        
        printf("\n=== Test Case %d: %s ===\n", test_idx + 1, tests[test_idx].name);
        
        int rows_A = transposeA ? k : m;
        int cols_A = transposeA ? m : k;
        int rows_B = transposeB ? n : k;
        int cols_B = transposeB ? k : n;
        
        int size_A = rows_A * cols_A * sizeof(float);
        int size_B = rows_B * cols_B * sizeof(float);
        int size_C = m * n * sizeof(float);
        
        float *h_A = (float*)malloc(size_A);
        float *h_B = (float*)malloc(size_B);
        float *h_C = (float*)malloc(size_C);
        float *h_C_ref = (float*)malloc(size_C);
        
        for (int i = 0; i < rows_A * cols_A; i++) h_A[i] = (i % 3) + 1.0f;
        for (int i = 0; i < rows_B * cols_B; i++) h_B[i] = (i % 3) + 1.0f;
        for (int i = 0; i < m * n; i++) h_C[i] = 0.5f;
        for (int i = 0; i < m * n; i++) h_C_ref[i] = 0.5f;
        
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, size_A));
        CUDA_CHECK(cudaMalloc(&d_B, size_B));
        CUDA_CHECK(cudaMalloc(&d_C, size_C));
        
        CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));
        
        launch_gemm_extended(m, n, k, alpha, d_A, transposeA, 
                           d_B, transposeB, beta, d_C);
        
        CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
        
        gemm_cpu_extended(m, n, k, alpha, h_A, transposeA, 
                         h_B, transposeB, beta, h_C_ref);
        
        if (matrices_equal(h_C, h_C_ref, m * n)) {
            printf("✓ Test PASSED\n");
        } else {
            printf("✗ Test FAILED\n");
        }
        
        free(h_A); free(h_B); free(h_C); free(h_C_ref);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }
}

int main() {
    printf("Testing Extended GEMM with Transpose Support\n");
    printf("============================================\n");
    test_all_cases();
    return 0;
}
