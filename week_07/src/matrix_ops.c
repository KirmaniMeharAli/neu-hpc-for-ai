/*
 * matrix_ops.c - Matrix operation helpers for DeepSeekV3 MoE
 *
 * Week 7: Communication - Pure C Implementation
 * All operations are single-threaded, no CUDA, no parallelism.
 */

#include "matrix_ops.h"
#include <string.h>
#include <stdio.h>

/*
 * Standard matrix multiply: C = A * B
 * A: (M x K), B: (K x N), C: (M x N)
 * Row-major layout.
 */
void matmul(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/*
 * Matrix multiply with B transposed: C = A * B^T
 * A: (M x K), B: (N x K), C: (M x N)
 *
 * This is the operation for PyTorch's F.linear(input, weight):
 *   output = input @ weight.T
 * where weight is stored as (out_features, in_features).
 */
void matmul_bt(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

/*
 * Element-wise multiply: C[i] = A[i] * B[i]
 */
void elementwise_mul(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] * B[i];
    }
}

/*
 * Element-wise add: C[i] = A[i] + B[i]
 */
void elementwise_add(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

/*
 * Scale: A[i] *= scalar
 */
void scale(float *A, float scalar, int n) {
    for (int i = 0; i < n; i++) {
        A[i] *= scalar;
    }
}

/*
 * Zero out array
 */
void zero_array(float *A, int n) {
    memset(A, 0, n * sizeof(float));
}

/*
 * Copy array
 */
void copy_array(const float *src, float *dst, int n) {
    memcpy(dst, src, n * sizeof(float));
}

/*
 * Allocate float array, zero-initialized
 */
float *alloc_floats(int n) {
    float *ptr = (float *)calloc(n, sizeof(float));
    if (!ptr) {
        fprintf(stderr, "ERROR: Failed to allocate %d floats\n", n);
        exit(1);
    }
    return ptr;
}
