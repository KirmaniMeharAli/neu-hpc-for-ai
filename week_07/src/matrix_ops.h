/*
 * matrix_ops.h - Matrix operation helpers for DeepSeekV3 MoE
 *
 * Week 7: Communication - Pure C Implementation
 * All operations are single-threaded, no CUDA, no parallelism.
 */

#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <stdlib.h>

/*
 * Matrix multiply: C = A * B
 *   A: (M x K)
 *   B: (K x N)
 *   C: (M x N)
 * All stored in row-major order.
 */
void matmul(const float *A, const float *B, float *C, int M, int K, int N);

/*
 * Matrix multiply with B transposed: C = A * B^T
 *   A: (M x K)
 *   B: (N x K)  -- stored row-major, but treated as B^T = (K x N)
 *   C: (M x N)
 *
 * This matches PyTorch's F.linear(input, weight) where weight is (out_features x in_features).
 * F.linear computes: output = input @ weight.T
 */
void matmul_bt(const float *A, const float *B, float *C, int M, int K, int N);

/*
 * Element-wise multiply: C[i] = A[i] * B[i]
 */
void elementwise_mul(const float *A, const float *B, float *C, int n);

/*
 * Element-wise add: C[i] = A[i] + B[i]
 */
void elementwise_add(const float *A, const float *B, float *C, int n);

/*
 * Scale: A[i] *= scalar
 */
void scale(float *A, float scalar, int n);

/*
 * Zero out array
 */
void zero_array(float *A, int n);

/*
 * Copy array: dst[i] = src[i]
 */
void copy_array(const float *src, float *dst, int n);

/*
 * Allocate float array, zero-initialized
 */
float *alloc_floats(int n);

#endif /* MATRIX_OPS_H */
