/*
 * Week 1 Assignment: Single-Threaded Matrix Multiplication
 * Author: Syed Mehar Ali
 * Northeastern University - INFO 7390
 * 
 * Implements: C = A * B
 * Where A is m x k, B is k x n, C is m x n
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Matrix multiplication: C = A * B
void matmul_single(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// Test case structure
typedef struct {
    int m, n, k;
    const char* description;
} TestCase;

// Run a single test
int run_test(TestCase test) {
    printf("Testing: %s (A: %dx%d, B: %dx%d)\n", 
           test.description, test.m, test.k, test.k, test.n);
    
    // Allocate matrices
    float *A = (float*)malloc(test.m * test.k * sizeof(float));
    float *B = (float*)malloc(test.k * test.n * sizeof(float));
    float *C = (float*)malloc(test.m * test.n * sizeof(float));
    
    // Initialize with simple values
    for (int i = 0; i < test.m * test.k; i++) A[i] = 1.0f;
    for (int i = 0; i < test.k * test.n; i++) B[i] = 2.0f;
    
    // Compute
    matmul_single(A, B, C, test.m, test.n, test.k);
    
    // Verify: each element should be k * 1.0 * 2.0 = 2k
    float expected = 2.0f * test.k;
    int passed = 1;
    for (int i = 0; i < test.m * test.n; i++) {
        if (C[i] != expected) {
            printf("  ✗ FAILED: C[%d] = %f, expected %f\n", i, C[i], expected);
            passed = 0;
            break;
        }
    }
    
    if (passed) {
        printf("  ✓ PASSED\n");
    }
    
    free(A); free(B); free(C);
    return passed;
}

int main() {
    printf("Week 1: Single-Threaded Matrix Multiplication\n");
    printf("=============================================\n\n");
    
    // Test cases covering different dimensions
    TestCase tests[] = {
        {1, 1, 1, "1x1 * 1x1"},
        {1, 5, 1, "1x1 * 1x5"},
        {2, 3, 1, "2x1 * 1x3"},
        {2, 2, 2, "2x2 * 2x2"},
        {3, 3, 3, "3x3 * 3x3"},
        {4, 4, 4, "4x4 * 4x4"},
        {5, 3, 4, "5x4 * 4x3"},
        {10, 10, 10, "10x10 * 10x10"}
    };
    
    int num_tests = sizeof(tests) / sizeof(TestCase);
    int passed = 0;
    
    for (int i = 0; i < num_tests; i++) {
        passed += run_test(tests[i]);
        printf("\n");
    }
    
    printf("Results: %d/%d tests passed\n", passed, num_tests);
    
    return 0;
}
