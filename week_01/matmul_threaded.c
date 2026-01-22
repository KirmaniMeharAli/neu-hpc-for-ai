/*
 * Week 1 Assignment: Multi-Threaded Matrix Multiplication
 * Author: Syed Mehar Ali
 * Northeastern University - INFO 7390
 * 
 * Implements: C = A * B using pthreads
 * Each thread computes a subset of rows in the output matrix
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

// Thread arguments structure
typedef struct {
    float *A;
    float *B;
    float *C;
    int m, n, k;
    int start_row;
    int end_row;
} ThreadArgs;

// Thread function: compute assigned rows
void* matmul_thread(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    
    for (int i = args->start_row; i < args->end_row; i++) {
        for (int j = 0; j < args->n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < args->k; p++) {
                sum += args->A[i * args->k + p] * args->B[p * args->n + j];
            }
            args->C[i * args->n + j] = sum;
        }
    }
    
    return NULL;
}

// Multi-threaded matrix multiplication
void matmul_parallel(float *A, float *B, float *C, int m, int n, int k, int num_threads) {
    pthread_t *threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    ThreadArgs *args = (ThreadArgs*)malloc(num_threads * sizeof(ThreadArgs));
    
    int rows_per_thread = m / num_threads;
    int remaining_rows = m % num_threads;
    
    int current_row = 0;
    for (int i = 0; i < num_threads; i++) {
        args[i].A = A;
        args[i].B = B;
        args[i].C = C;
        args[i].m = m;
        args[i].n = n;
        args[i].k = k;
        args[i].start_row = current_row;
        
        // Distribute remaining rows to first threads
        int rows_for_this_thread = rows_per_thread + (i < remaining_rows ? 1 : 0);
        args[i].end_row = current_row + rows_for_this_thread;
        
        pthread_create(&threads[i], NULL, matmul_thread, &args[i]);
        current_row = args[i].end_row;
    }
    
    // Wait for all threads
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    free(threads);
    free(args);
}

// Single-threaded version for comparison
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

// Get current time in milliseconds
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// Benchmark function
void benchmark(int m, int n, int k, int num_threads) {
    // Allocate matrices
    float *A = (float*)malloc(m * k * sizeof(float));
    float *B = (float*)malloc(k * n * sizeof(float));
    float *C = (float*)malloc(m * n * sizeof(float));
    
    // Initialize
    for (int i = 0; i < m * k; i++) A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < k * n; i++) B[i] = (float)(rand() % 100) / 10.0f;
    
    // Benchmark
    double start = get_time_ms();
    
    if (num_threads == 1) {
        matmul_single(A, B, C, m, n, k);
    } else {
        matmul_parallel(A, B, C, m, n, k, num_threads);
    }
    
    double end = get_time_ms();
    double time_taken = end - start;
    
    printf("Threads: %3d | Time: %8.2f ms\n", num_threads, time_taken);
    
    free(A); free(B); free(C);
}

// Test correctness
int test_correctness() {
    printf("Testing Correctness\n");
    printf("===================\n");
    
    int m = 100, n = 100, k = 100;
    
    float *A = (float*)malloc(m * k * sizeof(float));
    float *B = (float*)malloc(k * n * sizeof(float));
    float *C_single = (float*)malloc(m * n * sizeof(float));
    float *C_parallel = (float*)malloc(m * n * sizeof(float));
    
    // Initialize with known values
    for (int i = 0; i < m * k; i++) A[i] = 1.0f;
    for (int i = 0; i < k * n; i++) B[i] = 2.0f;
    
    // Compute with both methods
    matmul_single(A, B, C_single, m, n, k);
    matmul_parallel(A, B, C_parallel, m, n, k, 4);
    
    // Compare results
    int passed = 1;
    for (int i = 0; i < m * n; i++) {
        if (C_single[i] != C_parallel[i]) {
            printf("✗ Mismatch at index %d: single=%f, parallel=%f\n", 
                   i, C_single[i], C_parallel[i]);
            passed = 0;
            break;
        }
    }
    
    if (passed) {
        printf("✓ Correctness test PASSED: Single-threaded and multi-threaded results match\n");
    }
    
    free(A); free(B); free(C_single); free(C_parallel);
    return passed;
}

int main() {
    printf("Week 1: Multi-Threaded Matrix Multiplication with Pthreads\n");
    printf("=========================================================\n\n");
    
    // Test correctness first
    if (!test_correctness()) {
        printf("Correctness test failed! Fix bugs before benchmarking.\n");
        return 1;
    }
    
    printf("\nPerformance Benchmark\n");
    printf("=====================\n");
    printf("Matrix size: 1000 x 1000\n\n");
    
    int m = 1000, n = 1000, k = 1000;
    int thread_counts[] = {1, 4, 16, 32, 64, 128};
    int num_configs = sizeof(thread_counts) / sizeof(int);
    
    double baseline_time = 0;
    
    for (int i = 0; i < num_configs; i++) {
        int num_threads = thread_counts[i];
        
        // Get baseline for speedup calculation
        if (num_threads == 1) {
            float *A = (float*)malloc(m * k * sizeof(float));
            float *B = (float*)malloc(k * n * sizeof(float));
            float *C = (float*)malloc(m * n * sizeof(float));
            
            for (int j = 0; j < m * k; j++) A[j] = 1.0f;
            for (int j = 0; j < k * n; j++) B[j] = 2.0f;
            
            double start = get_time_ms();
            matmul_single(A, B, C, m, n, k);
            double end = get_time_ms();
            baseline_time = end - start;
            
            printf("Threads: %3d | Time: %8.2f ms | Speedup: 1.00x\n", 
                   num_threads, baseline_time);
            
            free(A); free(B); free(C);
        } else {
            float *A = (float*)malloc(m * k * sizeof(float));
            float *B = (float*)malloc(k * n * sizeof(float));
            float *C = (float*)malloc(m * n * sizeof(float));
            
            for (int j = 0; j < m * k; j++) A[j] = 1.0f;
            for (int j = 0; j < k * n; j++) B[j] = 2.0f;
            
            double start = get_time_ms();
            matmul_parallel(A, B, C, m, n, k, num_threads);
            double end = get_time_ms();
            double time_taken = end - start;
            double speedup = baseline_time / time_taken;
            
            printf("Threads: %3d | Time: %8.2f ms | Speedup: %.2fx\n", 
                   num_threads, time_taken, speedup);
            
            free(A); free(B); free(C);
        }
    }
    
    return 0;
}
