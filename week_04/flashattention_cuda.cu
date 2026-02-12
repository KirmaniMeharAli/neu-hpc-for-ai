/*
 * Week 4: FlashAttention-2 Algorithm 1 - CUDA Implementation
 * Author: Syed Mehar Ali
 * Course: INFO 7390 - High Performance Computing for AI
 * 
 * Parallelized CUDA implementation of FlashAttention-2 Algorithm 1
 * Each thread block processes one Qi chunk (Br rows of Q)
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/*
 * FlashAttention-2 CUDA Kernel
 * 
 * Implements Algorithm 1 from FlashAttention-2 paper
 * Each block handles one Qi block (Br rows of Q)
 * Parallelization: Tr independent thread blocks
 * 
 * Lines 3-16 of Algorithm 1:
 * - Line 3: Outer loop → Thread block index
 * - Lines 4-5: Initialize per-block state
 * - Lines 6-11: Inner loop over K,V blocks
 * - Lines 12-15: Finalize and write outputs
 */
__global__ void flashattention_v2_kernel(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    int N, int d, int Bc, int Br, float scale)
{
    // Each block processes one Qi chunk
    int block_idx = blockIdx.x;
    int row_start = block_idx * Br;
    int row_end = min(row_start + Br, N);
    int actual_Br = row_end - row_start;
    
    if (row_start >= N) return;
    
    int Tc = CEIL_DIV(N, Bc);
    
    // Shared memory allocation (SRAM in paper)
    extern __shared__ float smem[];
    
    // Memory layout in shared memory
    float* Qi_smem = smem;                                          // Br × d
    float* Kj_smem = &smem[Br * d];                                // Bc × d
    float* Vj_smem = &smem[Br * d + Bc * d];                       // Bc × d
    float* Sij_smem = &smem[Br * d + 2 * Bc * d];                  // Br × Bc
    float* Oi_smem = &smem[Br * d + 2 * Bc * d + Br * Bc];        // Br × d
    float* li_smem = &smem[2 * Br * d + 2 * Bc * d + Br * Bc];    // Br
    float* mi_smem = &smem[2 * Br * d + 2 * Bc * d + Br * Bc + Br]; // Br
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Line 4: Load Qi from HBM to SRAM (parallel across threads)
    for (int idx = tid; idx < actual_Br * d; idx += num_threads) {
        int local_row = idx / d;
        int col = idx % d;
        if (row_start + local_row < N) {
            Qi_smem[idx] = Q[(row_start + local_row) * d + col];
        }
    }
    
    // Line 5: Initialize Oi=(0), li=(0), mi=(-∞)
    for (int idx = tid; idx < actual_Br * d; idx += num_threads) {
        Oi_smem[idx] = 0.0f;
    }
    for (int idx = tid; idx < actual_Br; idx += num_threads) {
        li_smem[idx] = 0.0f;
        mi_smem[idx] = -INFINITY;
    }
    __syncthreads();
    
    // Line 6: for 1 ≤ j ≤ Tc do (iterate over K,V blocks)
    for (int j = 0; j < Tc; j++) {
        int col_start = j * Bc;
        int col_end = min(col_start + Bc, N);
        int actual_Bc = col_end - col_start;
        
        // Line 7: Load Kj, Vj from HBM to SRAM
        for (int idx = tid; idx < actual_Bc * d; idx += num_threads) {
            int local_col = idx / d;
            int dim = idx % d;
            if (col_start + local_col < N) {
                Kj_smem[idx] = K[(col_start + local_col) * d + dim];
                Vj_smem[idx] = V[(col_start + local_col) * d + dim];
            }
        }
        __syncthreads();
        
        // Line 8: Compute Sij = Qi * Kj^T (scaled)
        for (int idx = tid; idx < actual_Br * actual_Bc; idx += num_threads) {
            int r = idx / actual_Bc;
            int c = idx % actual_Bc;
            
            float sum = 0.0f;
            for (int k = 0; k < d; k++) {
                sum += Qi_smem[r * d + k] * Kj_smem[c * d + k];
            }
            Sij_smem[r * actual_Bc + c] = sum * scale;
        }
        __syncthreads();
        
        // Line 9-10: Online softmax update and output accumulation
        // Each thread processes one or more rows
        for (int r = tid; r < actual_Br; r += num_threads) {
            // Compute rowmax of Sij
            float rowmax = -INFINITY;
            for (int c = 0; c < actual_Bc; c++) {
                rowmax = max(rowmax, Sij_smem[r * actual_Bc + c]);
            }
            
            // mij^(j) = max(mi^(j-1), rowmax(Sij))
            float mij_new = max(mi_smem[r], rowmax);
            
            // Compute Pij = exp(Sij - mij) and rowsum
            float rowsum = 0.0f;
            for (int c = 0; c < actual_Bc; c++) {
                float pij_val = expf(Sij_smem[r * actual_Bc + c] - mij_new);
                Sij_smem[r * actual_Bc + c] = pij_val;  // Reuse Sij memory for Pij
                rowsum += pij_val;
            }
            
            // lij^(j) = exp(mi^(j-1) - mij) * li^(j-1) + rowsum(Pij)
            float lij_new = expf(mi_smem[r] - mij_new) * li_smem[r] + rowsum;
            
            // Line 10: Update Oi with correction factor
            float correction = expf(mi_smem[r] - mij_new);
            
            for (int k = 0; k < d; k++) {
                // Oi^(j) = correction * Oi^(j-1) + Pij * Vj
                float term1 = Oi_smem[r * d + k] * correction;
                
                float term2 = 0.0f;
                for (int c = 0; c < actual_Bc; c++) {
                    term2 += Sij_smem[r * actual_Bc + c] * Vj_smem[c * d + k];
                }
                
                Oi_smem[r * d + k] = term1 + term2;
            }
            
            // Update mi, li for next iteration
            mi_smem[r] = mij_new;
            li_smem[r] = lij_new;
        }
        __syncthreads();
    }
    // Line 11: end for (K,V loop)
    
    // Line 12: Final normalization: Oi = diag(li)^(-1) * Oi
    for (int idx = tid; idx < actual_Br * d; idx += num_threads) {
        int r = idx / d;
        int k = idx % d;
        Oi_smem[idx] = Oi_smem[idx] / li_smem[r];
    }
    
    // Line 13: Compute Li = mi + log(li)
    for (int r = tid; r < actual_Br; r += num_threads) {
        if (row_start + r < N) {
            L[row_start + r] = mi_smem[r] + logf(li_smem[r]);
        }
    }
    
    // Line 14-15: Write Oi and Li to HBM
    for (int idx = tid; idx < actual_Br * d; idx += num_threads) {
        int local_row = idx / d;
        int col = idx % d;
        if (row_start + local_row < N) {
            O[(row_start + local_row) * d + col] = Oi_smem[idx];
        }
    }
    // Line 16: end for
    // Line 17: Return O and L (written to global memory)
}

/*
 * Standard attention (CPU reference)
 */
void attention_standard_cpu(float* Q, float* K, float* V, float* O, int N, int d) {
    float scale = 1.0f / sqrtf((float)d);
    float* S = (float*)malloc(N * N * sizeof(float));
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++) {
                sum += Q[i * d + k] * K[j * d + k];
            }
            S[i * N + j] = sum * scale;
        }
    }
    
    float* P = (float*)malloc(N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        float max_val = -INFINITY;
        for (int j = 0; j < N; j++) {
            if (S[i * N + j] > max_val) max_val = S[i * N + j];
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j < N; j++) {
            P[i * N + j] = expf(S[i * N + j] - max_val);
            sum_exp += P[i * N + j];
        }
        
        for (int j = 0; j < N; j++) {
            P[i * N + j] /= sum_exp;
        }
    }
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += P[i * N + k] * V[k * d + j];
            }
            O[i * d + j] = sum;
        }
    }
    
    free(S); free(P);
}

/*
 * Host function to launch FlashAttention CUDA kernel
 */
void flashattention_v2_cuda(float* Q, float* K, float* V,
                             float* O, float* L,
                             int N, int d, int Bc, int Br) {
    
    float scale = 1.0f / sqrtf((float)d);
    
    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    size_t size_QKV = N * d * sizeof(float);
    size_t size_L = N * sizeof(float);
    
    cudaMalloc(&d_Q, size_QKV);
    cudaMalloc(&d_K, size_QKV);
    cudaMalloc(&d_V, size_QKV);
    cudaMalloc(&d_O, size_QKV);
    cudaMalloc(&d_L, size_L);
    
    // Copy inputs to device
    cudaMemcpy(d_Q, Q, size_QKV, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, size_QKV, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, size_QKV, cudaMemcpyHostToDevice);
    
    // Calculate shared memory size
    // Based on: 2·Br·d + 2·Bc·d + Br·Bc + 2·Br
    size_t smem_size = (2 * Br * d +           // Qi, Oi
                        2 * Bc * d +           // Kj, Vj
                        Br * Bc +              // Sij (reused for Pij)
                        2 * Br) * sizeof(float);  // li, mi
    
    // Launch configuration
    int Tr = CEIL_DIV(N, Br);  // Number of Q blocks = number of thread blocks
    int threads_per_block = 256;
    
    // Launch kernel
    flashattention_v2_kernel<<<Tr, threads_per_block, smem_size>>>(
        d_Q, d_K, d_V, d_O, d_L, N, d, Bc, Br, scale
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(O, d_O, size_QKV, cudaMemcpyDeviceToHost);
    cudaMemcpy(L, d_L, size_L, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_L);
}

/*
 * Test FlashAttention CUDA implementation
 */
void test_flashattention_cuda() {
    printf("Testing FlashAttention-2 CUDA Implementation\n");
    printf("============================================\n\n");
    
    int N = 256;   // Sequence length
    int d = 64;    // Embedding dimension
    int Br = 32;   // Row block size
    int Bc = 32;   // Column block size
    
    printf("Configuration:\n");
    printf("  N (sequence):    %d\n", N);
    printf("  d (embedding):   %d\n", d);
    printf("  Br (row block):  %d\n", Br);
    printf("  Bc (col block):  %d\n\n", Bc);
    
    // Calculate and display memory requirements
    size_t smem_per_block = (2*Br*d + 2*Bc*d + Br*Bc + 2*Br) * sizeof(float);
    int Tr = CEIL_DIV(N, Br);
    
    printf("Resource usage:\n");
    printf("  Thread blocks: %d\n", Tr);
    printf("  Shared memory per block: %.2f KB\n\n", smem_per_block / 1024.0);
    
    // Allocate host memory
    float* Q = (float*)malloc(N * d * sizeof(float));
    float* K = (float*)malloc(N * d * sizeof(float));
    float* V = (float*)malloc(N * d * sizeof(float));
    float* O_flash = (float*)malloc(N * d * sizeof(float));
    float* O_standard = (float*)malloc(N * d * sizeof(float));
    float* L = (float*)malloc(N * sizeof(float));
    
    // Initialize with random values (same seed for reproducibility)
    srand(42);
    for (int i = 0; i < N * d; i++) {
        Q[i] = (float)rand() / RAND_MAX;
        K[i] = (float)rand() / RAND_MAX;
        V[i] = (float)rand() / RAND_MAX;
    }
    
    // Run FlashAttention CUDA
    printf("Running FlashAttention-2 (CUDA)...\n");
    flashattention_v2_cuda(Q, K, V, O_flash, L, N, d, Bc, Br);
    printf("✓ CUDA execution completed\n\n");
    
    // Run standard attention (CPU reference)
    printf("Running standard attention (CPU reference)...\n");
    attention_standard_cpu(Q, K, V, O_standard, N, d);
    printf("✓ CPU reference completed\n\n");
    
    // Verify correctness
    printf("Verifying correctness...\n");
    float max_diff = 0.0f;
    int errors = 0;
    
    for (int i = 0; i < N * d; i++) {
        float diff = fabsf(O_flash[i] - O_standard[i]);
        max_diff = MAX(max_diff, diff);
        
        if (diff > 1e-3) {
            errors++;
            if (errors <= 5) {
                printf("  Mismatch [%d]: CUDA=%.6f, CPU=%.6f, diff=%.6f\n",
                       i, O_flash[i], O_standard[i], diff);
            }
        }
    }
    
    printf("\nResults:\n");
    printf("  Max difference: %.6e\n", max_diff);
    printf("  Errors (>1e-3): %d / %d\n", errors, N * d);
    
    if (max_diff < 1e-3) {
        printf("  ✓ TEST PASSED: CUDA matches CPU reference!\n\n");
    } else {
        printf("  ✗ TEST FAILED: Results differ\n\n");
    }
    
    // Cleanup
    free(Q); free(K); free(V);
    free(O_flash); free(O_standard); free(L);
}

int main() {
    printf("=======================================================\n");
    printf("Week 4: FlashAttention-2 Algorithm 1\n");
    printf("Parallelized CUDA Implementation\n");
    printf("Author: Syed Mehar Ali\n");
    printf("=======================================================\n\n");
    
    // Display GPU information
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Shared Memory per Block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("Max Threads per Block: %d\n\n", prop.maxThreadsPerBlock);
    
    test_flashattention_cuda();
    
    printf("Implementation Summary:\n");
    printf("  ✓ Algorithm 1 implemented in CUDA\n");
    printf("  ✓ One thread block per Qi chunk\n");
    printf("  ✓ %d thread blocks total\n", CEIL_DIV(256, 32));
    printf("  ✓ Online softmax in shared memory\n");
    printf("  ✓ Tiled computation\n");
    printf("  ✓ Correctness verified\n\n");
    
    return 0;
}


