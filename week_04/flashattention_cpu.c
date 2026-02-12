/*
 * Week 4: FlashAttention-2 Algorithm 1 - CPU Implementation
 * Author: Syed Mehar Ali
 * Course: INFO 7390 - High Performance Computing for AI
 * 
 * Unparallelized C implementation of FlashAttention-2 Algorithm 1
 * From: "FlashAttention-2: Faster Attention with Better Parallelism" (Section 3.1)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/*
 * Standard Attention Implementation (Reference)
 * O = softmax(QK^T / sqrt(d)) * V
 */
void attention_standard(float* Q, float* K, float* V, float* O, int N, int d) {
    float scale = 1.0f / sqrtf((float)d);
    
    // Allocate S = QK^T (N x N)
    float* S = (float*)malloc(N * N * sizeof(float));
    
    // Compute S = QK^T with scaling
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++) {
                sum += Q[i * d + k] * K[j * d + k];
            }
            S[i * N + j] = sum * scale;
        }
    }
    
    // Apply softmax row-wise
    float* P = (float*)malloc(N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        // Find max for numerical stability
        float max_val = -INFINITY;
        for (int j = 0; j < N; j++) {
            max_val = MAX(max_val, S[i * N + j]);
        }
        
        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int j = 0; j < N; j++) {
            P[i * N + j] = expf(S[i * N + j] - max_val);
            sum_exp += P[i * N + j];
        }
        
        // Normalize
        for (int j = 0; j < N; j++) {
            P[i * N + j] /= sum_exp;
        }
    }
    
    // Compute O = P * V
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += P[i * N + k] * V[k * d + j];
            }
            O[i * d + j] = sum;
        }
    }
    
    free(S);
    free(P);
}

/*
 * FlashAttention-2 Algorithm 1 (Unparallelized C)
 * 
 * Implements tiled attention with online softmax
 * Lines 3-17 from Algorithm 1 in Section 3.1
 * 
 * Q, K, V: (N, d) input matrices
 * O: (N, d) output matrix
 * L: (N,) logsumexp output
 * Bc, Br: block sizes for tiling
 */
void flashattention_v2_cpu(float* Q, float* K, float* V, 
                            float* O, float* L,
                            int N, int d, int Bc, int Br) {
    
    float scale = 1.0f / sqrtf((float)d);
    
    // Calculate number of blocks
    int Tr = (N + Br - 1) / Br;  // Ceiling division for Q blocks
    int Tc = (N + Bc - 1) / Bc;  // Ceiling division for K,V blocks
    
    // Line 3: for 1 ≤ i ≤ Tr do (iterate over Q blocks)
    for (int i = 0; i < Tr; i++) {
        int row_start = i * Br;
        int row_end = MIN(row_start + Br, N);
        int actual_Br = row_end - row_start;
        
        // Line 4: Load Qi from HBM to SRAM
        float* Qi = (float*)malloc(actual_Br * d * sizeof(float));
        for (int r = 0; r < actual_Br; r++) {
            memcpy(&Qi[r * d], &Q[(row_start + r) * d], d * sizeof(float));
        }
        
        // Line 5: Initialize Oi=(0)_Br×d, li=(0)_Br, mi=(-∞)_Br on chip
        float* Oi = (float*)calloc(actual_Br * d, sizeof(float));  // Zero initialized
        float* li = (float*)calloc(actual_Br, sizeof(float));      // Zero initialized
        float* mi = (float*)malloc(actual_Br * sizeof(float));
        for (int r = 0; r < actual_Br; r++) {
            mi[r] = -INFINITY;
        }
        
        // Line 6: for 1 ≤ j ≤ Tc do (iterate over K,V blocks)
        for (int j = 0; j < Tc; j++) {
            int col_start = j * Bc;
            int col_end = MIN(col_start + Bc, N);
            int actual_Bc = col_end - col_start;
            
            // Line 7: Load Kj, Vj from HBM to on-chip SRAM
            float* Kj = (float*)malloc(actual_Bc * d * sizeof(float));
            float* Vj = (float*)malloc(actual_Bc * d * sizeof(float));
            for (int c = 0; c < actual_Bc; c++) {
                memcpy(&Kj[c * d], &K[(col_start + c) * d], d * sizeof(float));
                memcpy(&Vj[c * d], &V[(col_start + c) * d], d * sizeof(float));
            }
            
            // Line 8: On chip, compute Sij = Qi * Kj^T ∈ R^(Br × Bc)
            float* Sij = (float*)malloc(actual_Br * actual_Bc * sizeof(float));
            for (int r = 0; r < actual_Br; r++) {
                for (int c = 0; c < actual_Bc; c++) {
                    float sum = 0.0f;
                    for (int k = 0; k < d; k++) {
                        sum += Qi[r * d + k] * Kj[c * d + k];
                    }
                    Sij[r * actual_Bc + c] = sum * scale;
                }
            }
            
            // Line 9: Online softmax update
            // mij = max(mi^(j-1), rowmax(Sij))
            // Pij = exp(Sij - mij) (pointwise)
            // lij = exp(mi^(j-1) - mij) * li^(j-1) + rowsum(Pij)
            
            float* mij = (float*)malloc(actual_Br * sizeof(float));
            float* Pij = (float*)malloc(actual_Br * actual_Bc * sizeof(float));
            float* lij = (float*)malloc(actual_Br * sizeof(float));
            
            for (int r = 0; r < actual_Br; r++) {
                // Compute rowmax of Sij
                float rowmax = -INFINITY;
                for (int c = 0; c < actual_Bc; c++) {
                    rowmax = MAX(rowmax, Sij[r * actual_Bc + c]);
                }
                
                // Update mij
                mij[r] = MAX(mi[r], rowmax);
                
                // Compute Pij = exp(Sij - mij) pointwise
                for (int c = 0; c < actual_Bc; c++) {
                    Pij[r * actual_Bc + c] = expf(Sij[r * actual_Bc + c] - mij[r]);
                }
                
                // Compute rowsum of Pij
                float rowsum = 0.0f;
                for (int c = 0; c < actual_Bc; c++) {
                    rowsum += Pij[r * actual_Bc + c];
                }
                
                // Update lij
                lij[r] = expf(mi[r] - mij[r]) * li[r] + rowsum;
            }
            
            // Line 10: Update Oi
            // Oi^(j) = diag(exp(mi^(j-1) - mij))^(-1) * Oi^(j-1) + Pij * Vj
            float* Oi_new = (float*)malloc(actual_Br * d * sizeof(float));
            
            for (int r = 0; r < actual_Br; r++) {
                float correction = expf(mi[r] - mij[r]);
                
                for (int k = 0; k < d; k++) {
                    // First term: correction factor * old Oi
                    float term1 = Oi[r * d + k] * correction;
                    
                    // Second term: Pij * Vj
                    float term2 = 0.0f;
                    for (int c = 0; c < actual_Bc; c++) {
                        term2 += Pij[r * actual_Bc + c] * Vj[c * d + k];
                    }
                    
                    Oi_new[r * d + k] = term1 + term2;
                }
            }
            
            // Update Oi, li, mi for next iteration
            memcpy(Oi, Oi_new, actual_Br * d * sizeof(float));
            memcpy(li, lij, actual_Br * sizeof(float));
            memcpy(mi, mij, actual_Br * sizeof(float));
            
            // Cleanup iteration
            free(Kj); free(Vj); free(Sij);
            free(mij); free(Pij); free(lij); free(Oi_new);
        }
        // Line 11: end for (K,V loop)
        
        // Line 12: Final normalization: Oi = diag(li^(Tc))^(-1) * Oi^(Tc)
        for (int r = 0; r < actual_Br; r++) {
            for (int k = 0; k < d; k++) {
                Oi[r * d + k] = Oi[r * d + k] / li[r];
            }
        }
        
        // Line 13: Compute Li = mi^(Tc) + log(li^(Tc))
        for (int r = 0; r < actual_Br; r++) {
            L[row_start + r] = mi[r] + logf(li[r]);
        }
        
        // Line 14: Write Oi to HBM as i-th block of O
        for (int r = 0; r < actual_Br; r++) {
            memcpy(&O[(row_start + r) * d], &Oi[r * d], d * sizeof(float));
        }
        
        // Line 15: Write Li to HBM (already done in line 13)
        
        // Cleanup block
        free(Qi); free(Oi); free(li); free(mi);
    }
    // Line 16: end for (Q loop)
    
    // Line 17: Return O and L (already in output parameters)
}

/*
 * Test FlashAttention CPU implementation
 */
void test_flashattention_cpu() {
    printf("Testing FlashAttention-2 CPU Implementation\n");
    printf("===========================================\n\n");
    
    int N = 128;   // Sequence length
    int d = 64;    // Embedding dimension
    int Br = 32;   // Row block size
    int Bc = 32;   // Column block size
    
    printf("Configuration:\n");
    printf("  N (sequence length): %d\n", N);
    printf("  d (embedding dim):   %d\n", d);
    printf("  Br (row block):      %d\n", Br);
    printf("  Bc (col block):      %d\n\n", Bc);
    
    // Allocate matrices
    float* Q = (float*)malloc(N * d * sizeof(float));
    float* K = (float*)malloc(N * d * sizeof(float));
    float* V = (float*)malloc(N * d * sizeof(float));
    float* O_flash = (float*)malloc(N * d * sizeof(float));
    float* O_standard = (float*)malloc(N * d * sizeof(float));
    float* L = (float*)malloc(N * sizeof(float));
    
    // Initialize with random values
    srand(42);
    for (int i = 0; i < N * d; i++) {
        Q[i] = (float)rand() / RAND_MAX;
        K[i] = (float)rand() / RAND_MAX;
        V[i] = (float)rand() / RAND_MAX;
    }
    
    // Compute FlashAttention
    printf("Running FlashAttention-2 (C implementation)...\n");
    flashattention_v2_cpu(Q, K, V, O_flash, L, N, d, Bc, Br);
    printf("✓ FlashAttention completed\n\n");
    
    // Compute standard attention for verification
    printf("Running standard attention (reference)...\n");
    attention_standard(Q, K, V, O_standard, N, d);
    printf("✓ Standard attention completed\n\n");
    
    // Compare results
    printf("Verifying correctness...\n");
    float max_diff = 0.0f;
    int errors = 0;
    
    for (int i = 0; i < N * d; i++) {
        float diff = fabsf(O_flash[i] - O_standard[i]);
        max_diff = MAX(max_diff, diff);
        
        if (diff > 1e-4) {
            errors++;
            if (errors <= 5) {
                printf("  Mismatch at [%d]: Flash=%.6f, Std=%.6f, diff=%.6f\n",
                       i, O_flash[i], O_standard[i], diff);
            }
        }
    }
    
    printf("\nResults:\n");
    printf("  Max difference: %.6e\n", max_diff);
    printf("  Errors (>1e-4): %d / %d elements\n", errors, N * d);
    
    if (max_diff < 1e-4) {
        printf("  ✓ TEST PASSED: FlashAttention matches standard attention!\n\n");
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
    printf("Unparallelized C Implementation\n");
    printf("Author: Syed Mehar Ali\n");
    printf("=======================================================\n\n");
    
    test_flashattention_cpu();
    
    printf("Implementation Details:\n");
    printf("  ✓ Algorithm 1 from FlashAttention-2 (Section 3.1)\n");
    printf("  ✓ Online softmax (lines 9-10)\n");
    printf("  ✓ Tiled computation with Br×d and Bc×d blocks\n");
    printf("  ✓ Output correction on each iteration (line 10)\n");
    printf("  ✓ Final normalization (lines 12-13)\n");
    printf("  ✓ Verified against standard attention\n\n");
    
    return 0;
}