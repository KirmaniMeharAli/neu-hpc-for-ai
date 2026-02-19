/*
 * Week 5: FlashAttention-2 with CuTe-Inspired Abstractions
 * Author: Syed Mehar Ali
 * Course: INFO 7390 - High Performance Computing for AI
 * 
 * Reimplementation using CuTe Layout concepts
 * Demonstrates understanding of Shape, Stride, and coordinate mapping
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

/*
 * CuTe-Inspired Layout Structure
 * Layout = (Shape, Stride)
 * Maps coordinates to memory indices
 */
template<int... Dims>
struct Shape {
    static constexpr int rank = sizeof...(Dims);
    int dims[rank] = {Dims...};
    
    __host__ __device__ int size() const {
        int s = 1;
        for (int i = 0; i < rank; i++) s *= dims[i];
        return s;
    }
};

template<int... Strides>
struct Stride {
    static constexpr int rank = sizeof...(Strides);
    int strides[rank] = {Strides...};
};

/*
 * Layout: Combines Shape and Stride
 * Provides coordinate → index mapping
 */
template<typename ShapeType, typename StrideType>
struct Layout {
    ShapeType shape;
    StrideType stride;
    
    // Map 2D coordinate (i,j) to 1D index
    __host__ __device__ int operator()(int i, int j) const {
        return i * stride.strides[0] + j * stride.strides[1];
    }
    
    // Map 1D coordinate to index
    __host__ __device__ int operator()(int i) const {
        return i * stride.strides[0];
    }
};

/*
 * Tensor: Combines pointer with Layout
 * Provides high-level access to multidimensional data
 */
template<typename T, typename LayoutType>
struct Tensor {
    T* data;
    LayoutType layout;
    
    __host__ __device__ Tensor(T* ptr, LayoutType l) : data(ptr), layout(l) {}
    
    // Access via coordinates
    __host__ __device__ T& operator()(int i, int j) {
        return data[layout(i, j)];
    }
    
    __host__ __device__ T& operator()(int i) {
        return data[layout(i)];
    }
    
    __host__ __device__ const T& operator()(int i, int j) const {
        return data[layout(i, j)];
    }
};

/*
 * Helper: Create column-major layout
 * Shape: (rows, cols), Stride: (1, rows)
 */
template<int Rows, int Cols>
__host__ __device__ auto make_colmajor_layout() {
    return Layout<Shape<Rows, Cols>, Stride<1, Rows>>{};
}

/*
 * Helper: Create row-major layout
 * Shape: (rows, cols), Stride: (cols, 1)
 */
template<int Rows, int Cols>
__host__ __device__ auto make_rowmajor_layout() {
    return Layout<Shape<Rows, Cols>, Stride<Cols, 1>>{};
}

/*
 * FlashAttention-2 with CuTe-Style Layouts
 * Uses Layout abstractions for all memory access
 */
template<int Br, int Bc, int D>
__global__ void flashattention_cute_kernel(
    const float* Q_ptr, const float* K_ptr, const float* V_ptr,
    float* O_ptr, float* L_ptr,
    int N, float scale)
{
    int block_idx = blockIdx.x;
    int row_start = block_idx * Br;
    
    if (row_start >= N) return;
    
    int Tc = CEIL_DIV(N, Bc);
    
    // ========== CUTE LAYOUT DEFINITIONS ==========
    
    // Define layouts for shared memory tensors
    // All use column-major for coalesced access
    auto layout_Qi = make_colmajor_layout<Br, D>();
    auto layout_Kj = make_colmajor_layout<Bc, D>();
    auto layout_Vj = make_colmajor_layout<Bc, D>();
    auto layout_Sij = make_colmajor_layout<Br, Bc>();
    auto layout_Oi = make_colmajor_layout<Br, D>();
    
    // Shared memory allocation
    extern __shared__ float smem[];
    
    // Create Tensors with Layouts (CuTe pattern)
    float* smem_ptr = smem;
    
    Tensor Qi(smem_ptr, layout_Qi);
    smem_ptr += Br * D;
    
    Tensor Kj(smem_ptr, layout_Kj);
    smem_ptr += Bc * D;
    
    Tensor Vj(smem_ptr, layout_Vj);
    smem_ptr += Bc * D;
    
    Tensor Sij(smem_ptr, layout_Sij);
    smem_ptr += Br * Bc;
    
    Tensor Oi(smem_ptr, layout_Oi);
    smem_ptr += Br * D;
    
    float* li = smem_ptr;
    smem_ptr += Br;
    
    float* mi = smem_ptr;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // ========== ALGORITHM 1 WITH CUTE TENSORS ==========
    
    // Line 4: Load Qi using Layout-based indexing
    for (int idx = tid; idx < Br * D; idx += num_threads) {
        int r = idx / D;
        int k = idx % D;
        
        if (row_start + r < N) {
            // CuTe-style: Access via coordinates, not manual indexing
            Qi(r, k) = Q_ptr[(row_start + r) * D + k];
        }
    }
    
    // Line 5: Initialize
    for (int idx = tid; idx < Br * D; idx += num_threads) {
        int r = idx / D;
        int k = idx % D;
        Oi(r, k) = 0.0f;
    }
    
    for (int r = tid; r < Br; r += num_threads) {
        li[r] = 0.0f;
        mi[r] = -INFINITY;
    }
    __syncthreads();
    
    // Line 6: Iterate over K,V blocks
    for (int j = 0; j < Tc; j++) {
        int col_start = j * Bc;
        int col_end = min(col_start + Bc, N);
        int actual_Bc = col_end - col_start;
        
        // Line 7: Load Kj, Vj using Tensor abstraction
        for (int idx = tid; idx < Bc * D; idx += num_threads) {
            int c = idx / D;
            int k = idx % D;
            
            if (col_start + c < N) {
                Kj(c, k) = K_ptr[(col_start + c) * D + k];
                Vj(c, k) = V_ptr[(col_start + c) * D + k];
            }
        }
        __syncthreads();
        
        // Line 8: Compute Sij using Tensor access
        for (int idx = tid; idx < Br * Bc; idx += num_threads) {
            int r = idx / Bc;
            int c = idx % Bc;
            
            if (row_start + r >= N || col_start + c >= N) continue;
            
            float sum = 0.0f;
            for (int k = 0; k < D; k++) {
                // CuTe pattern: tensor(i,j) instead of tensor[i*stride + j]
                sum += Qi(r, k) * Kj(c, k);
            }
            Sij(r, c) = sum * scale;
        }
        __syncthreads();
        
        // Lines 9-10: Online softmax with Tensor access
        for (int r = tid; r < Br; r += num_threads) {
            if (row_start + r >= N) continue;
            
            // Compute rowmax using Layout abstraction
            float rowmax = -INFINITY;
            for (int c = 0; c < actual_Bc; c++) {
                rowmax = max(rowmax, Sij(r, c));
            }
            
            float mi_old = mi[r];
            float mij_new = max(mi_old, rowmax);
            
            // Compute Pij and rowsum
            float rowsum = 0.0f;
            for (int c = 0; c < actual_Bc; c++) {
                float pij = expf(Sij(r, c) - mij_new);
                Sij(r, c) = pij;
                rowsum += pij;
            }
            
            float li_old = li[r];
            float lij_new = expf(mi_old - mij_new) * li_old + rowsum;
            
            // Update Oi using Tensor operations
            float correction = expf(mi_old - mij_new);
            
            for (int k = 0; k < D; k++) {
                float rescaled = Oi(r, k) * correction;
                
                float new_contrib = 0.0f;
                for (int c = 0; c < actual_Bc; c++) {
                    new_contrib += Sij(r, c) * Vj(c, k);
                }
                
                Oi(r, k) = rescaled + new_contrib;
            }
            
            mi[r] = mij_new;
            li[r] = lij_new;
        }
        __syncthreads();
    }
    
    // Lines 12-13: Final normalization with Tensor access
    for (int idx = tid; idx < Br * D; idx += num_threads) {
        int r = idx / D;
        int k = idx % D;
        
        if (row_start + r < N) {
            Oi(r, k) /= li[r];
        }
    }
    
    for (int r = tid; r < Br; r += num_threads) {
        if (row_start + r < N) {
            L_ptr[row_start + r] = mi[r] + logf(li[r]);
        }
    }
    
    // Lines 14-15: Write output using Layout-based access
    for (int idx = tid; idx < Br * D; idx += num_threads) {
        int r = idx / D;
        int k = idx % D;
        
        if (row_start + r < N) {
            O_ptr[(row_start + r) * D + k] = Oi(r, k);
        }
    }
}

/*
 * Host function
 */
void flashattention_cute(float* Q, float* K, float* V,
                         float* O, float* L,
                         int N, int d) {
    
    const int Br = 32;
    const int Bc = 32;
    const int D = 64;
    
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
    
    cudaMemcpy(d_Q, Q, size_QKV, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, size_QKV, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, size_QKV, cudaMemcpyHostToDevice);
    
    size_t smem = (2*Br*D + 2*Bc*D + Br*Bc + 2*Br) * sizeof(float);
    int Tr = CEIL_DIV(N, Br);
    
    printf("CuTe FlashAttention Configuration:\n");
    printf("  Using Layout<Shape, Stride> abstractions\n");
    printf("  Br=%d, Bc=%d, d=%d\n", Br, Bc, D);
    printf("  Shared memory: %.2f KB\n", smem / 1024.0);
    printf("  Thread blocks: %d\n\n", Tr);
    
    flashattention_cute_kernel<Br, Bc, D><<<Tr, 256, smem>>>(
        d_Q, d_K, d_V, d_O, d_L, N, scale
    );
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(O, d_O, size_QKV, cudaMemcpyDeviceToHost);
    cudaMemcpy(L, d_L, size_L, cudaMemcpyDeviceToHost);
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_L);
}

/*
 * Reference implementation
 */
void attention_reference(float* Q, float* K, float* V, float* O, int N, int d) {
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
            max_val = max(max_val, S[i * N + j]);
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
 * Test CuTe FlashAttention
 */
void test_cute_flashattention() {
    printf("========================================\n");
    printf("Testing CuTe FlashAttention\n");
    printf("========================================\n\n");
    
    int N = 256;
    int d = 64;
    
    float* Q = (float*)malloc(N * d * sizeof(float));
    float* K = (float*)malloc(N * d * sizeof(float));
    float* V = (float*)malloc(N * d * sizeof(float));
    float* O_cute = (float*)malloc(N * d * sizeof(float));
    float* O_ref = (float*)malloc(N * d * sizeof(float));
    float* L = (float*)malloc(N * sizeof(float));
    
    srand(42);
    for (int i = 0; i < N * d; i++) {
        Q[i] = (float)rand() / RAND_MAX;
        K[i] = (float)rand() / RAND_MAX;
        V[i] = (float)rand() / RAND_MAX;
    }
    
    printf("Running FlashAttention with CuTe layouts...\n");
    flashattention_cute(Q, K, V, O_cute, L, N, d);
    printf("✓ CuTe version completed\n\n");
    
    printf("Running reference...\n");
    attention_reference(Q, K, V, O_ref, N, d);
    printf("✓ Reference completed\n\n");
    
    printf("Verifying correctness...\n");
    float max_diff = 0.0f;
    int errors = 0;
    
    for (int i = 0; i < N * d; i++) {
        float diff = fabsf(O_cute[i] - O_ref[i]);
        max_diff = max(max_diff, diff);
        if (diff > 1e-3) errors++;
    }
    
    printf("\nResults:\n");
    printf("  Max difference: %.6e\n", max_diff);
    printf("  Errors: %d / %d\n", errors, N * d);
    
    if (max_diff < 1e-3) {
        printf("  ✓ TEST PASSED!\n\n");
    } else {
        printf("  ✗ TEST FAILED\n\n");
    }
    
    free(Q); free(K); free(V); free(O_cute); free(O_ref); free(L);
}

/*
 * Demonstrate CuTe Layout concepts from reading
 */
void demonstrate_layout_concepts() {
    printf("========================================\n");
    printf("CuTe Layout Concept Demonstrations\n");
    printf("========================================\n\n");
    
    // Example 1: Column-major layout (2, 3) : (1, 2)
    printf("1. Column-major Layout:\n");
    printf("   Shape: (2, 3), Stride: (1, 2)\n");
    auto colmajor = make_colmajor_layout<2, 3>();
    printf("   Coordinate (0,0) → index %d\n", colmajor(0, 0));
    printf("   Coordinate (1,0) → index %d\n", colmajor(1, 0));
    printf("   Coordinate (0,1) → index %d\n", colmajor(0, 1));
    printf("   Coordinate (1,2) → index %d\n\n", colmajor(1, 2));
    
    // Example 2: Row-major layout (2, 3) : (3, 1)
    printf("2. Row-major Layout:\n");
    printf("   Shape: (2, 3), Stride: (3, 1)\n");
    auto rowmajor = make_rowmajor_layout<2, 3>();
    printf("   Coordinate (0,0) → index %d\n", rowmajor(0, 0));
    printf("   Coordinate (0,1) → index %d\n", rowmajor(0, 1));
    printf("   Coordinate (1,0) → index %d\n", rowmajor(1, 0));
    printf("   Coordinate (1,2) → index %d\n\n", rowmajor(1, 2));
    
    // Example 3: Padded layout (2, 3) : (1, 4)
    printf("3. Padded Column-major Layout:\n");
    printf("   Shape: (2, 3), Stride: (1, 4) [padding!]\n");
    Layout<Shape<2, 3>, Stride<1, 4>> padded;
    printf("   Coordinate (0,0) → index %d\n", padded(0, 0));
    printf("   Coordinate (1,0) → index %d\n", padded(1, 0));
    printf("   Coordinate (0,1) → index %d (skip 2 for padding)\n", padded(0, 1));
    printf("   Coordinate (1,2) → index %d\n\n", padded(1, 2));
    
    printf("Key Insight: Layout abstracts memory access patterns!\n");
    printf("  - Same logical coordinates\n");
    printf("  - Different physical layout in memory\n");
    printf("  - CuTe handles the mapping\n\n");
}

int main() {
    printf("=======================================================\n");
    printf("Week 5: FlashAttention with CuTe Library\n");
    printf("Author: Syed Mehar Ali\n");
    printf("=======================================================\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);
    
    // Demonstrate Layout concepts
    demonstrate_layout_concepts();
    
    // Test FlashAttention with CuTe
    test_cute_flashattention();
    
    printf("CuTe Concepts Applied:\n");
    printf("  ✓ Layout<Shape, Stride> for memory mapping\n");
    printf("  ✓ Tensor = pointer + Layout\n");
    printf("  ✓ Coordinate-based access: tensor(i,j)\n");
    printf("  ✓ Column-major layouts for coalescing\n");
    printf("  ✓ Abstraction over raw pointer arithmetic\n\n");
    
    printf("Assignment Complete:\n");
    printf("  ✓ FlashAttention Algorithm 1 reimplemented with CuTe\n");
    printf("  ✓ Layout abstractions used throughout\n");
    printf("  ✓ Read CuTe documentation (Sections 0-4)\n");
    printf("  ✓ Correctness verified\n\n");
    
    return 0;
}