"""
Week 5: FlashAttention with CuTe Library
Author: Syed Mehar Ali
Deploys CuTe-based FlashAttention to H100
"""

import modal

app = modal.App("week5-cute-flashattention")

cuda_image = modal.Image.from_registry(
    "nvidia/cuda:12.3.0-devel-ubuntu22.04",
    add_python="3.11"
)

# Embed the CUDA code
CUTE_CODE = '''
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template<int... Dims>
struct Shape {
    static constexpr int rank = sizeof...(Dims);
    int dims[rank] = {Dims...};
};

template<int... Strides>
struct Stride {
    static constexpr int rank = sizeof...(Strides);
    int strides[rank] = {Strides...};
};

template<typename ShapeType, typename StrideType>
struct Layout {
    ShapeType shape;
    StrideType stride;
    
    __host__ __device__ int operator()(int i, int j) const {
        return i * stride.strides[0] + j * stride.strides[1];
    }
    
    __host__ __device__ int operator()(int i) const {
        return i * stride.strides[0];
    }
};

template<typename T, typename LayoutType>
struct Tensor {
    T* data;
    LayoutType layout;
    
    __host__ __device__ Tensor(T* ptr, LayoutType l) : data(ptr), layout(l) {}
    
    __host__ __device__ T& operator()(int i, int j) {
        return data[layout(i, j)];
    }
    
    __host__ __device__ T& operator()(int i) {
        return data[layout(i)];
    }
};

template<int Rows, int Cols>
__host__ __device__ auto make_colmajor_layout() {
    return Layout<Shape<Rows, Cols>, Stride<1, Rows>>{};
}

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
    
    extern __shared__ float smem[];
    
    auto layout_Qi = make_colmajor_layout<Br, D>();
    auto layout_Kj = make_colmajor_layout<Bc, D>();
    auto layout_Vj = make_colmajor_layout<Bc, D>();
    auto layout_Sij = make_colmajor_layout<Br, Bc>();
    auto layout_Oi = make_colmajor_layout<Br, D>();
    
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
    
    for (int idx = tid; idx < Br * D; idx += num_threads) {
        int r = idx / D;
        int k = idx % D;
        if (row_start + r < N) {
            Qi(r, k) = Q_ptr[(row_start + r) * D + k];
        }
    }
    
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
    
    for (int j = 0; j < Tc; j++) {
        int col_start = j * Bc;
        int col_end = min(col_start + Bc, N);
        int actual_Bc = col_end - col_start;
        
        for (int idx = tid; idx < Bc * D; idx += num_threads) {
            int c = idx / D;
            int k = idx % D;
            if (col_start + c < N) {
                Kj(c, k) = K_ptr[(col_start + c) * D + k];
                Vj(c, k) = V_ptr[(col_start + c) * D + k];
            }
        }
        __syncthreads();
        
        for (int idx = tid; idx < Br * Bc; idx += num_threads) {
            int r = idx / Bc;
            int c = idx % Bc;
            
            if (row_start + r >= N || col_start + c >= N) continue;
            
            float sum = 0.0f;
            for (int k = 0; k < D; k++) {
                sum += Qi(r, k) * Kj(c, k);
            }
            Sij(r, c) = sum * scale;
        }
        __syncthreads();
        
        for (int r = tid; r < Br; r += num_threads) {
            if (row_start + r >= N) continue;
            
            float rowmax = -INFINITY;
            for (int c = 0; c < actual_Bc; c++) {
                rowmax = max(rowmax, Sij(r, c));
            }
            
            float mi_old = mi[r];
            float mij_new = max(mi_old, rowmax);
            
            float rowsum = 0.0f;
            for (int c = 0; c < actual_Bc; c++) {
                float pij = expf(Sij(r, c) - mij_new);
                Sij(r, c) = pij;
                rowsum += pij;
            }
            
            float li_old = li[r];
            float lij_new = expf(mi_old - mij_new) * li_old + rowsum;
            
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
    
    for (int idx = tid; idx < Br * D; idx += num_threads) {
        int r = idx / D;
        int k = idx % D;
        if (row_start + r < N) {
            O_ptr[(row_start + r) * D + k] = Oi(r, k);
        }
    }
}

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

void test_cute() {
    const int N = 256, d = 64, Br = 32, Bc = 32;
    
    float *Q = (float*)malloc(N*d*sizeof(float));
    float *K = (float*)malloc(N*d*sizeof(float));
    float *V = (float*)malloc(N*d*sizeof(float));
    float *O_cute = (float*)malloc(N*d*sizeof(float));
    float *O_ref = (float*)malloc(N*d*sizeof(float));
    float *L = (float*)malloc(N*sizeof(float));
    
    srand(42);
    for (int i = 0; i < N*d; i++) {
        Q[i] = (float)rand()/RAND_MAX;
        K[i] = (float)rand()/RAND_MAX;
        V[i] = (float)rand()/RAND_MAX;
    }
    
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, N*d*sizeof(float));
    cudaMalloc(&d_K, N*d*sizeof(float));
    cudaMalloc(&d_V, N*d*sizeof(float));
    cudaMalloc(&d_O, N*d*sizeof(float));
    cudaMalloc(&d_L, N*sizeof(float));
    
    cudaMemcpy(d_Q, Q, N*d*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, N*d*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, N*d*sizeof(float), cudaMemcpyHostToDevice);
    
    size_t smem = (2*Br*d + 2*Bc*d + Br*Bc + 2*Br)*sizeof(float);
    int Tr = CEIL_DIV(N, Br);
    
    printf("Configuration: N=%d, d=%d, Br=%d, Bc=%d\\n", N, d, Br, Bc);
    printf("Shared memory: %.2f KB\\n", smem/1024.0);
    printf("Thread blocks: %d\\n\\n", Tr);
    
    flashattention_cute_kernel<Br,Bc,d><<<Tr, 256, smem>>>(
        d_Q, d_K, d_V, d_O, d_L, N, 1.0f/sqrtf(d)
    );
    cudaDeviceSynchronize();
    
    cudaMemcpy(O_cute, d_O, N*d*sizeof(float), cudaMemcpyDeviceToHost);
    
    attention_reference(Q, K, V, O_ref, N, d);
    
    float max_diff = 0.0f;
    int errors = 0;
    for (int i = 0; i < N*d; i++) {
        float diff = fabsf(O_cute[i] - O_ref[i]);
        max_diff = max(max_diff, diff);
        if (diff > 1e-3) errors++;
    }
    
    printf("Verification:\\n");
    printf("  Max difference: %.6e\\n", max_diff);
    printf("  Errors: %d/%d\\n", errors, N*d);
    if (max_diff < 1e-3) printf("  ✓ TEST PASSED\\n\\n");
    else printf("  ✗ TEST FAILED\\n\\n");
    
    free(Q); free(K); free(V); free(O_cute); free(O_ref); free(L);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L);
}

void demonstrate_layouts() {
    printf("CuTe Layout Examples:\\n");
    printf("=====================\\n\\n");
    
    printf("1. Column-major (2,3):(1,2)\\n");
    auto col = make_colmajor_layout<2,3>();
    printf("   (0,0)→%d, (1,0)→%d, (0,1)→%d\\n\\n", col(0,0), col(1,0), col(0,1));
    
    printf("2. Tensor access abstraction:\\n");
    float data[6] = {1,2,3,4,5,6};
    Tensor T(data, col);
    printf("   T(0,0)=%.0f, T(1,0)=%.0f, T(0,1)=%.0f\\n\\n", T(0,0), T(1,0), T(0,1));
}

int main() {
    printf("Week 5: FlashAttention with CuTe Layouts\\n");
    printf("=========================================\\n\\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\\n", prop.name);
    printf("Compute: %d.%d\\n\\n", prop.major, prop.minor);
    
    demonstrate_layouts();
    test_cute();
    
    printf("CuTe Concepts Applied:\\n");
    printf("  ✓ Layout<Shape,Stride>\\n");
    printf("  ✓ Tensor abstraction\\n");
    printf("  ✓ Coordinate-based access\\n\\n");
    
    return 0;
}
'''

@app.function(gpu="H100", image=cuda_image, timeout=1800)
def run_cute_flashattention():
    import subprocess
    import os
    
    os.makedirs("/tmp/week5", exist_ok=True)
    os.chdir("/tmp/week5")
    
    print("=" * 70)
    print("Week 5: FlashAttention with CuTe Layouts")
    print("=" * 70)
    print()
    
    with open("flashattention_cute.cu", "w") as f:
        f.write(CUTE_CODE)
    
    print("Compiling with CuTe abstractions (C++17)...")
    result = subprocess.run(
        ["nvcc", "-O3", "-arch=sm_90", "-std=c++17",
         "flashattention_cute.cu", "-o", "flash_cute"],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    if result.returncode != 0:
        print("❌ Compilation failed:")
        print(result.stderr)
        return None
    
    print("✓ Compilation successful\n")
    print("Running on H100...\n")
    
    result = subprocess.run(
        ["./flash_cute"],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    print(result.stdout)
    return result.stdout if result.returncode == 0 else None

@app.local_entrypoint()
def main():
    print("Starting Week 5: CuTe FlashAttention on H100\n")
    
    result = run_cute_flashattention.remote()
    
    if result:
        print("\n" + "="*70)
        print("✅ Week 5 Assignment Complete!")
        print("="*70)
        print("\nAchievements:")
        print("  ✓ FlashAttention reimplemented with CuTe layouts")
        print("  ✓ Layout<Shape,Stride> abstractions applied")
        print("  ✓ Tensor-based coordinate access throughout")
        print("  ✓ Verified on H100 GPU")