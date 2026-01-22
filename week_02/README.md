# Week 2: CUDA GEMM Implementation

**Author:** Syed Mehar Ali  
**Course:** INFO 7390 - High Performance Computing for AI  
**University:** Northeastern University Vancouver

## Assignment Overview

This assignment implements General Matrix Multiplication (GEMM) kernels in CUDA:
- **Part 1**: Naive GEMM implementation (`D = α·AB + β·C`)
- **Part 2**: Extended GEMM with transpose support (`C ← α·op(A)·op(B) + β·C`)

## Files

- `gemm_naive.cu` - Part 1: Basic GEMM kernel
- `gemm_extended.cu` - Part 2: GEMM with transpose options
- `README.md` - This documentation

## Implementation Details

### Part 1: Naive GEMM

**Formula**: `D = α·AB + β·C`

**Matrix Dimensions**:
- A: m × k
- B: k × n  
- C: m × n
- D: m × n (output)

**Kernel Design**:
- Each thread computes one element of output matrix D
- Uses 2D thread blocks (16×16 = 256 threads per block)
- Direct global memory access (no shared memory optimization)
- Row-major matrix layout

**Key Implementation**:
```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

if (row < m && col < n) {
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        sum += A[row * k + i] * B[i * n + col];
    }
    D[row * n + col] = alpha * sum + beta * C[row * n + col];
}
```

### Part 2: Extended GEMM with Transpose

**Formula**: `C ← α·op(A)·op(B) + β·C` where op(X) is X or X^T

**Supported Operations**:
1. `C ← α·AB + β·C` (no transpose)
2. `C ← α·A^T·B + β·C` (transpose A)
3. `C ← α·A·B^T + β·C` (transpose B)
4. `C ← α·A^T·B^T + β·C` (transpose both)

**Key Differences**:
- In-place update of C (no separate D matrix)
- Boolean flags `transposeA` and `transposeB`
- Conditional indexing based on transpose flags

**Memory Layout**:
- **No transpose**: A[row, i] = A[row * k + i]
- **Transpose**: A^T[row, i] = A[i * m + row]

## Testing Strategy

Both implementations include:
- CPU reference implementation for verification
- Automated testing comparing GPU vs CPU results
- Small test matrices (3×3, 4×4) for easy verification
- Error tolerance of 1e-3 for floating-point comparison

## Results

**Part 1 (Naive GEMM)**:
- ✓ Test PASSED: GPU and CPU results match

**Part 2 (Extended GEMM)**:
- ✓ Test Case 1 PASSED: C ← α·AB + β·C
- ✓ Test Case 2 PASSED: C ← α·A^T·B + β·C
- ✓ Test Case 3 PASSED: C ← α·A·B^T + β·C
- ✓ Test Case 4 PASSED: C ← α·A^T·B^T + β·C

## Compilation and Execution
```bash
# Compile naive GEMM
nvcc -o gemm_naive gemm_naive.cu

# Run naive GEMM test
./gemm_naive

# Compile extended GEMM
nvcc -o gemm_extended gemm_extended.cu

# Run extended GEMM test
./gemm_extended
```

## Key CUDA Concepts Used

1. **Thread Hierarchy**: Grid → Blocks → Threads
2. **Memory Management**: cudaMalloc, cudaMemcpy, cudaFree
3. **Kernel Launch**: `<<<gridDim, blockDim>>>`
4. **Error Handling**: CUDA_CHECK macro
5. **Device Synchronization**: cudaDeviceSynchronize()

## Performance Considerations

**Current Implementation (Naive)**:
- Simple and correct
- No memory optimizations
- Suitable for learning

**Future Optimizations** (Later weeks):
- Shared memory tiling (Week 3)
- Memory coalescing
- Kernel fusion
- Tensor cores (Week 6)

## References

- Course Textbook: Programming Massively Parallel Processors (PMPP) Chapter 5
- NVIDIA CUDA Programming Guide
- Course lecture notes and diagrams

---

**Submission Date:** January 21, 2026  
**Development Environment:** Google Colab with Tesla T4 GPU
