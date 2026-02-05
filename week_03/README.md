# Week 3: CUDA Matrix Multiplication Optimization

**Author:** Syed Mehar Ali  
**Course:** INFO 7390 - High Performance Computing for AI  
**University:** Northeastern University Vancouver

## Assignment Overview

Replicated the CUDA matrix multiplication optimization worklog "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance" on H100 GPU via Modal.com.

**Requirements Completed:**
- ✅ Read through complete optimization worklog
- ✅ Implemented naive and optimized CUDA kernels
- ✅ Ran benchmarks on H100 GPU
- ✅ Analyzed performance improvements

## Environment

- **Platform**: Modal.com Cloud
- **GPU**: NVIDIA H100 80GB HBM3
- **Compute Capability**: 9.0
- **CUDA Version**: 12.3.0
- **Matrix Size**: 4096 × 4096 × 4096

## Implementation

### Kernel 0: Naive Implementation
Basic implementation with each thread computing one output element.
- Direct global memory access
- No optimization
- Baseline for comparison

### Kernel 2: Shared Memory Tiling
Optimized version using shared memory caching.
- **Tile dimensions**: 64×64×8 (BM×BN×BK)
- **Thread tile**: 8 elements per thread (TM)
- Shared memory caching reduces global memory traffic
- Memory coalescing for efficient access

## Performance Results

### Benchmark Configuration
- Matrix dimensions: 4096×4096×4096
- Precision: FP32 (single precision)
- Runs: 100 iterations per kernel
- GPU: H100 80GB HBM3

### Results

| Kernel | Time (ms) | Performance | Speedup |
|--------|-----------|-------------|---------|
| Naive | 276.092 | 0.50 TFLOPS | 1.00x (baseline) |
| Shared Memory Tiling | 24.554 | 5.60 TFLOPS | **11.24x** |

### Analysis

**Why 11.24x speedup?**

1. **Reduced Global Memory Access**: 
   - Naive: Every thread reads from global memory K times
   - Optimized: Each block loads tiles into shared memory, reuses data

2. **Memory Bandwidth Utilization**:
   - Shared memory: ~19 TB/s (H100)
   - Global memory: ~3 TB/s
   - Shared memory is ~6x faster

3. **Data Reuse**:
   - Each element loaded from global memory is reused by multiple threads
   - Tile size 64×64 means each loaded element is reused 64 times

4. **Memory Coalescing**:
   - Threads access consecutive memory locations
   - Maximizes memory transaction efficiency

## Key Concepts Learned

### 1. Memory Hierarchy
- **Global Memory (HBM)**: Largest (85 GB), slowest (~3 TB/s)
- **Shared Memory**: Medium (48 KB/block), fast (~19 TB/s)
- **Registers**: Smallest, fastest

### 2. Tiling Strategy
Breaking computation into tiles that fit in shared memory:
- Block tiles: 64×64 output elements
- Each block processes 8 elements of K dimension at a time
- Threads cooperatively load tiles

### 3. Thread Organization
- Threads per block: (BM × BN) / TM = (64 × 64) / 8 = 512 threads
- Each thread computes 8 output elements
- Efficient resource utilization

### 4. Optimization Progression
```
Naive (0.50 TFLOPS)
    ↓ Add shared memory caching
Optimized (5.60 TFLOPS) - 11.24x faster
    ↓ Further optimizations possible:
        - 1D/2D blocktiling
        - Vectorized loads
        - Warp-level primitives
        → Can reach 20-30+ TFLOPS
```

## Code Structure
```
week_03/
├── week3_complete.py       # Modal.com deployment script
├── week_03_README.md       # This file
└── (embedded CUDA code in Python file)
```

## Compilation and Execution

### Via Modal.com (Used for Assignment)
```bash
pip install modal
modal setup
modal run week3_complete.py
```

### Local Compilation (if CUDA available)
```bash
nvcc -O3 -arch=sm_90 matmul.cu -o matmul
./matmul
```

## FLOP Calculations

For `C = α·AB + β·C` where A is (M×K), B is (K×N):

**Total FLOPs**: 2 × M × N × K ≈ 2 × 4096³ = 137.44 billion

**At 5.60 TFLOPS**: 137.44 GFLOPS / 5.60 TFLOPS = 24.5 ms (matches result!)

## Connection to Course Topics

### Week 3 Focus: Memory Hierarchy
- Demonstrated practical impact of memory levels
- Quantified performance differences
- Understood tiling as a fundamental optimization

### Relation to Reading
"How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance":
- Implemented Kernel 0 (naive) and Kernel 2 (shared memory)
- Observed expected performance improvements
- Validated optimization strategies from worklog

### Future Work
Week 4+ will build on these concepts:
- FlashAttention uses similar tiling strategies
- Tensor cores (Week 6) provide further acceleration
- Multi-GPU (Week 7) extends tiling across devices

## References

1. **Primary**: "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance" by Simon Boehm
   - https://siboehm.com/articles/22/CUDA-MMM

2. **PMPP Chapter 5**: Memory Architecture and Data Locality

3. **NVIDIA H100 Architecture White Paper**

4. **CUDA Programming Guide**: Shared Memory, Memory Coalescing

## Conclusion

Successfully implemented and benchmarked CUDA matrix multiplication optimizations on H100 GPU, achieving **11.24x speedup** through shared memory tiling. This demonstrates understanding of:
- GPU memory hierarchy
- Tiling and blocking strategies
- Performance optimization techniques
- Practical CUDA programming on modern hardware

---

**Date:** February 4, 2026  
**Platform:** Modal.com H100 GPU  
**Status:** All requirements completed ✅
```


```
https://github.com/YOUR_USERNAME/neu-hpc-for-ai/tree/main/week_03
