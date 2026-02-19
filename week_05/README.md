# Week 5: FlashAttention with CuTe Layouts

**Author:** Syed Mehar Ali  
**Course:** INFO 7390 - High Performance Computing for AI  
**University:** Northeastern University Vancouver

## Assignment Overview

Reimplemented FlashAttention-2 Algorithm 1 using CuTe (CUDA C++ Template) library concepts, demonstrating understanding of Layout abstractions for memory access.

**Assignment Requirements Completed:**
- ✅ Read CuTe library documentation (Sections 0-4)
- ✅ Understood Layout<Shape, Stride> abstraction
- ✅ Reimplemented FlashAttention Algorithm 1 using CuTe concepts
- ✅ Applied coordinate mapping and tensor abstractions
- ✅ Verified correctness on H100 GPU

## Files

- `flashattention_cute_simple.cu` - FlashAttention with CuTe-inspired abstractions
- `week5_modal_deploy.py` - Modal.com H100 deployment
- `week_05_README.md` - This documentation

## CuTe Library Fundamentals

### What is CuTe?

**CuTe** (CUDA Templates) is NVIDIA's C++ template library for writing high-performance CUDA kernels, part of the CUTLASS library.

**Purpose:**
- Abstract away low-level memory indexing
- Provide composable building blocks
- Enable portability across GPU architectures
- Support tensor core programming

**Core Abstraction:** `Layout = (Shape, Stride)`

### Layout: The Core Concept

**Definition:** A Layout maps from **coordinate space** to **index space**
```cpp
Layout<Shape, Stride>

index = inner_product(coordinate, stride)
```

**Components:**

1. **Shape**: Defines the coordinate space dimensions
   - Example: `Shape(2, 3)` = 2 rows, 3 columns
   - Creates coordinates: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)

2. **Stride**: Defines memory layout
   - Example: `Stride(1, 2)` = column-major
   - Maps coordinate (i,j) to index i*1 + j*2

### Layout Examples from Reading

#### Example 1: Column-Major (2,3):(1,2)
```
Logical:        Physical Memory:
[a b c]         [a d b e c f]
[d e f]         
                idx(i,j) = i*1 + j*2
                (0,0) → 0
                (1,0) → 1  
                (0,1) → 2
```

**Code:**
```cpp
auto layout = make_layout(make_shape(2, 3),
                          make_stride(1, 2));
// Column-major: contiguous in rows
```

#### Example 2: Row-Major (2,3):(3,1)
```
Logical:        Physical Memory:
[a b c]         [a b c d e f]
[d e f]         
                idx(i,j) = i*3 + j*1
                (0,0) → 0
                (0,1) → 1
                (1,0) → 3
```

**Code:**
```cpp
auto layout = make_layout(make_shape(2, 3),
                          make_stride(3, 1));
// Row-major: contiguous in columns
```

#### Example 3: Padded (2,3):(1,4)
```
Logical:        Physical Memory:
[a b c]         [a d _ _ b e _ _ c f _ _]
[d e f]         
                idx(i,j) = i*1 + j*4
                Padding between columns!
```

**Use Case:** Avoid bank conflicts in shared memory

#### Example 4: Hierarchical ((2,2),2)
```
Shape: ((2,2), 2) - Nested/hierarchical
Stride: ((4,2), 1)

Coordinate: (i, (j0, j1))
Index: 4*i + 2*j0 + 1*j1
```

**Use Case:** Tiled matrix layouts

### Tensor: Layout + Data

**Definition:** `Tensor` combines a pointer with a `Layout`
```cpp
Tensor<float, Layout> tensor(ptr, layout);

// Access via coordinates (not manual indexing!)
float val = tensor(i, j);  // Instead of ptr[i*stride + j]
```

**Benefits:**
- Cleaner code
- Type-safe
- Compiler can optimize
- Reusable across different layouts

## Implementation Using CuTe Concepts

### Key CuTe Patterns Applied

#### 1. Layout Definitions
```cpp
// Column-major layout for Qi (Br × d)
auto layout_Qi = make_colmajor_layout<Br, D>();
// Internally: Shape(Br, D), Stride(1, Br)

// Row-major layout for output
auto layout_O = make_rowmajor_layout<N, D>();
// Internally: Shape(N, D), Stride(D, 1)
```

#### 2. Tensor Creation
```cpp
// Create tensor from shared memory pointer + layout
Tensor Qi(smem_ptr, layout_Qi);

// Now access via coordinates:
float val = Qi(r, k);  // Instead of smem_ptr[r + k*Br]
```

#### 3. Coordinate-Based Access

**Before (raw pointers):**
```cpp
float val = smem[row * Br + col];  // Manual index calculation
```

**After (CuTe Tensor):**
```cpp
float val = Qi(row, col);  // Layout handles indexing
```

#### 4. Memory Layout Flexibility

**Same algorithm, different layouts:**
```cpp
// Column-major for coalescing
Tensor Qi_col(ptr, make_colmajor_layout<Br, D>());

// Row-major for different access pattern
Tensor Qi_row(ptr, make_rowmajor_layout<Br, D>());

// Padded for bank conflict avoidance
Tensor Qi_pad(ptr, Layout<Shape<Br,D>, Stride<1,Br+1>>{});
```

### FlashAttention with CuTe Patterns

**Algorithm Structure:**
```cpp
// Define layouts for all shared memory tensors
auto layout_Qi = make_layout(make_shape(Br, d), make_stride(1, Br));
auto layout_Kj = make_layout(make_shape(Bc, d), make_stride(1, Bc));
auto layout_Sij = make_layout(make_shape(Br, Bc), make_stride(1, Br));

// Create Tensors with layouts
Tensor Qi(smem_qi_ptr, layout_Qi);
Tensor Kj(smem_kj_ptr, layout_Kj);
Tensor Sij(smem_sij_ptr, layout_Sij);

// Algorithm logic with clean coordinate access
for (int r = 0; r < Br; r++) {
    for (int c = 0; c < Bc; c++) {
        float sum = 0.0f;
        for (int k = 0; k < d; k++) {
            sum += Qi(r, k) * Kj(c, k);  // Clean! No manual indexing
        }
        Sij(r, c) = sum * scale;
    }
}
```

## Advantages of CuTe Approach

### 1. Abstraction Over Memory Layout

**Problem without CuTe:**
```cpp
// Hard-coded indexing, tied to specific layout
float val = array[row * num_cols + col];  // Row-major assumed

// To change to column-major, must rewrite ALL accesses!
float val = array[col * num_rows + row];  // Different everywhere
```

**Solution with CuTe:**
```cpp
Tensor tensor(ptr, layout);
float val = tensor(row, col);  // Works with any layout!

// Change layout in ONE place:
auto layout = make_colmajor_layout<Rows, Cols>();  // or
auto layout = make_rowmajor_layout<Rows, Cols>();
```

### 2. Compile-Time Optimization

**CuTe uses C++ templates:**
- Shapes and strides can be compile-time constants
- Compiler unrolls loops
- Zero runtime overhead
- Optimizes away abstraction

**Example:**
```cpp
Shape<Int<32>, Int<64>>  // Compile-time: 32×64
vs
Shape<int, int>          // Runtime: dynamic size
```

### 3. Portability Across GPU Architectures

**Different GPUs need different patterns:**
- A100: MMA.SYNC instructions
- H100: WGMMA.ASYNC instructions  
- B200: TCGEN05 instructions

**CuTe allows:**
```cpp
// Same high-level code
Tensor A(ptr, layout);
gemm(A, B, C);  // CuTe picks right implementation for GPU!
```

### 4. Composability

**CuTe operations compose:**
```cpp
auto layout_full = make_layout(Shape(N, d), Stride(d, 1));

// Tile into blocks
auto layout_tile = local_tile(layout_full, Shape(Br, d));

// Hierarchical composition
auto layout_nested = make_layout(
    Shape((Br/4, 4), d),    // Nested shape
    Stride((4, 1), Br)      // Nested stride
);
```

## Test Results

### Configuration
- N = 256 (sequence length)
- d = 64 (embedding dimension)
- Br = 32 (row blocks)
- Bc = 32 (column blocks)

### Execution Details
- GPU: H100 80GB HBM3
- Thread blocks: 8 (one per Qi)
- Threads per block: 256
- Shared memory: ~36 KB per block

### Correctness Verification

**Results:**
```
Max difference: < 1e-6
Errors: 0 / 16384 elements
Status: ✓ TEST PASSED
```

**Conclusion:** CuTe implementation produces identical results to standard attention.

## Comparison: Week 4 vs Week 5

| Aspect | Week 4 (Raw CUDA) | Week 5 (CuTe) |
|--------|-------------------|---------------|
| **Memory Access** | Manual: `ptr[i*stride + j]` | Abstracted: `tensor(i,j)` |
| **Layout Changes** | Rewrite all access | Change Layout definition |
| **Readability** | Lower-level | Higher-level |
| **Optimization** | Manual | Compiler-assisted |
| **Portability** | GPU-specific | Architecture-agnostic |
| **Complexity** | Simpler (plain CUDA) | More complex (C++ templates) |

**Both produce same results - CuTe adds abstraction layer!**

## CuTe Layout Algebra (From Reading)

### Flattening and Reshaping

**Example:** 2×2×2 tensor can be viewed as 4×2 matrix
```
Shape: (2,2,2)           →  Shape: ((2,2), 2)
Stride: (4,1,2)          →  Stride: ((4,2), 1)

Coordinate: (i,j,k)      →  Coordinate: ((i,j), k)
Index: 4i + 1j + 2k      →  Index: 4i + 2j + 1k (same!)
```

**CuTe supports hierarchical shapes naturally!**

### Coordinate Mapping Examples

From lecture diagrams:

**Column-major (4,8):(1,4):**
```
(0,0)→0  (0,1)→4  (0,2)→8  ...
(1,0)→1  (1,1)→5  (1,2)→9  ...
(2,0)→2  (2,1)→6  (2,2)→10 ...
(3,0)→3  (3,1)→7  (3,2)→11 ...
```

**Row-major (4,8):(8,1):**
```
(0,0)→0  (0,1)→1  (0,2)→2  ...
(1,0)→8  (1,1)→9  (1,2)→10 ...
(2,0)→16 (2,1)→17 (2,2)→18 ...
```

**CuTe handles both with same code, different Layout!**

## Key Learning Outcomes

### 1. Layout Abstraction
- ✅ Separate logical view from physical layout
- ✅ Coordinate space vs index space
- ✅ Shape defines coordinates, Stride defines mapping

### 2. Tensor Programming
- ✅ Tensor = data + metadata (Layout)
- ✅ High-level operations on tensors
- ✅ Compiler optimizes abstractions away

### 3. Memory Access Patterns
- ✅ Column-major vs row-major
- ✅ Padded layouts for performance
- ✅ Hierarchical layouts for tiling

### 4. C++ Template Metaprogramming
- ✅ Compile-time vs runtime values
- ✅ `Int<N>{}` for static constants
- ✅ Template specialization for optimization

### 5. FlashAttention with Abstractions
- ✅ Applied Layout to all memory regions
- ✅ Tensor-based coordinate access
- ✅ Cleaner, more maintainable code

## Compilation

### Standalone (if CUTLASS available)
```bash
# With full CUTLASS/CuTe
nvcc -O3 -arch=sm_90 -std=c++17 \
     -I /path/to/cutlass/include \
     flashattention_cute.cu -o flash_cute
```

### Simplified Version (our implementation)
```bash
# Self-contained Layout implementation
nvcc -O3 -arch=sm_90 -std=c++17 \
     flashattention_cute_simple.cu -o flash_cute
```

### Modal.com (H100)
```bash
modal run week5_modal_deploy.py
```

## Why CuTe Matters for HPC

### 1. Performance Portability
Write once, run optimally on:
- Volta (sm_70)
- Turing (sm_75)
- Ampere (sm_80)
- Hopper (sm_90)
- Blackwell (sm_100)

### 2. Tensor Core Programming
CuTe simplifies tensor core usage:
```cpp
// Without CuTe: 100+ lines of architecture-specific code
// With CuTe: Few lines, works across architectures
cute::gemm(mma_atom, A, B, C);
```

### 3. Production Use
Used in:
- cuBLAS (NVIDIA's BLAS library)
- cuDNN (Deep learning primitives)
- FlashAttention-2/3 (official implementations)
- CUTLASS GEMM kernels

### 4. Maintainability
- Change layout in one place
- Algorithm code stays same
- Easier to optimize and debug

## Connection to Course Topics

### Week 3: Memory Optimization
- CuTe makes tiling easier to express
- Layout handles index calculations
- Focus on algorithm, not indexing

### Week 4: FlashAttention
- Same algorithm, better abstraction
- Layout for each memory region
- Clearer code structure

### Week 5: CuTe Layouts
- **Core concept**: Separate logic from layout
- Applied to real algorithm (FlashAttention)
- Demonstrated Layout algebra

### Future Weeks
- Week 6: Tensor cores (CuTe makes this easier!)
- Week 7+: Advanced CuTe features

## CuTe vs Raw CUDA

### Raw CUDA (Week 4)
```cpp
// Manual indexing everywhere
float val = smem[row * Br + col];
Sij[r * Bc + c] = sum;
Oi[r * d + k] = result;

// Error-prone, hard to change layout
```

### CuTe (Week 5)
```cpp
// Coordinate-based access
float val = Qi_smem(row, col);
Sij(r, c) = sum;
Oi(r, k) = result;

// Easy to change layout, safer, clearer
```

**Same computation, better expression!**

## References

### Primary Sources

1. **CuTe Documentation** (Required Reading)
   - https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/
   - Sections 0-4 (as assigned)
   - Layout fundamentals and examples

2. **CUTLASS Repository**
   - https://github.com/NVIDIA/cutlass
   - CuTe source code and examples

### Supporting Materials

3. **FlashAttention-2 Paper**
   - Algorithm 1 (what we reimplemented)
   - Section 3.1

4. **C++ Templates**
   - Template metaprogramming concepts
   - Compile-time computation

5. **Course Materials**
   - Week 5 lecture: Layout algebra
   - Diagrams of coordinate mapping

## Learning Outcomes

### Technical Skills
1. ✅ Read and understood template library documentation
2. ✅ Applied Layout abstractions to real algorithm
3. ✅ Created custom Layout types
4. ✅ Used Tensor for memory access
5. ✅ Maintained algorithm correctness with new abstraction

### Conceptual Understanding
1. ✅ Separation of concerns: logic vs layout
2. ✅ Coordinate space vs index space mapping
3. ✅ Compile-time vs runtime computation
4. ✅ Abstraction without performance cost
5. ✅ How production CUDA code is structured

### Practical Benefits
1. ✅ Can read CUTLASS/CuTe code
2. ✅ Understand modern CUDA libraries
3. ✅ Apply abstractions to own kernels
4. ✅ Appreciate why libraries like CuTe exist

## Future Work

**Not implemented (beyond scope):**
- Full CUTLASS integration
- Tensor core operations via CuTe
- Copy atoms and tiled copy
- MMA (Matrix Multiply-Accumulate) atoms
- Swizzling for bank conflict avoidance
- Advanced layout composition

**These are covered in advanced CUTLASS tutorials, beyond this course.**

## Conclusion

Successfully reimplemented FlashAttention Algorithm 1 using CuTe Layout abstractions, demonstrating:

- ✅ Understanding of Layout<Shape, Stride> concept
- ✅ Ability to apply abstractions to complex algorithm
- ✅ Knowledge of coordinate → index mapping
- ✅ Appreciation for modern CUDA programming patterns

The implementation maintains correctness while showcasing how high-level abstractions can simplify GPU programming without sacrificing performance.

---

**Date:** February 18, 2026  
**Platform:** Modal.com H100 GPU  
**Status:** All requirements completed ✅
```

