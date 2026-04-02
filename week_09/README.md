# Week 9: Sequence Parallelism - DeepSeekV3 MoE with ThunderKittens

**Course:** INFO 7390 - High Performance Computing for AI  
**Student:** Syed Mehar Ali  
**University:** Northeastern University Vancouver  

## Overview

This assignment reimplements the DeepSeekV3 MoE operator (from Week 7) using
**ThunderKittens (TK)** to leverage **WMMA tensor cores** and **TMA (Tensor Memory
Accelerator)** on NVIDIA B200 GPUs, then measures the performance improvement
compared to the baseline implementation.

## Architecture

### Baseline (Week 7 equivalent)
- Naive CUDA GEMM on CUDA cores (no tensor cores)
- Standard `cudaMemcpy` for global↔shared memory transfers
- fp32 or bf16 with scalar FMA instructions

### ThunderKittens Implementation

#### Level 04: WMMA Tensor Cores
- Uses TK's `warp::mma_AB()` for hardware matrix multiply-accumulate
- 32×32 tile-based computation using `st_bf` / `rt_bf` / `rt_fl` tiles
- `warp::load()` for global→shared→register data movement
- `warp::swap_layout()` to prepare B operand in col-major for tensor cores

#### Level 05: WMMA + TMA
- Adds TMA hardware unit for async global↔shared memory transfers
- `tma::load_async()` fires off transfer without occupying any threads
- `tma::expect_bytes()` + semaphore-based synchronization
- `tma::store_async()` for result writeback
- Frees compute warps to do MMA while TMA handles all data movement

### MoE Blocks (same as Week 7)

| Block | Operation | TK Acceleration |
|-------|-----------|-----------------|
| **MLP** | `down_proj(silu(gate_proj(x)) * up_proj(x))` | 3× WMMA+TMA GEMM |
| **Router** | `F.linear(hidden_states, weight)` | 1× WMMA+TMA GEMM |
| **Experts** | Per-expert gate-up-down with routing weights | N× WMMA+TMA GEMM |
| **Full MoE** | Router + routing + experts + shared experts | All above combined |

## Project Structure

```
week_09/
├── src/
│   ├── moe_tk_gemm.cuh    # TK GEMM kernels (Level 04 + Level 05)
│   ├── moe_tk_ops.cuh     # Element-wise ops (SiLU, sigmoid, mul, add)
│   └── moe_tk.cuh         # High-level MoE blocks using TK GEMM
│
├── baseline/
│   └── baseline_gemm.cuh  # Naive CUDA GEMM (no tensor cores)
│
├── benchmark/
│   └── benchmark.cu       # Performance measurement program
│
├── scripts/
│   └── run_benchmark.py   # Modal.com runner for B200/H100
│
├── Makefile
└── README.md
```

## Building

### Prerequisites
- CUDA 12.3+ (we use 12.6)
- ThunderKittens (cloned from GitHub)
- B200 or H100 GPU
- C++20 capable compiler (gcc-10+)

### Build
```bash
export THUNDERKITTENS_ROOT=/path/to/ThunderKittens
make GPU_TARGET=B200   # or H100
```

### Run on Modal.com
```bash
cd week_09/scripts
modal run run_benchmark.py
```

## Performance Results

### GEMM Performance (C = A @ B^T)

| N | Naive CUDA (fp32) | TK WMMA (bf16) | TK WMMA+TMA (bf16) | Speedup |
|---|---|---|---|---|
| 256 | ~0.5 TFLOPs | ~5 TFLOPs | ~8 TFLOPs | ~16x |
| 512 | ~1 TFLOPs | ~12 TFLOPs | ~20 TFLOPs | ~20x |
| 1024 | ~2 TFLOPs | ~20 TFLOPs | ~35 TFLOPs | ~18x |
| 2048 | ~3 TFLOPs | ~24 TFLOPs | ~45 TFLOPs | ~15x |

*(Expected results - actual numbers depend on specific GPU)*

### Key Performance Insights

1. **WMMA tensor cores** provide the largest speedup (~4-10x over naive) by
   replacing scalar FMA with hardware matrix multiply units that process
   entire 16×16 tiles in a single instruction.

2. **TMA** adds another ~1.5-2x on top of WMMA by offloading all memory
   address calculation and data movement to dedicated hardware, freeing
   the compute warps to issue more MMA instructions.

3. **bf16 precision** halves memory bandwidth requirements compared to fp32,
   which is critical for bandwidth-bound operations.

4. The MoE MLP (3 GEMMs + element-wise ops) sees slightly less than 3× the
   GEMM speedup because the element-wise SiLU and multiply operations are
   memory-bound and don't benefit from tensor cores.

## ThunderKittens API Summary

### Tile Types
- `st_bf<M,N>`: Shared memory tile, bf16, M×N
- `rt_bf<M,N>`: Register tile, bf16, M×N, row layout
- `rt_bf<M,N,col>`: Register tile, bf16, col layout (for tensor core B operand)
- `rt_fl<M,N>`: Register tile, float32 (accumulator)

### Key Operations
- `warp::load(shared, global, {coords})` — Global → Shared
- `warp::load(reg, shared)` — Shared → Register
- `warp::swap_layout(dst_col, src_row)` — Row → Col layout for MMA
- `warp::mma_AB(C, A, B, C)` — Tensor core MMA: C += A × B
- `warp::store(shared, reg)` — Register → Shared
- `tma::load_async(shared, global, {coords}, sem)` — TMA async load
- `tma::store_async(global, shared, {coords})` — TMA async store
- `tma::expect_bytes(sem, nbytes)` — Set TMA completion threshold
- `wait(sem, phase)` — Block until TMA completes

## References

1. [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) - Tile primitives for speedy kernels
2. [Blockwise Parallel Transformer](https://arxiv.org/pdf/2305.19370) - BPT for large context
3. [Ring Attention](https://arxiv.org/pdf/2310.01889) - Near-infinite context via ring communication
4. [DeepSeekMoE Paper](https://arxiv.org/pdf/2401.06066) - MoE architecture
5. [NVIDIA HGX H100](https://developer.nvidia.com/blog/introducing-nvidia-hgx-h100-an-accelerated-server-platform-for-ai-and-high-performance-computing)
