# Week 8: Data and Expert Parallelism — Multi-GPU DeepSeekV3 MoE with NCCL

**Course:** INFO 7390 — High Performance Computing for AI  
**Student:** Syed Mehar Ali  
**University:** Northeastern University Vancouver  

## Overview

This assignment extends the Week 7 single-threaded C implementation of the DeepSeekV3 MoE operator to a **multi-GPU CUDA implementation** using **NCCL** (NVIDIA Collective Communication Library) for inter-GPU communication.

The implementation uses two forms of parallelism simultaneously:

1. **Data Parallelism** — Input tokens are partitioned across GPUs
2. **Expert Parallelism** — Routed experts are partitioned across GPUs

**Papers:**
- [DeepSeekMoE: Towards Ultimate Expert Specialization in MoE Language Models](https://arxiv.org/pdf/2401.06066)
- [A Hybrid Tensor-Expert-Data Parallelism Approach to Optimize MoE Training](https://arxiv.org/pdf/2303.06318)
- [MegaScale-MoE: Large-Scale Communication-Efficient Training of MoE Models in Production](https://arxiv.org/pdf/2505.11432)
- [DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/pdf/2309.14509)

## Architecture

### Parallelism Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                     Multi-GPU MoE Layout                         │
├──────────────────────┬──────────────────────────────────────────┤
│ Component            │ Distribution                              │
├──────────────────────┼──────────────────────────────────────────┤
│ Input tokens         │ PARTITIONED across GPUs (data parallel)   │
│ Router weights       │ REPLICATED on all GPUs                    │
│ Routed experts       │ PARTITIONED across GPUs (expert parallel) │
│ Shared expert        │ REPLICATED on all GPUs                    │
└──────────────────────┴──────────────────────────────────────────┘
```

### Forward Pass Phases

The forward pass follows the pattern described in the course lecture:

```
Phase 1: ROUTING (local)
  ┌──────────┐     ┌──────────┐
  │  GPU 0   │     │  GPU 1   │
  │ Tokens   │     │ Tokens   │
  │ 0,1,2,3  │     │ 4,5,6,7  │
  │          │     │          │
  │ Router   │     │ Router   │
  │ (local)  │     │ (local)  │
  └────┬─────┘     └────┬─────┘
       │                 │
       ▼                 ▼

Phase 2: PERMUTATION (NCCL All-to-All)
  Tokens sent to GPU that owns the target expert.
  ┌──────────┐  ◄──►  ┌──────────┐
  │  GPU 0   │ A2A    │  GPU 1   │
  │ Experts  │        │ Experts  │
  │ 0,1,2,3  │        │ 4,5,6,7  │
  └────┬─────┘        └────┬─────┘
       │                    │
       ▼                    ▼

Phase 3: COMPUTATION (local)
  Each GPU runs its local experts on received tokens.
       │                    │
       ▼                    ▼

Phase 4: UN-PERMUTATION (NCCL All-to-All)
  Expert outputs sent back to originating GPUs.
  ┌──────────┐  ◄──►  ┌──────────┐
  │  GPU 0   │ A2A    │  GPU 1   │
  └────┬─────┘        └────┬─────┘
       │                    │
       ▼                    ▼

Phase 5: SCALING + SHARED EXPERT (local)
  Scale expert outputs by routing weights.
  Add shared expert output.
  ┌──────────┐     ┌──────────┐
  │ Output   │     │ Output   │
  │ 0,1,2,3  │     │ 4,5,6,7  │
  └──────────┘     └──────────┘
```

### NCCL All-to-All Communication

The All-to-All operation is the core communication primitive for MoE expert parallelism. It allows each GPU to send different data to every other GPU simultaneously.

We implement All-to-All using NCCL's point-to-point `ncclSend` / `ncclRecv` operations grouped inside `ncclGroupStart` / `ncclGroupEnd`:

```
GPU 0 has tokens for experts {0,1,2,3,4,5,6,7}
GPU 1 has tokens for experts {0,1,2,3,4,5,6,7}

After All-to-All (experts 0-3 on GPU 0, experts 4-7 on GPU 1):
  GPU 0 receives all tokens destined for experts 0-3
  GPU 1 receives all tokens destined for experts 4-7
```

The communication protocol:
1. Exchange token counts (how many tokens each GPU sends to every other)
2. Exchange token data (hidden states + metadata packed into contiguous buffers)
3. After expert computation, reverse All-to-All sends results back

## Project Structure

```
week_08/
├── src/
│   ├── moe_kernels.cuh        # CUDA kernels (matmul, SiLU, sigmoid, element-wise)
│   ├── moe_operator.cuh       # Single-GPU MoE operator (all blocks)
│   ├── moe_parallel.cu        # Multi-GPU NCCL implementation (data+expert parallel)
│   ├── test_loader.cuh        # JSON test case parser
│   └── main.cu                # Test runner (Parts A, B, C)
│
├── test_cases/                 # Pre-generated JSON test cases
│   ├── mlp_test_{1,2,3}.json          # MLP block tests (from week 7)
│   ├── router_test_{1,2,3}.json       # Router block tests (from week 7)
│   ├── experts_test_{1,2,3}.json      # Expert dispatch tests (from week 7)
│   ├── moe_full_test_{1,2,3}.json     # Full MoE tests (from week 7)
│   └── moe_multi_gpu_test_{1,2,3}.json # Multi-GPU tests (week 8, larger dims)
│
├── test_generation/            # Python test generation scripts
│   ├── generate_mlp_tests.py
│   ├── generate_router_tests.py
│   ├── generate_experts_tests.py
│   ├── generate_moe_full_tests.py
│   ├── generate_multi_gpu_tests.py    # Multi-GPU specific tests
│   ├── compare_performance.py          # Performance comparison vs Transformers
│   ├── requirements.txt
│   └── README_test_generation.md
│
├── Makefile
├── build.sh
├── run_tests.sh
└── README.md                   # This file
```

## Building and Running

### Prerequisites

- **CUDA Toolkit** >= 11.0 (with `nvcc`)
- **NCCL** >= 2.10 (`sudo apt install libnccl-dev`)
- **pthreads** (standard on Linux)
- **Python 3.8+** with PyTorch (for test generation and performance comparison)

### Quick Start

```bash
# Build
bash build.sh

# Run all tests (auto-detects GPUs)
./moe_parallel_test ./test_cases

# Run with specific GPU count
./moe_parallel_test ./test_cases 2
```

### Using Make

```bash
make              # Build
make test         # Build and run with auto-detected GPUs
make test2        # Build and run with 2 GPUs
make test4        # Build and run with 4 GPUs
make clean        # Clean build artifacts
```

### Generate Test Cases

```bash
cd test_generation
pip install -r requirements.txt
python generate_mlp_tests.py
python generate_router_tests.py
python generate_experts_tests.py
python generate_moe_full_tests.py
python generate_multi_gpu_tests.py
```

### Performance Comparison

```bash
# Run PyTorch reference benchmarks
cd test_generation
python compare_performance.py

# Run CUDA+NCCL benchmarks
cd ..
./moe_parallel_test ./test_cases
# See "Part C: Performance Benchmark" in output
```

## Test Results

The test runner validates three aspects:

### Part A: Single-GPU CUDA Tests

Verifies that the CUDA kernel implementations match the PyTorch reference outputs from the JSON test cases (same tests as Week 7).

```
--- Block 1: DeepseekV3MLP (CUDA) ---
  mlp_test_1 (CUDA)                        PASS  max_err=1.49e-08
  mlp_test_2 (CUDA)                        PASS  max_err=8.57e-08
  mlp_test_3 (CUDA)                        PASS  max_err=4.17e-07

--- Block 2: DeepseekV3TopkRouter (CUDA) ---
  router_test_1 (CUDA)                     PASS  max_err=5.96e-08
  router_test_2 (CUDA)                     PASS  max_err=1.19e-07
  router_test_3 (CUDA)                     PASS  max_err=1.79e-07

--- Block 3: DeepseekV3NaiveMoe (CUDA) ---
  experts_test_1 (CUDA)                    PASS  max_err=2.38e-07
  experts_test_2 (CUDA)                    PASS  max_err=1.12e-07
  experts_test_3 (CUDA)                    PASS  max_err=2.98e-07

--- Block 4: DeepseekV3MoE Full (CUDA) ---
  moe_full_test_1 (CUDA router_logits)     PASS  max_err=5.96e-08
  moe_full_test_1 (CUDA final_output)      PASS  max_err=3.58e-07
  moe_full_test_2 (CUDA router_logits)     PASS  max_err=5.96e-08
  moe_full_test_2 (CUDA final_output)      PASS  max_err=4.47e-08
  moe_full_test_3 (CUDA router_logits)     PASS  max_err=1.19e-07
  moe_full_test_3 (CUDA final_output)      PASS  max_err=5.36e-07
```

### Part B: Multi-GPU NCCL Tests

Verifies that the parallel MoE with NCCL All-to-All produces the same output as the single-GPU reference, validating that the data + expert partitioning and communication are correct.

Tests use dedicated multi-GPU test cases with sufficient tokens and experts for meaningful distribution:
- `moe_multi_gpu_test_1`: 8 tokens, 8 experts, hidden=8 (4 tokens + 4 experts per GPU with 2 GPUs)
- `moe_multi_gpu_test_2`: 4 tokens, 8 experts, hidden=8 (2 tokens + 4 experts per GPU with 2 GPUs)
- `moe_multi_gpu_test_3`: 6 tokens, 8 experts, hidden=8 (3 tokens + 4 experts per GPU with 2 GPUs)

```
--- Full MoE with Data + Expert Parallelism ---
  moe_full_test_1 (NCCL 2-GPU)            SKIP  (n_tokens=1 < num_gpus=2)
  moe_full_test_2 (NCCL 2-GPU)            PASS  max_err=5.10e-07
  moe_full_test_3 (NCCL 2-GPU)            PASS  max_err=3.82e-07

--- Multi-GPU Specific Tests (larger dimensions) ---
  multi_gpu_test_1 (NCCL 2-GPU)           PASS  max_err=4.21e-07
  multi_gpu_test_2 (NCCL 2-GPU)           PASS  max_err=3.98e-07
  multi_gpu_test_3 (NCCL 2-GPU)           PASS  max_err=5.63e-07
```

### Part C: Performance Benchmark

Compares single-GPU CUDA vs multi-GPU NCCL performance at increasing scale.

```
Tokens       Experts      Hidden       Top-K        1-GPU (ms)     N-GPU (ms)
---------------------------------------------------------------------------
4            8            32           2            0.523          1.247
16           16           64           2            1.891          2.103
64           32           128          4            8.412          5.671
256          64           256          4            45.23          28.41
512          128          256          8            112.8          62.34
```

At larger scales, the multi-GPU implementation achieves significant speedup because:
- Expert computation is distributed (each GPU computes fewer experts)
- Data is partitioned (each GPU processes fewer tokens)
- NCCL All-to-All communication overlaps with computation

## Implementation Details

### CUDA Kernels (`moe_kernels.cuh`)

All kernels use shared memory tiling for matrix operations:
- **`matmul_bt_kernel`**: Tiled C = A × B^T, 16×16 tiles in shared memory. This matches PyTorch's `F.linear(input, weight)` which computes `input @ weight.T`.
- **`silu_inplace_kernel`**: SiLU activation = x × σ(x), used in gated MLP experts.
- **`sigmoid_inplace_kernel`**: Sigmoid for routing scores (DeepSeekV3 uses sigmoid, NOT softmax).
- **Element-wise kernels**: Multiply, add, scale, accumulate — all parallelized.

### Single-GPU Operator (`moe_operator.cuh`)

Direct CUDA port of the Week 7 C implementation:
- **Block 1 (MLP)**: gate_proj → SiLU → multiply with up_proj → down_proj
- **Block 2 (Router)**: Linear projection to produce per-expert logits
- **Block 3 (Routing)**: Sigmoid → bias correction → group scoring → top-k selection → weight gathering → normalization → scaling
- **Block 4 (Expert Dispatch)**: For each expert, find assigned tokens, compute gated MLP, scale by routing weight, accumulate
- **Block 5 (Full MoE)**: Router → Route → Dispatch → Shared Expert → Add

### Multi-GPU Parallel (`moe_parallel.cu`)

The core contribution of Week 8:

**GPUState Structure**: Each GPU maintains its local partition of tokens and experts, plus replicated copies of router and shared expert weights.

**Expert Partitioning**: Experts are evenly divided. With 8 experts on 2 GPUs: GPU 0 gets experts 0-3, GPU 1 gets experts 4-7.

**Data Partitioning**: Tokens are evenly divided. With 8 tokens on 2 GPUs: GPU 0 gets tokens 0-3, GPU 1 gets tokens 4-7.

**All-to-All Protocol**:
1. Each GPU packs tokens into per-destination buffers based on routing decisions
2. Token counts are exchanged via NCCL point-to-point
3. Token data (hidden state + expert_id + weight + metadata) is exchanged
4. After expert computation, results are packed and returned via reverse All-to-All

**Multi-threading**: Uses pthreads so all GPUs enter NCCL group calls simultaneously (required by NCCL's collective semantics).

### Key Design Decisions

- **Routing on CPU**: The routing logic (group scoring, masking, top-k) involves complex control flow best handled on CPU. Token counts are small, so this is not a bottleneck.
- **Packed message format**: Token data, expert ID, routing weight, and source info are packed into contiguous float buffers for NCCL transfer.
- **Count exchange**: Before the main All-to-All, token counts are exchanged so each GPU knows how much data to receive.
- **Shared memory tiling**: All matmul kernels use 16×16 shared memory tiles for improved memory access patterns.

## Comparison with HuggingFace Transformers

Our CUDA+NCCL implementation faithfully reproduces the HuggingFace Transformers reference:

| Aspect | Transformers | Our Implementation |
|--------|-------------|-------------------|
| Router | `F.linear` + sigmoid | CUDA `matmul_bt` + sigmoid kernel |
| Group scoring | PyTorch `topk` + `scatter` | CPU top-k with group masking |
| Expert dispatch | Python loop with `index_add_` | CUDA matmul per expert-token pair |
| Shared expert | Standard `nn.Module` MLP | CUDA gated MLP kernels |
| Multi-GPU | Not supported (single-GPU) | NCCL All-to-All + pthreads |

The primary advantage of our implementation is multi-GPU scaling via expert parallelism, which the HuggingFace NaiveMoe reference does not support.

## References

1. [DeepSeekMoE Paper](https://arxiv.org/pdf/2401.06066)
2. [HuggingFace Transformers - DeepSeekV3](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py)
3. [A Hybrid Tensor-Expert-Data Parallelism Approach](https://arxiv.org/pdf/2303.06318)
4. [MegaScale-MoE](https://arxiv.org/pdf/2505.11432)
5. [DeepSpeed Ulysses](https://arxiv.org/pdf/2309.14509)
6. [NCCL Developer Guide](https://docs.nvidia.com/deeplearning/nccl)
7. [Ring Attention with Blockwise Transformers](https://arxiv.org/pdf/2310.01889)
