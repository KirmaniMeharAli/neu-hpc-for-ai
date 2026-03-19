# Week 7: Communication - DeepSeekV3 MoE Operator Implementation

**Course:** INFO 7390 - High Performance Computing for AI  
**Student:** Syed Mehar Ali  
**University:** Northeastern University Vancouver  

## Overview

This assignment implements the Mixture-of-Experts (MoE) operator from DeepSeekV3 in pure C (no CUDA, no parallelism), with test cases generated from the HuggingFace Transformers reference implementation.

**Paper:** [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/pdf/2401.06066)

## DeepSeekV3 MoE Architecture

The MoE operator in DeepSeekV3 consists of 4 main blocks:

### Block 1: DeepseekV3MLP (Gated MLP)
The basic feedforward block used both as individual experts and as shared experts.
```
output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
```
- `gate_proj`: Linear(hidden_size → intermediate_size, no bias)
- `up_proj`: Linear(hidden_size → intermediate_size, no bias)  
- `down_proj`: Linear(intermediate_size → hidden_size, no bias)
- Activation: SiLU (Sigmoid Linear Unit) = x * sigmoid(x)

### Block 2: DeepseekV3TopkRouter
Computes router logits for expert selection via a linear projection.
```
router_logits = F.linear(hidden_states.float32, weight.float32)
```
- Weight shape: (n_routed_experts, hidden_size)
- Both inputs and weights are cast to float32

### Block 3: DeepseekV3NaiveMoe (Expert Dispatch)
Holds all expert weights as 3D tensors and dispatches tokens to selected experts.
- `gate_up_proj`: (num_experts, 2 * intermediate_dim, hidden_dim)
- `down_proj`: (num_experts, hidden_dim, intermediate_dim)
- For each expert: find assigned tokens → compute gated MLP → scale by routing weight → accumulate

### Block 4: DeepseekV3MoE (Full MoE with Routing)
Combines all blocks with the routing algorithm:
1. **Router**: Compute logits via linear projection
2. **Routing** (`route_tokens_to_experts`):
   - Apply sigmoid to logits
   - Add bias correction (`e_score_correction_bias`)
   - Group experts into `n_group` groups
   - Score each group by top-2-in-group sum
   - Select `topk_group` groups
   - Pick `top_k` experts from selected groups
   - Gather weights from original sigmoid scores
   - Normalize if `norm_topk_prob` is True
   - Scale by `routed_scaling_factor`
3. **Expert dispatch**: Route tokens through selected experts
4. **Shared experts**: Apply shared MLP to ALL tokens
5. **Combine**: output = expert_output + shared_expert_output

## Project Structure

```
week_07/
├── test_generation/
│   ├── requirements.txt              # Python dependencies
│   ├── setup_env.sh / setup_env.bat  # Environment setup scripts
│   ├── print_config.py               # Print DeepSeekV3Config parameters
│   ├── generate_mlp_tests.py         # MLP block test generation
│   ├── generate_router_tests.py      # Router block test generation
│   ├── generate_experts_tests.py     # NaiveMoe (expert dispatch) tests
│   ├── generate_moe_full_tests.py    # Full MoE test generation
│   ├── verify_determinism.py         # Verify test generation is deterministic
│   └── README_test_generation.md     # Test generation documentation
│
├── test_cases/                       # Generated test cases (JSON)
│   ├── mlp_test_1.json ... mlp_test_3.json
│   ├── router_test_1.json ... router_test_3.json
│   ├── experts_test_1.json ... experts_test_3.json
│   └── moe_full_test_1.json ... moe_full_test_3.json
│
├── src/
│   ├── moe_operator.h / .c          # All MoE blocks (main implementation)
│   ├── matrix_ops.h / .c            # Matrix multiplication helpers
│   ├── activations.h / .c           # SiLU, Sigmoid activation functions
│   ├── test_loader.h / .c           # JSON parser and test comparison
│   └── main.c                        # Test runner
│
├── Makefile                          # Build system
├── build.sh / build.bat              # Build scripts
├── run_tests.sh / run_tests.bat      # Test execution scripts
└── README.md                         # This file
```

## Building and Running

### Quick Start (Linux/Mac)
```bash
# Build
make

# Run tests
make test
```

### Quick Start (Windows)
```cmd
build.bat
run_tests.bat
```

### Step-by-Step

1. **Generate test cases** (requires Python + PyTorch):
```bash
cd test_generation
bash setup_env.sh
source venv/bin/activate
python generate_mlp_tests.py
python generate_router_tests.py
python generate_experts_tests.py
python generate_moe_full_tests.py
python verify_determinism.py
```

2. **Build C implementation**:
```bash
cd ..
make
```

3. **Run verification**:
```bash
./moe_test ./test_cases
```

## Test Results

```
============================================================
DeepSeekV3 MoE Operator - C Implementation Test Runner
============================================================
Tolerance: 1e-04
============================================================

--- Block 1: DeepseekV3MLP ---
  mlp_test_1                      PASS  max_err=1.49e-08  mean_err=9.31e-09
  mlp_test_2                      PASS  max_err=8.57e-08  mean_err=1.99e-08
  mlp_test_3                      PASS  max_err=4.17e-07  mean_err=1.20e-07

--- Block 2: DeepseekV3TopkRouter ---
  router_test_1                   PASS  max_err=5.96e-08  mean_err=1.07e-08
  router_test_2                   PASS  max_err=1.19e-07  mean_err=2.01e-08
  router_test_3                   PASS  max_err=1.79e-07  mean_err=5.54e-08

--- Block 3: DeepseekV3NaiveMoe (Experts) ---
  experts_test_1                  PASS  max_err=2.38e-07  mean_err=8.52e-08
  experts_test_2                  PASS  max_err=1.12e-07  mean_err=2.11e-08
  experts_test_3                  PASS  max_err=2.98e-07  mean_err=9.66e-08

--- Block 4: DeepseekV3MoE (Full) ---
  moe_full_test_1 (router_logits) PASS  max_err=5.96e-08  mean_err=2.33e-08
  moe_full_test_1 (final_output)  PASS  max_err=3.58e-07  mean_err=1.79e-07
  moe_full_test_2 (router_logits) PASS  max_err=5.96e-08  mean_err=1.66e-08
  moe_full_test_2 (final_output)  PASS  max_err=4.47e-08  mean_err=1.89e-08
  moe_full_test_3 (router_logits) PASS  max_err=1.19e-07  mean_err=4.18e-08
  moe_full_test_3 (final_output)  PASS  max_err=5.36e-07  mean_err=1.01e-07

============================================================
RESULTS: 15 / 15 tests passed
============================================================
```

All 15 tests pass with maximum error ~5e-7, well within the 1e-4 tolerance.

## Implementation Notes

### Design Decisions
- **Pure C99**: No C++, no external libraries beyond standard C math library
- **Single-threaded**: No pthreads, OpenMP, or any parallelism
- **F.linear mapping**: PyTorch's `F.linear(x, W)` computes `x @ W^T`, implemented as `matmul_bt()`
- **JSON test format**: Human-readable, easy to debug, nested arrays flattened to row-major
- **Separate seeds**: Weight and input generation use different seeds for independence

### Key Architecture Insights from the Paper
- DeepSeekV3 uses **sigmoid** (not softmax) for routing scores
- The routing algorithm includes **group-level selection** before expert selection
- **Bias correction** (`e_score_correction_bias`) is added for load balancing but weights are gathered from uncorrected sigmoid scores
- **Shared experts** process ALL tokens in parallel with routed experts
- The scaling factor compensates for the normalized routing weights

## References
1. [DeepSeekMoE Paper](https://arxiv.org/pdf/2401.06066)
2. [HuggingFace Transformers - DeepSeekV3](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py)
3. [NVIDIA HGX H100 Platform](https://developer.nvidia.com/blog/introducing-nvidia-hgx-h100-an-accelerated-server-platform-for-ai-and-high-performance-computing)
4. [NCCL Developer Guide](https://docs.nvidia.com/deeplearning/nccl)
