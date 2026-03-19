# Test Case Generation for DeepSeekV3 MoE Operator

## Overview

This directory contains Python scripts to generate test cases for each block
in the DeepSeekV3 Mixture-of-Experts (MoE) operator. The scripts use the
HuggingFace `transformers` library as the reference implementation.

## MoE Blocks Tested

| Block | Source Class | Script | Description |
|-------|-------------|--------|-------------|
| MLP | `DeepseekV3MLP` | `generate_mlp_tests.py` | Gated MLP: `down(silu(gate(x)) * up(x))` |
| Router | `DeepseekV3TopkRouter` | `generate_router_tests.py` | Linear projection to router logits |
| Experts | `DeepseekV3NaiveMoe` | `generate_experts_tests.py` | Expert dispatch with weighted accumulation |
| Full MoE | `DeepseekV3MoE` | `generate_moe_full_tests.py` | Complete MoE: router + routing + experts + shared |

## Setup

```bash
# Clone transformers
git clone https://github.com/huggingface/transformers.git

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

Or use the provided setup script:
```bash
bash setup_env.sh    # Linux/Mac
setup_env.bat        # Windows
```

## Generate Test Cases

```bash
python generate_mlp_tests.py
python generate_router_tests.py
python generate_experts_tests.py
python generate_moe_full_tests.py
```

## Verify Determinism

```bash
python verify_determinism.py
```

This runs each generator twice and verifies outputs are identical.

## Test Case Format

All test cases are saved as JSON in `../test_cases/`. Each file contains:
- Model configuration parameters (dimensions, seeds)
- Weight values (as nested lists)
- Input values (as nested lists)
- Expected output values (as nested lists)

The C implementation loads these JSON files and compares its outputs
against the expected values with a tolerance of 1e-4.

## Design Decisions

1. **Standalone blocks**: Each block is copied out of transformers and made
   self-contained to run in isolation without dependencies.

2. **Small dimensions**: Actual DeepSeekV3 uses hidden_size=7168 with 256
   experts. We shrink to hidden_size=4-8 to keep test cases manageable.

3. **Separate seeds**: Weight generation uses different seeds than input
   generation, so changes to weight config don't affect inputs.

4. **Deterministic mode**: `torch.use_deterministic_algorithms(True)` is
   enabled to ensure reproducibility across runs.
