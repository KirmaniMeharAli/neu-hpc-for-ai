#!/bin/bash
# Build script for Week 8: Multi-GPU DeepSeekV3 MoE with NCCL
set -e

echo "============================================"
echo "Building Week 8: Multi-GPU MoE with NCCL"
echo "============================================"

# Check for nvcc
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please install CUDA Toolkit."
    exit 1
fi

# Check CUDA version
echo "CUDA Compiler:"
nvcc --version | tail -1

# Check for NCCL
if ! ldconfig -p 2>/dev/null | grep -q libnccl; then
    echo "WARNING: NCCL library not found in ldconfig. Build may fail."
    echo "  Install with: sudo apt install libnccl-dev"
fi

# Detect GPU architecture
GPU_ARCH="sm_70"  # Default: Volta
if command -v nvidia-smi &> /dev/null; then
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    if [ ! -z "$COMPUTE_CAP" ]; then
        GPU_ARCH="sm_${COMPUTE_CAP}"
        echo "Detected GPU architecture: ${GPU_ARCH}"
    fi
fi

echo ""
echo "Compiling..."
nvcc -std=c++17 -O2 -arch=${GPU_ARCH} \
     -Xcompiler -Wall \
     -Xcompiler -pthread \
     -o moe_parallel_test \
     src/main.cu \
     -lnccl -lcudart -lpthread -lm

echo ""
echo "Build successful: ./moe_parallel_test"
echo ""
echo "Usage:"
echo "  ./moe_parallel_test [test_dir] [num_gpus]"
echo "  ./moe_parallel_test ./test_cases      # auto-detect GPUs"
echo "  ./moe_parallel_test ./test_cases 2    # use 2 GPUs"
