#!/bin/bash
# Run all tests for Week 8: Multi-GPU DeepSeekV3 MoE
set -e

# Build first
echo "Building..."
bash build.sh
echo ""

# Detect GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "Detected ${NUM_GPUS} GPU(s)"
echo ""

# Run tests
if [ "$1" != "" ]; then
    NUM_GPUS=$1
fi

echo "Running tests with ${NUM_GPUS} GPU(s)..."
echo ""
./moe_parallel_test ./test_cases ${NUM_GPUS}
