#!/bin/bash
# Build script for DeepSeekV3 MoE C Implementation

set -e

echo "=== Building DeepSeekV3 MoE Operator ==="

gcc -Wall -Wextra -O2 -std=c99 \
    src/main.c \
    src/moe_operator.c \
    src/matrix_ops.c \
    src/activations.c \
    src/test_loader.c \
    -o moe_test -lm

echo "Build successful: ./moe_test"
echo ""
echo "To run tests: ./moe_test ./test_cases"
