#!/bin/bash
# Run all tests for DeepSeekV3 MoE C Implementation

set -e

echo "=== Running DeepSeekV3 MoE Tests ==="
echo ""

if [ ! -f ./moe_test ]; then
    echo "ERROR: ./moe_test not found. Run build.sh first."
    exit 1
fi

if [ ! -d ./test_cases ]; then
    echo "ERROR: test_cases directory not found. Run test generation first."
    echo "  cd test_generation"
    echo "  python generate_mlp_tests.py"
    echo "  python generate_router_tests.py"
    echo "  python generate_experts_tests.py"
    echo "  python generate_moe_full_tests.py"
    exit 1
fi

./moe_test ./test_cases
