#!/bin/bash
# Setup Python environment for Week 8 test generation
set -e

echo "Setting up Python environment for Week 8 test generation..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Environment ready! Activate with: source venv/bin/activate"
echo "Then run:"
echo "  python generate_mlp_tests.py"
echo "  python generate_router_tests.py"
echo "  python generate_experts_tests.py"
echo "  python generate_moe_full_tests.py"
echo "  python generate_multi_gpu_tests.py"
echo "  python compare_performance.py"
