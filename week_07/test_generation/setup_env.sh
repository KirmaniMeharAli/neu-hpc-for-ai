#!/bin/bash
# Setup script for Week 7 test generation environment
# Run from the week_07/test_generation/ directory

set -e

echo "=== Week 7: DeepSeekV3 MoE Test Generation Setup ==="

# Step 1: Clone transformers if not already present
if [ ! -d "transformers" ]; then
    echo "Cloning HuggingFace transformers repository..."
    git clone https://github.com/huggingface/transformers.git
else
    echo "transformers directory already exists, skipping clone."
fi

# Step 2: Create virtual environment
echo "Creating Python virtual environment..."
python -m venv venv

# Step 3: Activate and install
echo "Activating environment and installing dependencies..."
source venv/bin/activate
pip install -r requirements.txt

echo ""
echo "=== Setup Complete ==="
echo "To activate the environment: source venv/bin/activate"
echo "To generate tests: python generate_mlp_tests.py"
