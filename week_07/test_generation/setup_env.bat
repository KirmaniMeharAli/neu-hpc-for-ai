@echo off
REM Setup script for Week 7 test generation environment
REM Run from the week_07/test_generation/ directory

echo === Week 7: DeepSeekV3 MoE Test Generation Setup ===

REM Step 1: Clone transformers if not already present
if not exist "transformers" (
    echo Cloning HuggingFace transformers repository...
    git clone https://github.com/huggingface/transformers.git
) else (
    echo transformers directory already exists, skipping clone.
)

REM Step 2: Create virtual environment
echo Creating Python virtual environment...
python -m venv venv

REM Step 3: Activate and install
echo Activating environment and installing dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo === Setup Complete ===
echo To activate the environment: venv\Scripts\activate.bat
echo To generate tests: python generate_mlp_tests.py
echo.
pause
