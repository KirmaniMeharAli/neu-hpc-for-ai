@echo off
REM Build script for Week 8: Multi-GPU DeepSeekV3 MoE with NCCL
REM Note: NCCL is Linux-only. On Windows, use WSL2 or Linux VM.

echo ============================================
echo Building Week 8: Multi-GPU MoE with NCCL
echo ============================================
echo.
echo NOTE: NCCL is only available on Linux.
echo       On Windows, use WSL2 or a Linux VM.
echo       Alternatively, build on Modal.com or
echo       a cloud GPU instance.
echo.

REM Check for nvcc
where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: nvcc not found. Install CUDA Toolkit.
    exit /b 1
)

echo CUDA Compiler:
nvcc --version

echo.
echo Compiling (single-GPU only, no NCCL)...
nvcc -std=c++17 -O2 ^
     -o moe_parallel_test.exe ^
     src/main.cu ^
     -lcudart -lm

echo.
echo Build successful: moe_parallel_test.exe
echo.
echo Usage:
echo   moe_parallel_test.exe test_cases
