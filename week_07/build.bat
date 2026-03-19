@echo off
REM Build script for DeepSeekV3 MoE C Implementation (Windows)
REM Requires gcc (MinGW) or cl (MSVC)

echo === Building DeepSeekV3 MoE Operator ===

gcc -Wall -Wextra -O2 -std=c99 ^
    src\main.c ^
    src\moe_operator.c ^
    src\matrix_ops.c ^
    src\activations.c ^
    src\test_loader.c ^
    -o moe_test.exe -lm

if %ERRORLEVEL% EQU 0 (
    echo Build successful: moe_test.exe
) else (
    echo Build FAILED!
    exit /b 1
)

echo.
echo To run tests: moe_test.exe test_cases
pause
