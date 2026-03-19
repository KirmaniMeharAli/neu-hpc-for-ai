@echo off
REM Run all tests for DeepSeekV3 MoE C Implementation (Windows)

echo === Running DeepSeekV3 MoE Tests ===
echo.

if not exist moe_test.exe (
    echo ERROR: moe_test.exe not found. Run build.bat first.
    pause
    exit /b 1
)

if not exist test_cases (
    echo ERROR: test_cases directory not found. Run test generation first.
    echo   cd test_generation
    echo   python generate_mlp_tests.py
    echo   python generate_router_tests.py
    echo   python generate_experts_tests.py
    echo   python generate_moe_full_tests.py
    pause
    exit /b 1
)

moe_test.exe test_cases
echo.
pause
