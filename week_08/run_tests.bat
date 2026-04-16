@echo off
REM Run tests for Week 8
echo Building...
call build.bat
if %errorlevel% neq 0 exit /b 1
echo.
echo Running tests...
moe_parallel_test.exe test_cases %1
