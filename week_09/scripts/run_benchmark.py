"""
Week 9: Run DeepSeekV3 MoE benchmark on Modal.com B200

Usage:
    modal run run_benchmark.py

This script:
    1. Clones ThunderKittens
    2. Compiles the TK-based MoE benchmark
    3. Runs the benchmark on a B200 GPU
    4. Prints performance comparison results
"""

import modal

app = modal.App("week9-moe-tk-benchmark")

# Image with CUDA 12.6 + build tools
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04")
    .apt_install("git", "build-essential", "python3")
    .run_commands(
        "git clone --depth 1 https://github.com/HazyResearch/ThunderKittens.git /opt/thunderkittens"
    )
)

@app.function(
    image=image,
    gpu="B200",  # Request B200 GPU
    timeout=1200,
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root/week_09")],
)
def run_benchmark():
    import subprocess
    import os

    os.chdir("/root/week_09")
    os.environ["THUNDERKITTENS_ROOT"] = "/opt/thunderkittens"

    # Build
    print("=" * 60)
    print("Building MoE benchmark with ThunderKittens...")
    print("=" * 60)
    result = subprocess.run(
        "make GPU_TARGET=B200 TK_ROOT=/opt/thunderkittens 2>&1",
        shell=True, capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("BUILD ERRORS:")
        print(result.stderr)
        return

    # Run benchmark
    print("\n" + "=" * 60)
    print("Running benchmark on B200...")
    print("=" * 60)
    result = subprocess.run(
        "./benchmark_moe 2>&1",
        shell=True, capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


@app.function(
    image=image,
    gpu="H100",  # Fallback to H100 if B200 unavailable
    timeout=1200,
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root/week_09")],
)
def run_benchmark_h100():
    import subprocess
    import os

    os.chdir("/root/week_09")
    os.environ["THUNDERKITTENS_ROOT"] = "/opt/thunderkittens"

    # Build for H100
    print("=" * 60)
    print("Building MoE benchmark for H100...")
    print("=" * 60)
    result = subprocess.run(
        "make GPU_TARGET=H100 TK_ROOT=/opt/thunderkittens 2>&1",
        shell=True, capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("BUILD ERRORS:")
        print(result.stderr)
        return

    # Run
    print("\n" + "=" * 60)
    print("Running benchmark on H100...")
    print("=" * 60)
    result = subprocess.run(
        "./benchmark_moe 2>&1",
        shell=True, capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


@app.local_entrypoint()
def main():
    try:
        print("Attempting B200...")
        run_benchmark.remote()
    except Exception as e:
        print(f"B200 not available ({e}), falling back to H100...")
        run_benchmark_h100.remote()
