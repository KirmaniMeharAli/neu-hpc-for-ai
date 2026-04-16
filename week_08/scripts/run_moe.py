"""
Week 8: Run Multi-GPU DeepSeekV3 MoE on Modal.com

Usage:
    cd C:\\Users\\mehar\\neu-hpc-for-ai\\week_08\\scripts
    modal run run_moe.py

This script:
    1. Uploads all source files and test cases to Modal
    2. Compiles with nvcc + NCCL on a multi-GPU instance
    3. Runs Part A (single-GPU CUDA tests)
    4. Runs Part B (multi-GPU NCCL tests) if 2+ GPUs available
    5. Runs Part C (performance benchmarks)
    6. Prints all results

GPU options:
    - "H100"   : 1x H100 (Part A + Part C only)
    - "H100:2" : 2x H100 (Full test: Part A + B + C)
    - "A100"   : 1x A100 (Part A + Part C only)
    - "A100:2" : 2x A100 (Full test: Part A + B + C)
"""

import modal
import os

app = modal.App("week8-moe-nccl")

# Image with CUDA 12.6 + NCCL
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "wget", "gnupg", "software-properties-common")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-12-6 libnccl-dev libnccl2",
    )
)

# ============================================================
# Single-GPU run (Part A + Part C only)
# Use this if you want to save credits
# ============================================================
@app.function(
    image=image,
    gpu="H100",
    timeout=600,
)
def run_single_gpu(src_files: dict, test_files: dict):
    """Run on single GPU - validates CUDA kernels + benchmarks"""
    import subprocess
    import os

    work_dir = "/root/week_08"
    os.makedirs(f"{work_dir}/src", exist_ok=True)
    os.makedirs(f"{work_dir}/test_cases", exist_ok=True)

    # Write source files
    for name, content in src_files.items():
        with open(f"{work_dir}/src/{name}", "w") as f:
            f.write(content)

    # Write test case files
    for name, content in test_files.items():
        with open(f"{work_dir}/test_cases/{name}", "w") as f:
            f.write(content)

    os.chdir(work_dir)
    os.environ["PATH"] = "/usr/local/cuda-12.6/bin:/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.6/lib64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")

    # Detect GPU
    result = subprocess.run("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader",
                            shell=True, capture_output=True, text=True)
    gpu_info = result.stdout.strip()
    print(f"GPU: {gpu_info}")

    # Determine architecture
    arch = "sm_80"  # default A100
    if "H100" in gpu_info:
        arch = "sm_90"
    elif "A100" in gpu_info:
        arch = "sm_80"
    elif "4090" in gpu_info:
        arch = "sm_89"

    # Compile
    print("\n" + "=" * 60)
    print("Compiling...")
    print("=" * 60)

    build_cmd = (
        f"nvcc -std=c++14 -O2 -arch={arch} "
        "-Xcompiler -Wall -Xcompiler -pthread "
        "-o moe_parallel_test src/main.cu "
        "-lnccl -lcudart -lpthread -lm 2>&1"
    )
    result = subprocess.run(build_cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("BUILD ERRORS:")
        print(result.stderr)
        return
    print("Build successful!")

    # Run tests (single GPU)
    print("\n" + "=" * 60)
    print("Running tests (1 GPU)...")
    print("=" * 60)
    result = subprocess.run("./moe_parallel_test ./test_cases 1 2>&1",
                            shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


# ============================================================
# Multi-GPU run (Part A + Part B + Part C)
# This is the full assignment - uses 2 GPUs
# ============================================================
@app.function(
    image=image,
    gpu="H100:2",  # 2x H100 for multi-GPU NCCL testing
    timeout=900,
)
def run_multi_gpu(src_files: dict, test_files: dict):
    """Run on 2 GPUs - validates NCCL data+expert parallelism"""
    import subprocess
    import os

    work_dir = "/root/week_08"
    os.makedirs(f"{work_dir}/src", exist_ok=True)
    os.makedirs(f"{work_dir}/test_cases", exist_ok=True)

    # Write source files
    for name, content in src_files.items():
        with open(f"{work_dir}/src/{name}", "w") as f:
            f.write(content)

    # Write test case files
    for name, content in test_files.items():
        with open(f"{work_dir}/test_cases/{name}", "w") as f:
            f.write(content)

    os.chdir(work_dir)
    os.environ["PATH"] = "/usr/local/cuda-12.6/bin:/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.6/lib64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")

    # Detect GPUs
    result = subprocess.run("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader",
                            shell=True, capture_output=True, text=True)
    gpu_info = result.stdout.strip()
    print(f"GPUs detected:\n{gpu_info}")

    num_gpus = len(gpu_info.strip().split("\n"))
    print(f"Number of GPUs: {num_gpus}")

    # Determine architecture
    arch = "sm_80"
    if "H100" in gpu_info:
        arch = "sm_90"
    elif "A100" in gpu_info:
        arch = "sm_80"

    # Compile
    print("\n" + "=" * 60)
    print("Compiling...")
    print("=" * 60)

    build_cmd = (
        f"nvcc -std=c++14 -O2 -arch={arch} "
        "-Xcompiler -Wall -Xcompiler -pthread "
        "-o moe_parallel_test src/main.cu "
        "-lnccl -lcudart -lpthread -lm 2>&1"
    )
    result = subprocess.run(build_cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("BUILD ERRORS:")
        print(result.stderr)
        return
    print("Build successful!")

    # Run tests with all available GPUs
    print("\n" + "=" * 60)
    print(f"Running tests ({num_gpus} GPUs)...")
    print("=" * 60)
    result = subprocess.run(f"./moe_parallel_test ./test_cases {num_gpus} 2>&1",
                            shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


def load_project_files():
    """Load all source and test files from local project directory"""
    # Find project root (parent of scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Load source files
    src_dir = os.path.join(project_dir, "src")
    src_files = {}
    for fname in os.listdir(src_dir):
        if fname.endswith((".cu", ".cuh")):
            with open(os.path.join(src_dir, fname), "r") as f:
                src_files[fname] = f.read()
    print(f"Loaded {len(src_files)} source files: {list(src_files.keys())}")

    # Load test case files
    test_dir = os.path.join(project_dir, "test_cases")
    test_files = {}
    for fname in os.listdir(test_dir):
        if fname.endswith(".json"):
            with open(os.path.join(test_dir, fname), "r") as f:
                test_files[fname] = f.read()
    print(f"Loaded {len(test_files)} test cases: {list(test_files.keys())}")

    return src_files, test_files


@app.local_entrypoint()
def main():
    """
    Entry point - runs on your local machine, dispatches to Modal GPU.

    To run single-GPU only (cheaper):
        modal run run_moe.py --mode single

    To run multi-GPU (full assignment):
        modal run run_moe.py
    """
    import sys

    mode = "multi"  # default: full multi-GPU test
    for arg in sys.argv:
        if arg == "--mode" or arg == "single":
            mode = "single"

    print("=" * 60)
    print("Week 8: Multi-GPU DeepSeekV3 MoE with NCCL")
    print("=" * 60)

    src_files, test_files = load_project_files()

    if mode == "single":
        print("\nMode: SINGLE GPU (Part A + Part C)")
        print("GPU: 1x H100")
        print("Estimated cost: ~$0.50")
        print("-" * 60)
        run_single_gpu.remote(src_files, test_files)
    else:
        print("\nMode: MULTI GPU (Part A + Part B + Part C)")
        print("GPU: 2x H100")
        print("Estimated cost: ~$1.50")
        print("-" * 60)
        run_multi_gpu.remote(src_files, test_files)

    print("\nDone!")
