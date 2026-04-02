"""
Week 9: Run DeepSeekV3 MoE benchmark on Modal.com
Tries B200 (CUDA 12.8) first, falls back to H100 (CUDA 12.6).

Usage:
    modal run run_benchmark.py
"""

import modal
import os

app = modal.App("week9-moe-benchmark")

# H100 image with CUDA 12.6
image_h100 = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "wget")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-12-6",
    )
)

# B200 image with CUDA 12.8 (needed for sm_100a)
image_b200 = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "wget")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-12-8",
    )
)

def _run(benchmark_src, arch, gpu_name, cuda_ver):
    import subprocess, os
    os.makedirs("/root/week_09", exist_ok=True)
    with open("/root/week_09/benchmark.cu", "w") as f:
        f.write(benchmark_src)
    os.chdir("/root/week_09")
    os.environ["PATH"] = f"/usr/local/cuda-{cuda_ver}/bin:" + os.environ.get("PATH", "")

    cmd = f"nvcc -std=c++17 -O3 -arch={arch} -lcurand -o benchmark_moe benchmark.cu 2>&1"
    print(f"Building for {gpu_name} ({arch}, CUDA {cuda_ver})...")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print("BUILD ERRORS:")
        print(r.stderr)
        return

    print(f"\nRunning benchmark on {gpu_name}...")
    print("=" * 60)
    r = subprocess.run("./benchmark_moe 2>&1", shell=True, capture_output=True, text=True)
    print(r.stdout)
    if r.stderr:
        print(r.stderr)

@app.function(image=image_b200, gpu="B200", timeout=600)
def run_b200(src: str):
    _run(src, "sm_100a", "B200", "12.8")

@app.function(image=image_h100, gpu="H100", timeout=600)
def run_h100(src: str):
    _run(src, "sm_90a", "H100", "12.6")

@app.local_entrypoint()
def main():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    with open(os.path.join(base, "benchmark", "benchmark.cu"), "r") as f:
        src = f.read()
    print(f"Sending benchmark.cu ({len(src)} bytes)")

    try:
        print("Trying B200 (CUDA 12.8)...")
        run_b200.remote(src)
    except Exception as e:
        print(f"B200 failed ({e}), falling back to H100...")
        run_h100.remote(src)
