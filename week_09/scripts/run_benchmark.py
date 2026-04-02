"""
Week 9: Run DeepSeekV3 MoE benchmark on Modal.com

Usage:
    modal run run_benchmark.py
"""

import modal
import os

app = modal.App("week9-moe-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "wget")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-12-6",
    )
)

@app.function(
    image=image,
    gpu="H100",
    timeout=600,
)
def run_benchmark(benchmark_src: str):
    import subprocess
    import os

    os.makedirs("/root/week_09", exist_ok=True)
    with open("/root/week_09/benchmark.cu", "w") as f:
        f.write(benchmark_src)

    os.chdir("/root/week_09")
    os.environ["PATH"] = "/usr/local/cuda-12.6/bin:" + os.environ.get("PATH", "")

    build_cmd = "nvcc -std=c++17 -O3 -arch=sm_90a -lcurand -o benchmark_moe benchmark.cu 2>&1"

    print("=" * 60)
    print("Building...")
    print("=" * 60)
    result = subprocess.run(build_cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("BUILD ERRORS:")
        print(result.stderr)
        return

    print("Build successful!\n")
    print("=" * 60)
    print("Running benchmark...")
    print("=" * 60)
    result = subprocess.run("./benchmark_moe 2>&1", shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


@app.local_entrypoint()
def main():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    bench_file = os.path.join(base_dir, "benchmark", "benchmark.cu")

    with open(bench_file, "r") as f:
        src = f.read()

    print(f"Sending benchmark.cu ({len(src)} bytes) to Modal...")
    run_benchmark.remote(src)
