# Makefile for Week 9: DeepSeekV3 MoE with ThunderKittens
#
# Requires:
#   - CUDA 12.3+
#   - ThunderKittens (set THUNDERKITTENS_ROOT)
#   - B200 or H100 GPU

# ThunderKittens root (set via env or here)
TK_ROOT ?= $(THUNDERKITTENS_ROOT)

# GPU target: H100 or B200
GPU_TARGET ?= B200

# CUDA compiler
NVCC = nvcc

# Compute capability
ifeq ($(GPU_TARGET),B200)
    ARCH = -arch=sm_100a
else ifeq ($(GPU_TARGET),H100)
    ARCH = -arch=sm_90a
else
    ARCH = -arch=sm_90a
endif

# Flags
NVCC_FLAGS = -std=c++20 -O3 $(ARCH) \
             --expt-relaxed-constexpr \
             --expt-extended-lambda \
             -I$(TK_ROOT)/include \
             -I$(TK_ROOT)/src \
             -Isrc \
             -Ibaseline \
             -lcurand

# Targets
BENCHMARK = benchmark_moe

.PHONY: all clean

all: $(BENCHMARK)

$(BENCHMARK): benchmark/benchmark.cu src/moe_tk_gemm.cuh src/moe_tk_ops.cuh src/moe_tk.cuh baseline/baseline_gemm.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(BENCHMARK)

run: $(BENCHMARK)
	./$(BENCHMARK)
