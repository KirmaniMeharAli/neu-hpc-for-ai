# Makefile for Week 8: Multi-GPU DeepSeekV3 MoE with NCCL
#
# Build targets:
#   make          - Build the parallel MoE test runner
#   make test     - Build and run all tests
#   make bench    - Build and run benchmarks only
#   make clean    - Remove build artifacts
#
# Requirements:
#   - CUDA Toolkit (nvcc)
#   - NCCL library
#   - pthreads

NVCC = nvcc
NVCC_FLAGS = -std=c++14 -O2 \
             -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_90,code=sm_90 \
             -Xcompiler -Wall \
             -Xcompiler -pthread

# NCCL paths (adjust if not in default location)
NCCL_LIB = -lnccl
CUDA_LIB = -lcudart

LDFLAGS = $(NCCL_LIB) $(CUDA_LIB) -lpthread -lm

# Source files
SRC_DIR = src
MAIN_SRC = $(SRC_DIR)/main.cu

# Headers that main.cu includes
HEADERS = $(SRC_DIR)/moe_kernels.cuh \
          $(SRC_DIR)/moe_operator.cuh \
          $(SRC_DIR)/moe_parallel.cu \
          $(SRC_DIR)/test_loader.cuh

# Target
TARGET = moe_parallel_test

# Default target
all: $(TARGET)

$(TARGET): $(MAIN_SRC) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(MAIN_SRC) $(LDFLAGS)

# Run all tests
test: $(TARGET)
	./$(TARGET) ./test_cases

# Run with specific GPU count
test2: $(TARGET)
	./$(TARGET) ./test_cases 2

test4: $(TARGET)
	./$(TARGET) ./test_cases 4

# Run benchmarks only (skip to large configs)
bench: $(TARGET)
	./$(TARGET) ./test_cases

# Clean
clean:
	rm -f $(TARGET)

# Rebuild
rebuild: clean all

# Generate test cases (requires Python + PyTorch)
generate_tests:
	cd test_generation && \
	python generate_mlp_tests.py && \
	python generate_router_tests.py && \
	python generate_experts_tests.py && \
	python generate_moe_full_tests.py && \
	python generate_multi_gpu_tests.py

.PHONY: all test test2 test4 bench clean rebuild generate_tests
