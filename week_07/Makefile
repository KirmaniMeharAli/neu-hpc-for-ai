# Makefile for DeepSeekV3 MoE Operator - Week 7
#
# Pure C implementation, no CUDA, no parallelism.

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm

# Source files
SRC_DIR = src
SRCS = $(SRC_DIR)/main.c \
       $(SRC_DIR)/moe_operator.c \
       $(SRC_DIR)/matrix_ops.c \
       $(SRC_DIR)/activations.c \
       $(SRC_DIR)/test_loader.c

# Object files
OBJS = $(SRCS:.c=.o)

# Target
TARGET = moe_test

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Pattern rule for object files
$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Clean build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

# Run tests (assumes test_cases/ directory exists with JSON files)
test: $(TARGET)
	./$(TARGET) ./test_cases

# Rebuild from scratch
rebuild: clean all

.PHONY: all clean test rebuild
