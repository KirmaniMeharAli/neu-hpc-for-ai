/*
 * activations.c - Activation functions for DeepSeekV3 MoE
 *
 * Week 7: Communication - Pure C Implementation
 */

#include "activations.h"
#include <math.h>

/*
 * SiLU (Sigmoid Linear Unit): silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * DeepSeekV3 uses SiLU as its activation function in MLP blocks.
 * This matches PyTorch's F.silu() / torch.nn.SiLU().
 */
float silu_scalar(float x) {
    return x / (1.0f + expf(-x));
}

void silu_inplace(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = silu_scalar(arr[i]);
    }
}

/*
 * Sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
 *
 * Used in the router's route_tokens_to_experts() to convert
 * raw router logits to probabilities before expert selection.
 * Note: DeepSeekV3 uses sigmoid (NOT softmax) for routing.
 */
float sigmoid_scalar(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void sigmoid_inplace(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = sigmoid_scalar(arr[i]);
    }
}
