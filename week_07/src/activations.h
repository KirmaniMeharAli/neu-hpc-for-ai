/*
 * activations.h - Activation functions for DeepSeekV3 MoE
 *
 * Week 7: Communication - Pure C Implementation
 */

#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

/*
 * SiLU (Sigmoid Linear Unit) activation: silu(x) = x * sigmoid(x)
 * Also known as the "Swish" activation function.
 * This is the activation used in DeepSeekV3's MLP blocks (hidden_act = "silu").
 */
float silu_scalar(float x);

/*
 * Apply SiLU in-place to an array: arr[i] = silu(arr[i])
 */
void silu_inplace(float *arr, int n);

/*
 * Sigmoid function: sigmoid(x) = 1 / (1 + exp(-x))
 * Used in the routing logic to convert logits to probabilities.
 */
float sigmoid_scalar(float x);

/*
 * Apply sigmoid in-place to an array: arr[i] = sigmoid(arr[i])
 */
void sigmoid_inplace(float *arr, int n);

#endif /* ACTIVATIONS_H */
