/*
 * main.cu - Test Runner for Multi-GPU DeepSeekV3 MoE
 *
 * Week 8: Data and Expert Parallelism
 *
 * Tests:
 *   Part A: Single-GPU CUDA tests (validates CUDA kernels against week 7 JSON test cases)
 *   Part B: Multi-GPU NCCL tests (validates parallel MoE with data + expert parallelism)
 *   Part C: Performance comparison (single-GPU vs multi-GPU scaling)
 *
 * Usage:
 *   ./moe_parallel_test [test_cases_dir] [num_gpus]
 *   Default: test_cases_dir = ./test_cases, num_gpus = auto-detect
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "moe_kernels.cuh"
#include "moe_operator.cuh"
#include "test_loader.cuh"

/* Include the parallel implementation */
#include "moe_parallel.cu"

#define TOLERANCE 1e-4f
/* Slightly higher tolerance for multi-GPU due to float accumulation order differences */
#define MULTI_GPU_TOLERANCE 1e-3f

static int total_tests = 0;
static int passed_tests = 0;


/* ============================================================
 * Part A: Single-GPU CUDA Tests
 * ============================================================
 * Verifies that CUDA kernel implementations match the
 * PyTorch reference from the week 7 JSON test cases.
 */

/* --- Test: DeepseekV3MLP (CUDA) --- */
static void test_mlp_cuda(const char *filepath) {
    char *json = read_file(filepath);
    if (!json) return;

    char *test_name = json_get_string(json, "test_name");
    int hidden_size = json_get_int(json, "hidden_size");
    int intermediate_size = json_get_int(json, "intermediate_size");
    int batch_size = json_get_int(json, "batch_size");
    int seq_len = json_get_int(json, "seq_len");
    int num_tokens = batch_size * seq_len;

    int cnt;
    float *h_gate_w = json_get_float_array(json, "gate_proj_weight", &cnt);
    float *h_up_w   = json_get_float_array(json, "up_proj_weight", &cnt);
    float *h_down_w = json_get_float_array(json, "down_proj_weight", &cnt);
    float *h_input  = json_get_float_array(json, "input", &cnt);
    float *h_expected = json_get_float_array(json, "expected_output", &cnt);

    /* Upload to GPU */
    float *d_gate_w, *d_up_w, *d_down_w, *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_gate_w, intermediate_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up_w, intermediate_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_down_w, hidden_size * intermediate_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, num_tokens * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, num_tokens * hidden_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_gate_w, h_gate_w, intermediate_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up_w, h_up_w, intermediate_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_down_w, h_down_w, hidden_size * intermediate_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, num_tokens * hidden_size * sizeof(float), cudaMemcpyHostToDevice));

    /* Run CUDA MLP */
    gpu_deepseek_mlp_forward(d_input, d_gate_w, d_up_w, d_down_w, d_output,
                              num_tokens, hidden_size, intermediate_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Download and compare */
    float *h_output = (float *)malloc(num_tokens * hidden_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, num_tokens * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));

    char name[256];
    snprintf(name, sizeof(name), "%s (CUDA)", test_name ? test_name : filepath);
    TestResult result = compare_outputs(name, h_expected, h_output,
                                         num_tokens * hidden_size, TOLERANCE);
    print_test_result(&result, TOLERANCE);
    total_tests++;
    if (result.passed) passed_tests++;

    /* Cleanup */
    CUDA_CHECK(cudaFree(d_gate_w)); CUDA_CHECK(cudaFree(d_up_w));
    CUDA_CHECK(cudaFree(d_down_w)); CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_gate_w); free(h_up_w); free(h_down_w);
    free(h_input); free(h_expected); free(h_output);
    if (test_name) free(test_name);
    free(json);
}

/* --- Test: DeepseekV3TopkRouter (CUDA) --- */
static void test_router_cuda(const char *filepath) {
    char *json = read_file(filepath);
    if (!json) return;

    char *test_name = json_get_string(json, "test_name");
    int hidden_size = json_get_int(json, "hidden_size");
    int n_routed_experts = json_get_int(json, "n_routed_experts");
    int batch_size = json_get_int(json, "batch_size");
    int seq_len = json_get_int(json, "seq_len");
    int num_tokens = batch_size * seq_len;

    int cnt;
    float *h_router_w = json_get_float_array(json, "router_weight", &cnt);
    float *h_input    = json_get_float_array(json, "input", &cnt);
    float *h_expected = json_get_float_array(json, "expected_output", &cnt);

    float *d_router_w, *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_router_w, n_routed_experts * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, num_tokens * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, num_tokens * n_routed_experts * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_router_w, h_router_w, n_routed_experts * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, num_tokens * hidden_size * sizeof(float), cudaMemcpyHostToDevice));

    gpu_deepseek_router_forward(d_input, d_router_w, d_output,
                                 num_tokens, hidden_size, n_routed_experts);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *h_output = (float *)malloc(num_tokens * n_routed_experts * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, num_tokens * n_routed_experts * sizeof(float), cudaMemcpyDeviceToHost));

    char name[256];
    snprintf(name, sizeof(name), "%s (CUDA)", test_name ? test_name : filepath);
    TestResult result = compare_outputs(name, h_expected, h_output,
                                         num_tokens * n_routed_experts, TOLERANCE);
    print_test_result(&result, TOLERANCE);
    total_tests++;
    if (result.passed) passed_tests++;

    CUDA_CHECK(cudaFree(d_router_w)); CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_router_w); free(h_input); free(h_expected); free(h_output);
    if (test_name) free(test_name);
    free(json);
}

/* --- Test: DeepseekV3NaiveMoe Experts (CUDA) --- */
static void test_experts_cuda(const char *filepath) {
    char *json = read_file(filepath);
    if (!json) return;

    char *test_name = json_get_string(json, "test_name");
    int num_experts = json_get_int(json, "num_experts");
    int hidden_dim = json_get_int(json, "hidden_dim");
    int intermediate_dim = json_get_int(json, "intermediate_dim");
    int num_tokens = json_get_int(json, "num_tokens");
    int top_k = json_get_int(json, "top_k");

    int cnt;
    float *h_gate_up  = json_get_float_array(json, "gate_up_proj", &cnt);
    float *h_down     = json_get_float_array(json, "down_proj", &cnt);
    float *h_hidden   = json_get_float_array(json, "hidden_states", &cnt);
    int   *h_tk_idx   = json_get_int_array(json, "top_k_index", &cnt);
    float *h_tk_w     = json_get_float_array(json, "top_k_weights", &cnt);
    float *h_expected = json_get_float_array(json, "expected_output", &cnt);

    float *d_gate_up, *d_down, *d_hidden, *d_tk_w, *d_output;
    int *d_tk_idx;
    CUDA_CHECK(cudaMalloc(&d_gate_up, num_experts * 2 * intermediate_dim * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_down, num_experts * hidden_dim * intermediate_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden, num_tokens * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tk_idx, num_tokens * top_k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tk_w, num_tokens * top_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, num_tokens * hidden_dim * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_gate_up, h_gate_up, num_experts * 2 * intermediate_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_down, h_down, num_experts * hidden_dim * intermediate_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hidden, h_hidden, num_tokens * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tk_idx, h_tk_idx, num_tokens * top_k * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tk_w, h_tk_w, num_tokens * top_k * sizeof(float), cudaMemcpyHostToDevice));

    gpu_deepseek_naive_moe_forward(d_hidden, d_tk_idx, d_tk_w,
                                    d_gate_up, d_down, d_output,
                                    num_tokens, hidden_dim, intermediate_dim,
                                    num_experts, top_k);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *h_output = (float *)malloc(num_tokens * hidden_dim * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, num_tokens * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));

    char name[256];
    snprintf(name, sizeof(name), "%s (CUDA)", test_name ? test_name : filepath);
    TestResult result = compare_outputs(name, h_expected, h_output,
                                         num_tokens * hidden_dim, TOLERANCE);
    print_test_result(&result, TOLERANCE);
    total_tests++;
    if (result.passed) passed_tests++;

    CUDA_CHECK(cudaFree(d_gate_up)); CUDA_CHECK(cudaFree(d_down));
    CUDA_CHECK(cudaFree(d_hidden)); CUDA_CHECK(cudaFree(d_tk_idx));
    CUDA_CHECK(cudaFree(d_tk_w)); CUDA_CHECK(cudaFree(d_output));
    free(h_gate_up); free(h_down); free(h_hidden);
    free(h_tk_idx); free(h_tk_w); free(h_expected); free(h_output);
    if (test_name) free(test_name);
    free(json);
}

/* --- Test: Full MoE (CUDA single-GPU) --- */
static void test_moe_full_cuda(const char *filepath) {
    char *json = read_file(filepath);
    if (!json) return;

    char *test_name = json_get_string(json, "test_name");
    int hidden_size = json_get_int(json, "hidden_size");
    int moe_intermediate_size = json_get_int(json, "moe_intermediate_size");
    int n_routed_experts = json_get_int(json, "n_routed_experts");
    int n_shared_experts = json_get_int(json, "n_shared_experts");
    int n_group = json_get_int(json, "n_group");
    int topk_group = json_get_int(json, "topk_group");
    int top_k = json_get_int(json, "top_k");
    int norm_topk_prob = json_get_bool(json, "norm_topk_prob");
    float routed_scaling_factor = json_get_float(json, "routed_scaling_factor");
    int batch_size = json_get_int(json, "batch_size");
    int seq_len = json_get_int(json, "seq_len");
    int num_tokens = batch_size * seq_len;

    int cnt;
    float *h_router_w   = json_get_float_array(json, "router_weight", &cnt);
    float *h_bias       = json_get_float_array(json, "e_score_correction_bias", &cnt);
    float *h_exp_gate   = json_get_float_array(json, "expert_gate_up_proj", &cnt);
    float *h_exp_down   = json_get_float_array(json, "expert_down_proj", &cnt);
    float *h_sh_gate    = json_get_float_array(json, "shared_gate_proj_weight", &cnt);
    float *h_sh_up      = json_get_float_array(json, "shared_up_proj_weight", &cnt);
    float *h_sh_down    = json_get_float_array(json, "shared_down_proj_weight", &cnt);
    float *h_input      = json_get_float_array(json, "input", &cnt);
    float *h_exp_output = json_get_float_array(json, "expected_output", &cnt);
    float *h_exp_logits = json_get_float_array(json, "router_logits", &cnt);

    int shared_inter = moe_intermediate_size * n_shared_experts;

    /* Upload all to GPU */
    float *d_router_w, *d_bias, *d_exp_gate, *d_exp_down;
    float *d_sh_gate, *d_sh_up, *d_sh_down, *d_input, *d_output;
    float *d_out_logits;

    CUDA_CHECK(cudaMalloc(&d_router_w, n_routed_experts * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, n_routed_experts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_exp_gate, n_routed_experts * 2 * moe_intermediate_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_exp_down, n_routed_experts * hidden_size * moe_intermediate_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sh_gate, shared_inter * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sh_up, shared_inter * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sh_down, hidden_size * shared_inter * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, num_tokens * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, num_tokens * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_logits, num_tokens * n_routed_experts * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_router_w, h_router_w, n_routed_experts * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias, n_routed_experts * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_exp_gate, h_exp_gate, n_routed_experts * 2 * moe_intermediate_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_exp_down, h_exp_down, n_routed_experts * hidden_size * moe_intermediate_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sh_gate, h_sh_gate, shared_inter * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sh_up, h_sh_up, shared_inter * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sh_down, h_sh_down, hidden_size * shared_inter * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, num_tokens * hidden_size * sizeof(float), cudaMemcpyHostToDevice));

    gpu_deepseek_moe_forward(
        d_input, d_router_w, d_bias,
        d_exp_gate, d_exp_down,
        d_sh_gate, d_sh_up, d_sh_down,
        d_output, d_out_logits, NULL, NULL,
        num_tokens, hidden_size, moe_intermediate_size,
        n_routed_experts, n_shared_experts,
        n_group, topk_group, top_k,
        norm_topk_prob, routed_scaling_factor);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *h_output = (float *)malloc(num_tokens * hidden_size * sizeof(float));
    float *h_logits_out = (float *)malloc(num_tokens * n_routed_experts * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, num_tokens * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_logits_out, d_out_logits, num_tokens * n_routed_experts * sizeof(float), cudaMemcpyDeviceToHost));

    /* Compare router logits */
    char logits_name[256];
    snprintf(logits_name, sizeof(logits_name), "%s (CUDA router_logits)",
             test_name ? test_name : filepath);
    TestResult lr = compare_outputs(logits_name, h_exp_logits, h_logits_out,
                                     num_tokens * n_routed_experts, TOLERANCE);
    print_test_result(&lr, TOLERANCE);
    total_tests++; if (lr.passed) passed_tests++;

    /* Compare final output */
    char output_name[256];
    snprintf(output_name, sizeof(output_name), "%s (CUDA final_output)",
             test_name ? test_name : filepath);
    TestResult or_ = compare_outputs(output_name, h_exp_output, h_output,
                                      num_tokens * hidden_size, TOLERANCE);
    print_test_result(&or_, TOLERANCE);
    total_tests++; if (or_.passed) passed_tests++;

    /* Cleanup */
    CUDA_CHECK(cudaFree(d_router_w)); CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_exp_gate)); CUDA_CHECK(cudaFree(d_exp_down));
    CUDA_CHECK(cudaFree(d_sh_gate)); CUDA_CHECK(cudaFree(d_sh_up));
    CUDA_CHECK(cudaFree(d_sh_down)); CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output)); CUDA_CHECK(cudaFree(d_out_logits));
    free(h_router_w); free(h_bias); free(h_exp_gate); free(h_exp_down);
    free(h_sh_gate); free(h_sh_up); free(h_sh_down);
    free(h_input); free(h_exp_output); free(h_exp_logits);
    free(h_output); free(h_logits_out);
    if (test_name) free(test_name);
    free(json);
}


/* ============================================================
 * Part B: Multi-GPU NCCL Tests
 * ============================================================
 * Validates the parallel MoE by comparing its output against
 * the single-GPU reference (which itself is validated in Part A).
 */

/* Thread argument for pthreads-based multi-GPU execution */
typedef struct {
    GPUState *state;
} ParallelThreadArg;

/* Thread function: each thread runs parallel_moe_forward on its GPU */
static void *parallel_thread_fn(void *arg) {
    ParallelThreadArg *ta = (ParallelThreadArg *)arg;
    parallel_moe_forward(ta->state);
    return NULL;
}

static void test_moe_parallel(const char *filepath, int num_gpus) {
    char *json = read_file(filepath);
    if (!json) return;

    char *test_name = json_get_string(json, "test_name");
    int hidden_size = json_get_int(json, "hidden_size");
    int moe_intermediate_size = json_get_int(json, "moe_intermediate_size");
    int n_routed_experts = json_get_int(json, "n_routed_experts");
    int n_shared_experts = json_get_int(json, "n_shared_experts");
    int n_group = json_get_int(json, "n_group");
    int topk_group = json_get_int(json, "topk_group");
    int top_k = json_get_int(json, "top_k");
    int norm_topk_prob = json_get_bool(json, "norm_topk_prob");
    float routed_scaling_factor = json_get_float(json, "routed_scaling_factor");
    int batch_size = json_get_int(json, "batch_size");
    int seq_len = json_get_int(json, "seq_len");
    int num_tokens = batch_size * seq_len;

    int cnt;
    float *h_router_w   = json_get_float_array(json, "router_weight", &cnt);
    float *h_bias       = json_get_float_array(json, "e_score_correction_bias", &cnt);
    float *h_exp_gate   = json_get_float_array(json, "expert_gate_up_proj", &cnt);
    float *h_exp_down   = json_get_float_array(json, "expert_down_proj", &cnt);
    float *h_sh_gate    = json_get_float_array(json, "shared_gate_proj_weight", &cnt);
    float *h_sh_up      = json_get_float_array(json, "shared_up_proj_weight", &cnt);
    float *h_sh_down    = json_get_float_array(json, "shared_down_proj_weight", &cnt);
    float *h_input      = json_get_float_array(json, "input", &cnt);
    float *h_exp_output = json_get_float_array(json, "expected_output", &cnt);

    /* Check if we have enough experts and tokens for num_gpus */
    if (n_routed_experts < num_gpus) {
        printf("  %-40s  SKIP  (n_experts=%d < num_gpus=%d)\n",
               test_name ? test_name : filepath, n_routed_experts, num_gpus);
        free(h_router_w); free(h_bias); free(h_exp_gate); free(h_exp_down);
        free(h_sh_gate); free(h_sh_up); free(h_sh_down);
        free(h_input); free(h_exp_output);
        if (test_name) free(test_name);
        free(json);
        return;
    }
    if (num_tokens < num_gpus) {
        printf("  %-40s  SKIP  (n_tokens=%d < num_gpus=%d)\n",
               test_name ? test_name : filepath, num_tokens, num_gpus);
        free(h_router_w); free(h_bias); free(h_exp_gate); free(h_exp_down);
        free(h_sh_gate); free(h_sh_up); free(h_sh_down);
        free(h_input); free(h_exp_output);
        if (test_name) free(test_name);
        free(json);
        return;
    }

    /* Initialize NCCL */
    ncclComm_t *comms = (ncclComm_t *)malloc(num_gpus * sizeof(ncclComm_t));
    int *dev_list = (int *)malloc(num_gpus * sizeof(int));
    for (int i = 0; i < num_gpus; i++) dev_list[i] = i;

    ncclUniqueId nccl_id;
    NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    NCCL_CHECK(ncclCommInitAll(comms, num_gpus, dev_list));

    MoEConfig config;
    config.hidden_size = hidden_size;
    config.moe_intermediate_size = moe_intermediate_size;
    config.n_routed_experts = n_routed_experts;
    config.n_shared_experts = n_shared_experts;
    config.n_group = n_group;
    config.topk_group = topk_group;
    config.top_k = top_k;
    config.norm_topk_prob = norm_topk_prob;
    config.routed_scaling_factor = routed_scaling_factor;

    /* Initialize all GPU states */
    GPUState *states = (GPUState *)malloc(num_gpus * sizeof(GPUState));
    for (int g = 0; g < num_gpus; g++) {
        init_gpu_state(&states[g], g, num_gpus, comms[g], &config, num_tokens);
        upload_weights_to_gpu(&states[g],
                               h_router_w, h_bias,
                               h_exp_gate, h_exp_down,
                               h_sh_gate, h_sh_up, h_sh_down);
        upload_local_input(&states[g], h_input, num_tokens);
    }

    /* Run parallel MoE forward on each GPU sequentially
     * (In production this would be multi-threaded; here we run
     *  sequentially since NCCL handles the synchronization) */
    /* NOTE: For NCCL all-to-all to work correctly, all ranks must
     * call the communication collectively. We use a simple
     * sequential approach where each GPU's forward pass includes
     * its NCCL calls, which works because ncclCommInitAll creates
     * intra-process communicators that handle this correctly. */

    /* Actually, for proper NCCL operation, all ranks must enter
     * the group calls together. We'll use pthreads for this. */
    pthread_t *threads = (pthread_t *)malloc(num_gpus * sizeof(pthread_t));
    ParallelThreadArg *args = (ParallelThreadArg *)malloc(num_gpus * sizeof(ParallelThreadArg));

    for (int g = 0; g < num_gpus; g++) {
        args[g].state = &states[g];
    }

    for (int g = 0; g < num_gpus; g++) {
        pthread_create(&threads[g], NULL, parallel_thread_fn, &args[g]);
    }
    for (int g = 0; g < num_gpus; g++) {
        pthread_join(threads[g], NULL);
    }

    /* Gather outputs from all GPUs */
    float *h_parallel_output = (float *)calloc(num_tokens * hidden_size, sizeof(float));
    for (int g = 0; g < num_gpus; g++) {
        download_local_output(&states[g], h_parallel_output, num_tokens);
    }

    /* Compare with expected (PyTorch reference) */
    char par_name[256];
    snprintf(par_name, sizeof(par_name), "%s (NCCL %d-GPU)",
             test_name ? test_name : filepath, num_gpus);
    TestResult pr = compare_outputs(par_name, h_exp_output, h_parallel_output,
                                     num_tokens * hidden_size, MULTI_GPU_TOLERANCE);
    print_test_result(&pr, MULTI_GPU_TOLERANCE);
    total_tests++;
    if (pr.passed) passed_tests++;

    /* Cleanup */
    for (int g = 0; g < num_gpus; g++) {
        cleanup_gpu_state(&states[g]);
        NCCL_CHECK(ncclCommDestroy(comms[g]));
    }
    free(states); free(comms); free(dev_list);
    free(threads); free(args);
    free(h_parallel_output);
    free(h_router_w); free(h_bias); free(h_exp_gate); free(h_exp_down);
    free(h_sh_gate); free(h_sh_up); free(h_sh_down);
    free(h_input); free(h_exp_output);
    if (test_name) free(test_name);
    free(json);
}


/* ============================================================
 * Part C: Performance Benchmark
 * ============================================================
 * Measures throughput for single-GPU vs multi-GPU with
 * increasing token counts and expert counts.
 */
static void benchmark_moe(int num_gpus) {
    printf("\n");
    printf("============================================================\n");
    printf("Part C: Performance Benchmark\n");
    printf("============================================================\n");
    printf("%-12s %-12s %-12s %-12s %-14s %-14s\n",
           "Tokens", "Experts", "Hidden", "Top-K",
           "1-GPU (ms)", num_gpus > 1 ? "N-GPU (ms)" : "N/A");
    printf("------------------------------------------------------------"
           "--------------------\n");

    /* Benchmark configs: increasing scale */
    struct {
        int num_tokens, n_experts, hidden_size, inter_size, top_k;
    } bench_configs[] = {
        {   4,   8,   32,   64, 2},
        {  16,  16,   64,  128, 2},
        {  64,  32,  128,  256, 4},
        { 256,  64,  256,  512, 4},
        { 512, 128,  256,  512, 8},
    };
    int num_configs = sizeof(bench_configs) / sizeof(bench_configs[0]);

    for (int c = 0; c < num_configs; c++) {
        int nt = bench_configs[c].num_tokens;
        int ne = bench_configs[c].n_experts;
        int hs = bench_configs[c].hidden_size;
        int mi = bench_configs[c].inter_size;
        int tk = bench_configs[c].top_k;
        int ng = (ne >= 4) ? ne / 4 : 1;  /* n_group */
        int tg = (ng >= 2) ? ng / 2 : 1;  /* topk_group */

        /* Generate random weights and inputs */
        srand(42);
        int router_w_size = ne * hs;
        int exp_gate_size = ne * 2 * mi * hs;
        int exp_down_size = ne * hs * mi;
        int shared_inter = mi;
        int sh_gate_size = shared_inter * hs;
        int sh_down_size = hs * shared_inter;

        float *h_router_w = (float *)malloc(router_w_size * sizeof(float));
        float *h_bias = (float *)calloc(ne, sizeof(float));
        float *h_exp_gate = (float *)malloc(exp_gate_size * sizeof(float));
        float *h_exp_down = (float *)malloc(exp_down_size * sizeof(float));
        float *h_sh_gate = (float *)malloc(sh_gate_size * sizeof(float));
        float *h_sh_up = (float *)malloc(sh_gate_size * sizeof(float));
        float *h_sh_down = (float *)malloc(sh_down_size * sizeof(float));
        float *h_input = (float *)malloc(nt * hs * sizeof(float));

        for (int i = 0; i < router_w_size; i++) h_router_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.04f;
        for (int i = 0; i < exp_gate_size; i++) h_exp_gate[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.04f;
        for (int i = 0; i < exp_down_size; i++) h_exp_down[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.04f;
        for (int i = 0; i < sh_gate_size; i++) h_sh_gate[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.04f;
        for (int i = 0; i < sh_gate_size; i++) h_sh_up[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.04f;
        for (int i = 0; i < sh_down_size; i++) h_sh_down[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.04f;
        for (int i = 0; i < nt * hs; i++) h_input[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

        /* ---- Single-GPU benchmark ---- */
        CUDA_CHECK(cudaSetDevice(0));
        float *d_rw, *d_b, *d_eg, *d_ed, *d_sg, *d_su, *d_sd, *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_rw, router_w_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b, ne * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_eg, exp_gate_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ed, exp_down_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sg, sh_gate_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_su, sh_gate_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sd, sh_down_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_in, nt * hs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, nt * hs * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_rw, h_router_w, router_w_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_bias, ne * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_eg, h_exp_gate, exp_gate_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ed, h_exp_down, exp_down_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sg, h_sh_gate, sh_gate_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_su, h_sh_up, sh_gate_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sd, h_sh_down, sh_down_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_in, h_input, nt * hs * sizeof(float), cudaMemcpyHostToDevice));

        /* Warmup */
        gpu_deepseek_moe_forward(d_in, d_rw, d_b, d_eg, d_ed, d_sg, d_su, d_sd,
                                  d_out, NULL, NULL, NULL,
                                  nt, hs, mi, ne, 1, ng, tg, tk, 1, 2.0f);
        CUDA_CHECK(cudaDeviceSynchronize());

        /* Timed run */
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        int num_iters = 5;
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < num_iters; i++) {
            gpu_deepseek_moe_forward(d_in, d_rw, d_b, d_eg, d_ed, d_sg, d_su, d_sd,
                                      d_out, NULL, NULL, NULL,
                                      nt, hs, mi, ne, 1, ng, tg, tk, 1, 2.0f);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float single_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&single_ms, start, stop));
        single_ms /= num_iters;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_rw)); CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_eg)); CUDA_CHECK(cudaFree(d_ed));
        CUDA_CHECK(cudaFree(d_sg)); CUDA_CHECK(cudaFree(d_su));
        CUDA_CHECK(cudaFree(d_sd)); CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));

        /* ---- Multi-GPU benchmark ---- */
        float multi_ms = -1.0f;
        if (num_gpus > 1 && ne >= num_gpus && nt >= num_gpus) {
            ncclComm_t *comms = (ncclComm_t *)malloc(num_gpus * sizeof(ncclComm_t));
            int *devs = (int *)malloc(num_gpus * sizeof(int));
            for (int i = 0; i < num_gpus; i++) devs[i] = i;
            NCCL_CHECK(ncclCommInitAll(comms, num_gpus, devs));

            MoEConfig cfg;
            cfg.hidden_size = hs;
            cfg.moe_intermediate_size = mi;
            cfg.n_routed_experts = ne;
            cfg.n_shared_experts = 1;
            cfg.n_group = ng;
            cfg.topk_group = tg;
            cfg.top_k = tk;
            cfg.norm_topk_prob = 1;
            cfg.routed_scaling_factor = 2.0f;

            GPUState *st = (GPUState *)malloc(num_gpus * sizeof(GPUState));
            for (int g = 0; g < num_gpus; g++) {
                init_gpu_state(&st[g], g, num_gpus, comms[g], &cfg, nt);
                upload_weights_to_gpu(&st[g], h_router_w, h_bias,
                                       h_exp_gate, h_exp_down,
                                       h_sh_gate, h_sh_up, h_sh_down);
                upload_local_input(&st[g], h_input, nt);
            }

            /* Warmup */
            pthread_t *thr = (pthread_t *)malloc(num_gpus * sizeof(pthread_t));
            ParallelThreadArg *tas = (ParallelThreadArg *)malloc(num_gpus * sizeof(ParallelThreadArg));
            for (int g = 0; g < num_gpus; g++) tas[g].state = &st[g];

            for (int g = 0; g < num_gpus; g++) pthread_create(&thr[g], NULL, parallel_thread_fn, &tas[g]);
            for (int g = 0; g < num_gpus; g++) pthread_join(thr[g], NULL);

            /* Timed run */
            struct timespec ts_start, ts_end;
            clock_gettime(CLOCK_MONOTONIC, &ts_start);
            for (int i = 0; i < num_iters; i++) {
                for (int g = 0; g < num_gpus; g++) {
                    upload_local_input(&st[g], h_input, nt);
                }
                for (int g = 0; g < num_gpus; g++) pthread_create(&thr[g], NULL, parallel_thread_fn, &tas[g]);
                for (int g = 0; g < num_gpus; g++) pthread_join(thr[g], NULL);
            }
            clock_gettime(CLOCK_MONOTONIC, &ts_end);

            multi_ms = (float)((ts_end.tv_sec - ts_start.tv_sec) * 1000.0 +
                                (ts_end.tv_nsec - ts_start.tv_nsec) / 1e6) / num_iters;

            for (int g = 0; g < num_gpus; g++) {
                cleanup_gpu_state(&st[g]);
                NCCL_CHECK(ncclCommDestroy(comms[g]));
            }
            free(st); free(comms); free(devs); free(thr); free(tas);
        }

        printf("%-12d %-12d %-12d %-12d %-14.3f",
               nt, ne, hs, tk, single_ms);
        if (multi_ms > 0) {
            printf(" %-14.3f\n", multi_ms);
        } else {
            printf(" %-14s\n", "N/A");
        }

        free(h_router_w); free(h_bias); free(h_exp_gate); free(h_exp_down);
        free(h_sh_gate); free(h_sh_up); free(h_sh_down); free(h_input);
    }
}


/* ============================================================
 * Main
 * ============================================================ */
int main(int argc, char *argv[]) {
    const char *test_dir = (argc > 1) ? argv[1] : "./test_cases";

    /* Detect GPUs */
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    int num_gpus = (argc > 2) ? atoi(argv[2]) : device_count;
    if (num_gpus > device_count) num_gpus = device_count;
    if (num_gpus < 1) num_gpus = 1;

    /* Print GPU info */
    printf("============================================================\n");
    printf("Week 8: Multi-GPU DeepSeekV3 MoE with NCCL\n");
    printf("Data Parallelism + Expert Parallelism\n");
    printf("============================================================\n");
    printf("Available GPUs: %d\n", device_count);
    printf("Using GPUs:     %d\n", num_gpus);
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("  GPU %d: %s (%.1f GB, SM %d.%d)\n",
               i, prop.name,
               (float)prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f),
               prop.major, prop.minor);
    }
    printf("Test directory: %s\n", test_dir);
    printf("Tolerance:      %.0e (single-GPU), %.0e (multi-GPU)\n",
           TOLERANCE, MULTI_GPU_TOLERANCE);
    printf("============================================================\n");

    char path[512];

    /* ============================================================
     * Part A: Single-GPU CUDA Tests
     * ============================================================ */
    printf("\n");
    printf("============================================================\n");
    printf("Part A: Single-GPU CUDA Tests\n");
    printf("============================================================\n");

    printf("\n--- Block 1: DeepseekV3MLP (CUDA) ---\n");
    for (int i = 1; i <= 3; i++) {
        snprintf(path, sizeof(path), "%s/mlp_test_%d.json", test_dir, i);
        test_mlp_cuda(path);
    }

    printf("\n--- Block 2: DeepseekV3TopkRouter (CUDA) ---\n");
    for (int i = 1; i <= 3; i++) {
        snprintf(path, sizeof(path), "%s/router_test_%d.json", test_dir, i);
        test_router_cuda(path);
    }

    printf("\n--- Block 3: DeepseekV3NaiveMoe (CUDA) ---\n");
    for (int i = 1; i <= 3; i++) {
        snprintf(path, sizeof(path), "%s/experts_test_%d.json", test_dir, i);
        test_experts_cuda(path);
    }

    printf("\n--- Block 4: DeepseekV3MoE Full (CUDA) ---\n");
    for (int i = 1; i <= 3; i++) {
        snprintf(path, sizeof(path), "%s/moe_full_test_%d.json", test_dir, i);
        test_moe_full_cuda(path);
    }

    /* ============================================================
     * Part B: Multi-GPU NCCL Tests
     * ============================================================ */
    if (num_gpus > 1) {
        printf("\n");
        printf("============================================================\n");
        printf("Part B: Multi-GPU NCCL Tests (%d GPUs)\n", num_gpus);
        printf("============================================================\n");

        printf("\n--- Full MoE with Data + Expert Parallelism ---\n");
        for (int i = 1; i <= 3; i++) {
            snprintf(path, sizeof(path), "%s/moe_full_test_%d.json", test_dir, i);
            test_moe_parallel(path, num_gpus);
        }

        printf("\n--- Multi-GPU Specific Tests (larger dimensions) ---\n");
        for (int i = 1; i <= 3; i++) {
            snprintf(path, sizeof(path), "%s/moe_multi_gpu_test_%d.json", test_dir, i);
            test_moe_parallel(path, num_gpus);
        }
    } else {
        printf("\n");
        printf("============================================================\n");
        printf("Part B: SKIPPED (only 1 GPU available, need >= 2 for NCCL)\n");
        printf("============================================================\n");
    }

    /* ============================================================
     * Part C: Performance Benchmark
     * ============================================================ */
    benchmark_moe(num_gpus);

    /* ============================================================
     * Summary
     * ============================================================ */
    printf("\n============================================================\n");
    printf("RESULTS: %d / %d tests passed\n", passed_tests, total_tests);
    printf("============================================================\n");

    return (passed_tests == total_tests) ? 0 : 1;
}
