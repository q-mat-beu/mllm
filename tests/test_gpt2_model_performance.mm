#include <gtest/gtest.h>
#include "backend.h"
#include "metal_backend.h"
#include "cpu_backend.h"
#include "tensor.h"
#include "gpt2_model.h"
#include <vector>
#include <numeric>
#include <map>
#include <string>
#include <chrono>
#include <iostream>
#include <memory>

template<typename BackendType>
void TestGPT2ModelForwardPass(BackendType& backend) {
    int vocab_size = 50257;
    int max_seq_len = 1024;
    int embed_dim = 768;
    int num_heads = 12;
    int num_layers = 12;
    float epsilon = 1e-5f;

    GPT2Model model(&backend, vocab_size, max_seq_len, embed_dim, num_heads, num_layers, epsilon);

    std::map<std::string, std::vector<float>> dummy_weights;
    dummy_weights["wte.weight"] = std::vector<float>(vocab_size * embed_dim, 0.1f);
    dummy_weights["wpe.weight"] = std::vector<float>(max_seq_len * embed_dim, 0.1f);

    for (int i = 0; i < num_layers; ++i) {
        std::string prefix = "h." + std::to_string(i) + ".";
        dummy_weights[prefix + "ln_1.weight"] = std::vector<float>(embed_dim, 1.0f);
        dummy_weights[prefix + "ln_1.bias"] = std::vector<float>(embed_dim, 0.0f);
        dummy_weights[prefix + "attn.w_q_data"] = std::vector<float>(embed_dim * embed_dim, 0.1f);
        dummy_weights[prefix + "attn.b_q_data"] = std::vector<float>(embed_dim, 0.0f);
        dummy_weights[prefix + "attn.w_k_data"] = std::vector<float>(embed_dim * embed_dim, 0.1f);
        dummy_weights[prefix + "attn.b_k_data"] = std::vector<float>(embed_dim, 0.0f);
        dummy_weights[prefix + "attn.w_v_data"] = std::vector<float>(embed_dim * embed_dim, 0.1f);
        dummy_weights[prefix + "attn.b_v_data"] = std::vector<float>(embed_dim, 0.0f);
        dummy_weights[prefix + "attn.c_proj.weight"] = std::vector<float>(embed_dim * embed_dim, 0.1f);
        dummy_weights[prefix + "attn.c_proj.bias"] = std::vector<float>(embed_dim, 0.0f);
        dummy_weights[prefix + "ln_2.weight"] = std::vector<float>(embed_dim, 1.0f);
        dummy_weights[prefix + "ln_2.bias"] = std::vector<float>(embed_dim, 0.0f);
        dummy_weights[prefix + "mlp.c_fc.weight"] = std::vector<float>(embed_dim * 4 * embed_dim, 0.1f);
        dummy_weights[prefix + "mlp.c_fc.bias"] = std::vector<float>(4 * embed_dim, 0.0f);
        dummy_weights[prefix + "mlp.c_proj.weight"] = std::vector<float>(4 * embed_dim * embed_dim, 0.1f);
        dummy_weights[prefix + "mlp.c_proj.bias"] = std::vector<float>(embed_dim, 0.0f);
    }

    dummy_weights["ln_f.weight"] = std::vector<float>(embed_dim, 1.0f);
    dummy_weights["ln_f.bias"] = std::vector<float>(embed_dim, 0.0f);
    dummy_weights["lm_head.weight"] = std::vector<float>(embed_dim * vocab_size, 0.1f);
    dummy_weights["lm_head.bias"] = std::vector<float>(vocab_size, 0.0f);

    model.load_weights(dummy_weights);

    int test_batch_size = 1;
    int test_seq_len = 512;
    auto input_token_ids = backend.create_tensor({test_batch_size, test_seq_len}, MLLM_INT32);
    input_token_ids->allocate();
    std::vector<int> input_data(test_batch_size * test_seq_len);
    std::iota(input_data.begin(), input_data.end(), 0);
    input_token_ids->copy_from_int(input_data);

    std::unique_ptr<Tensor> output = nullptr;
    int num_runs = 1;
    double total_time = 0.0;

    // Warm-up run
    output = model.forward(input_token_ids.get());

    for (int i = 0; i < num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        output = model.forward(input_token_ids.get());
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        total_time += diff.count();
    }

    double avg_time = total_time / num_runs;
    std::cout << "Average forward pass time over " << num_runs << " runs: " << avg_time << " s" << std::endl;

    ASSERT_TRUE(true); // Test is for performance, not correctness
}

TEST(GPT2ModelPerformanceTest, ForwardPassMetal) {
    MetalBackend backend;
    TestGPT2ModelForwardPass(backend);
}

TEST(GPT2ModelPerformanceTest, ForwardPassCpu) {
    CpuBackend backend;
    TestGPT2ModelForwardPass(backend);
}