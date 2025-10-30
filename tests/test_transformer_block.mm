#include <gtest/gtest.h>
#include "backend.h"
#include "metal_backend.h"
#include "cpu_backend.h"
#include "tensor.h"
#include "transformer_block.h"
#include <vector>
#include <numeric>
#include <cmath>
#include <memory>

template<typename BackendType>
void TestTransformerBlockForwardShapeAndNoCrash(BackendType& backend) {
    int embed_dim = 6;
    int num_heads = 2;
    int seq_len = 4;
    int batch_size = 1;
    float epsilon = 1e-5f;

    ASSERT_EQ(embed_dim % num_heads, 0);

    TransformerBlock block(&backend, embed_dim, num_heads, epsilon);
    
    auto input = backend.create_tensor({batch_size, seq_len, embed_dim}, MLLM_FLOAT32);
    input->allocate();
    std::vector<float> input_data(input->get_size());
    std::iota(input_data.begin(), input_data.end(), 0.0f);
    input->copy_from_float(input_data);

    std::vector<float> dummy_gamma(embed_dim, 1.0f);
    std::vector<float> dummy_beta(embed_dim, 0.0f);
    std::vector<float> dummy_weight(embed_dim * embed_dim, 0.1f);
    std::vector<float> dummy_bias(embed_dim, 0.0f);
    std::vector<float> dummy_fc_in_weight(embed_dim * 4 * embed_dim, 0.1f);
    std::vector<float> dummy_fc_in_bias(4 * embed_dim, 0.0f);
    std::vector<float> dummy_fc_out_weight(4 * embed_dim * embed_dim, 0.1f);
    std::vector<float> dummy_fc_out_bias(embed_dim, 0.0f);

    block.load_weights(
        dummy_gamma, dummy_beta, // ln_1
        dummy_weight, dummy_bias, // attn_q
        dummy_weight, dummy_bias, // attn_k
        dummy_weight, dummy_bias, // attn_v
        dummy_weight, dummy_bias, // attn_o
        dummy_gamma, dummy_beta, // ln_2
        dummy_fc_in_weight, dummy_fc_in_bias, // fc_in
        dummy_fc_out_weight, dummy_fc_out_bias // fc_out
    );

    std::unique_ptr<Tensor> output = nullptr;
    ASSERT_NO_THROW(output = block.forward(input.get()));

    ASSERT_NE(output, nullptr);
    ASSERT_EQ(output->get_shape().size(), 3);
    ASSERT_EQ(output->get_shape()[0], batch_size);
    ASSERT_EQ(output->get_shape()[1], seq_len);
    ASSERT_EQ(output->get_shape()[2], embed_dim);
}

TEST(TransformerBlockTest, ForwardShapeAndNoCrashMetal) {
    MetalBackend backend;
    TestTransformerBlockForwardShapeAndNoCrash(backend);
}

TEST(TransformerBlockTest, ForwardShapeAndNoCrashCpu) {
    CpuBackend backend;
    TestTransformerBlockForwardShapeAndNoCrash(backend);
}