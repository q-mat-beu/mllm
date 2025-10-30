#include <gtest/gtest.h>
#include "backend.h"
#include "metal_backend.h"
#include "cpu_backend.h"
#include "tensor.h"
#include "attention.h"
#include <vector>
#include <numeric>
#include <cmath>
#include <memory>

// Helper to print a tensor for debugging
void print_tensor(const Tensor& t, const std::string& name) {
    std::vector<float> data;
    t.copy_to_float(data);
    std::cout << "--- " << name << " ---" << std::endl;
    std::cout << "Shape: [ ";
    for (int dim : t.get_shape()) {
        std::cout << dim << " ";
    }
    std::cout << "]" << std::endl;
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

template<typename BackendType>
void TestMultiHeadAttentionForward(BackendType& backend) {
    int embed_dim = 4;
    int num_heads = 2;
    int seq_len = 3;
    int batch_size = 1;

    MultiHeadAttention layer(&backend, embed_dim, num_heads);

    std::vector<float> w_q(embed_dim * embed_dim, 0.0f);
    std::vector<float> w_k(embed_dim * embed_dim, 0.0f);
    std::vector<float> w_v(embed_dim * embed_dim, 0.0f);
    std::vector<float> w_o(embed_dim * embed_dim, 0.0f);
    for(int i = 0; i < embed_dim; ++i) w_q[i*embed_dim+i] = 1.0f;
    for(int i = 0; i < embed_dim; ++i) w_k[i*embed_dim+i] = 1.0f;
    for(int i = 0; i < embed_dim; ++i) w_v[i*embed_dim+i] = 1.0f;
    for(int i = 0; i < embed_dim; ++i) w_o[i*embed_dim+i] = 1.0f;

    std::vector<float> b_q(embed_dim, 0.0f);
    std::vector<float> b_k(embed_dim, 0.0f);
    std::vector<float> b_v(embed_dim, 0.0f);
    std::vector<float> b_o(embed_dim, 0.0f);

    layer.load_weights(w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o);

    auto input = backend.create_tensor({batch_size, seq_len, embed_dim}, MLLM_FLOAT32);
    input->allocate();
    std::vector<float> input_data = {
        1.0f, 0.0f, 0.5f, 0.0f, // seq 1
        0.0f, 1.0f, 0.0f, 0.5f, // seq 2
        0.5f, 0.0f, 1.0f, 0.0f  // seq 3
    };
    input->copy_from_float(input_data);

    auto output = layer.forward(input.get());

    ASSERT_NE(output, nullptr);
    ASSERT_EQ(output->get_shape().size(), 3);
    ASSERT_EQ(output->get_shape()[0], batch_size);
    ASSERT_EQ(output->get_shape()[1], seq_len);
    ASSERT_EQ(output->get_shape()[2], embed_dim);

    std::vector<float> result;
    output->copy_to_float(result);

    ASSERT_EQ(result.size(), input_data.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result[i], input_data[i], 0.5); 
    }
}

TEST(MultiHeadAttentionTest, ForwardMetal) {
    MetalBackend backend;
    TestMultiHeadAttentionForward(backend);
}

TEST(MultiHeadAttentionTest, ForwardCpu) {
    CpuBackend backend;
    TestMultiHeadAttentionForward(backend);
}