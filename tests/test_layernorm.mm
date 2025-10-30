#include <gtest/gtest.h>
#include "backend.h"
#include "metal_backend.h"
#include "cpu_backend.h"
#include "tensor.h"
#include "layernorm.h"
#include <vector>
#include <numeric>
#include <cmath>
#include <memory>

template<typename BackendType>
void TestLayerNormForward(BackendType& backend) {
    int normalized_shape = 4;
    int batch_size = 2;
    float epsilon = 1e-5f;

    LayerNorm layer(&backend, normalized_shape, epsilon);
    
    auto input = backend.create_tensor({batch_size, normalized_shape}, MLLM_FLOAT32);
    input->allocate();

    std::vector<float> gamma_data = {0.5f, 1.5f, 1.0f, 2.0f};
    std::vector<float> beta_data = {0.1f, -0.1f, 0.2f, -0.2f};
    layer.load_weights(gamma_data, beta_data);

    std::vector<float> input_data = {
        1.0f, 2.0f, 3.0f, 4.0f,   // batch 1
        5.0f, 5.0f, 5.0f, 5.0f    // batch 2
    };
    input->copy_from_float(input_data);

    auto output = layer.forward(input.get());

    std::vector<float> result;
    output->copy_to_float(result);

    std::vector<float> expected(input_data.size());

    float mean1 = 2.5f;
    float var1 = 1.25f;
    float inv_std1 = 1.0f / sqrtf(var1 + epsilon);
    expected[0] = (1.0f - mean1) * inv_std1 * gamma_data[0] + beta_data[0];
    expected[1] = (2.0f - mean1) * inv_std1 * gamma_data[1] + beta_data[1];
    expected[2] = (3.0f - mean1) * inv_std1 * gamma_data[2] + beta_data[2];
    expected[3] = (4.0f - mean1) * inv_std1 * gamma_data[3] + beta_data[3];

    expected[4] = beta_data[0];
    expected[5] = beta_data[1];
    expected[6] = beta_data[2];
    expected[7] = beta_data[3];

    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result[i], expected[i], 1e-3);
    }
}

TEST(LayerNormTest, ForwardMetal) {
    MetalBackend backend;
    TestLayerNormForward(backend);
}

TEST(LayerNormTest, ForwardCpu) {
    CpuBackend backend;
    TestLayerNormForward(backend);
}
