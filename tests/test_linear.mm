#include <gtest/gtest.h>
#include "backend.h"
#include "metal_backend.h"
#include "cpu_backend.h"
#include "tensor.h"
#include "linear.h"
#include <vector>
#include <numeric>
#include <memory>

template<typename BackendType>
void TestLinearLayerForward(BackendType& backend) {
    int in_features = 3;
    int out_features = 2;
    int batch_size = 2;

    Linear layer(&backend, in_features, out_features);
    
    auto input = backend.create_tensor({batch_size, in_features}, MLLM_FLOAT32);
    input->allocate();

    std::vector<float> weight_data = {
        1.0f, 4.0f, 
        2.0f, 5.0f, 
        3.0f, 6.0f
    }; 
    
    std::vector<float> bias_data = {0.5f, -0.5f};
    layer.load_weights(weight_data, bias_data);

    std::vector<float> input_data = {
        10.0f, 20.0f, 30.0f, // batch 1
        -1.0f, -2.0f, -3.0f  // batch 2
    };
    input->copy_from_float(input_data);

    auto output = layer.forward(input.get());

    std::vector<float> result;
    output->copy_to_float(result);

    std::vector<float> expected = {140.5f, 319.5f, -13.5f, -32.5f};

    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }
}

TEST(LinearLayerTest, ForwardMetal) {
    MetalBackend backend;
    TestLinearLayerForward(backend);
}

TEST(LinearLayerTest, ForwardCpu) {
    CpuBackend backend;
    TestLinearLayerForward(backend);
}
