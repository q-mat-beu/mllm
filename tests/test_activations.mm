#include <gtest/gtest.h>
#include "backend.h"
#include "metal_backend.h" // For MetalBackend specific tests
#include "cpu_backend.h"   // For CpuBackend specific tests
#include "tensor.h"
#include <vector>
#include <cmath>
#include <memory>

// Helper function for GELU calculation on CPU for verification (using the approximation from the Metal shader)
float gelu_cpu(float x) {
    float sigmoid_arg = 1.702f * x;
    float sigmoid_val = 1.0f / (1.0f + expf(-sigmoid_arg)); // Use expf for float
    return x * sigmoid_val;
}

template<typename BackendType>
void TestGelu(BackendType& backend) {
    int size = 6;
    auto in = backend.create_tensor({size}, MLLM_FLOAT32);
    auto out = backend.create_tensor({size}, MLLM_FLOAT32);
    in->allocate();
    out->allocate();

    std::vector<float> data = {-3.0f, -1.0f, -0.5f, 0.0f, 1.0f, 3.0f};
    in->copy_from_float(data);

    backend.gelu(in.get(), out.get());

    std::vector<float> result;
    out->copy_to_float(result);

    std::vector<float> expected;
    for (float val : data) {
        expected.push_back(gelu_cpu(val));
    }

    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result[i], expected[i], 1e-6);
    }
}

TEST(ActivationTest, GeluMetal) {
    MetalBackend backend;
    TestGelu(backend);
}

TEST(ActivationTest, GeluCpu) {
    CpuBackend backend;
    TestGelu(backend);
}

template<typename BackendType>
void TestGeluLargeValues(BackendType& backend) {
    int size = 10;
    auto in = backend.create_tensor({size}, MLLM_FLOAT32);
    auto out = backend.create_tensor({size}, MLLM_FLOAT32);
    in->allocate();
    out->allocate();

    std::vector<float> data(size);
    for (int i = 0; i < size; ++i) {
        data[i] = 1000.0f; // Large positive value
    }
    in->copy_from_float(data);

    backend.gelu(in.get(), out.get());

    std::vector<float> result;
    out->copy_to_float(result);

    for (size_t i = 0; i < result.size(); ++i) {
        ASSERT_FALSE(std::isnan(result[i])) << "NaN detected in GELU output with large input at index " << i;
        ASSERT_FALSE(std::isinf(result[i])) << "Inf detected in GELU output with large input at index " << i;
    }

    // Also test with large negative values
    for (int i = 0; i < size; ++i) {
        data[i] = -1000.0f; // Large negative value
    }
    in->copy_from_float(data);
    backend.gelu(in.get(), out.get());
    out->copy_to_float(result);

    for (size_t i = 0; i < result.size(); ++i) {
        ASSERT_FALSE(std::isnan(result[i])) << "NaN detected in GELU output with large negative input at index " << i;
        ASSERT_FALSE(std::isinf(result[i])) << "Inf detected in GELU output with large negative input at index " << i;
    }
}

TEST(ActivationTest, GeluLargeValuesMetal) {
    MetalBackend backend;
    TestGeluLargeValues(backend);
}

TEST(ActivationTest, GeluLargeValuesCpu) {
    CpuBackend backend;
    TestGeluLargeValues(backend);
}