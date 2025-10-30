#include <gtest/gtest.h>
#include "backend.h"
#include "metal_backend.h"
#include "cpu_backend.h"
#include "tensor.h"
#include <vector>
#include <memory>
#include <cmath> // For INFINITY

template<typename BackendType>
void TestCausalMaskAppliesMaskCorrectly(BackendType& backend) {
    int batch_size = 1;
    int num_heads = 2;
    int seq_len = 4;

    std::vector<int> shape = {batch_size, num_heads, seq_len, seq_len};
    auto scores = backend.create_tensor(shape, MLLM_FLOAT32);
    scores->allocate();

    // Fill with ones
    std::vector<float> data(scores->get_size(), 1.0f);
    scores->copy_from_float(data);

    // Apply the mask
    backend.apply_causal_mask(scores.get());

    // Copy back and check
    std::vector<float> result;
    scores->copy_to_float(result);

    // The CPU backend uses -INFINITY, Metal uses a large negative number.
    // We need to check for both.
    float masked_value_cpu = -INFINITY;
    float masked_value_metal = -1.0e9f; // This is what the Metal shader likely uses

    for (int batch = 0; batch < batch_size * num_heads; ++batch) {
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float val = result[batch * seq_len * seq_len + i * seq_len + j];
                if (j > i) {
                    // Check if it's either -INFINITY or the Metal masked value
                    if (std::isinf(val) && std::signbit(val) == std::signbit(masked_value_cpu)) {
                        // Correct for CPU backend
                    } else {
                        EXPECT_NEAR(val, masked_value_metal, 1e-6); // Correct for Metal backend
                    }
                } else {
                    EXPECT_FLOAT_EQ(val, 1.0f);
                }
            }
        }
    }
}

TEST(CausalMaskTest, AppliesMaskCorrectlyMetal) {
    MetalBackend backend;
    TestCausalMaskAppliesMaskCorrectly(backend);
}

TEST(CausalMaskTest, AppliesMaskCorrectlyCpu) {
    CpuBackend backend;
    TestCausalMaskAppliesMaskCorrectly(backend);
}