#include <gtest/gtest.h>
#include "metal_backend.h"
#include "metal_tensor.h"
#include <vector>
#include <numeric>
#include <cmath>

TEST(MetalBackendTest, MatrixMultiply) {
    MetalBackend backend;

    int M = 2, K = 3, N = 2;
    std::vector<int> shapeA = {M, K};
    std::vector<int> shapeB = {K, N};
    std::vector<int> shapeC = {M, N};

    auto inA = backend.create_tensor(shapeA, MLLM_FLOAT32);
    auto inB = backend.create_tensor(shapeB, MLLM_FLOAT32);
    auto outC = backend.create_tensor(shapeC, MLLM_FLOAT32);
    inA->allocate();
    inB->allocate();
    outC->allocate();

    std::vector<float> dataA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> dataB = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    inA->copy_from_float(dataA);
    inB->copy_from_float(dataB);

    backend.matrix_multiply(inA.get(), inB.get(), outC.get());

    std::vector<float> resultC;
    outC->copy_to_float(resultC);

    std::vector<float> expectedC = {58.0f, 64.0f, 139.0f, 154.0f};
    ASSERT_EQ(resultC.size(), expectedC.size());
    for (size_t i = 0; i < resultC.size(); ++i) {
        EXPECT_FLOAT_EQ(resultC[i], expectedC[i]);
    }
}

TEST(MetalBackendTest, BatchedMatrixMultiply) {
    MetalBackend backend;

    int batch = 2;
    int heads = 2;
    int M = 2, K = 3, N = 2;
    std::vector<int> shapeA = {batch, heads, M, K};
    std::vector<int> shapeB = {batch, heads, K, N};
    std::vector<int> shapeC = {batch, heads, M, N};

    auto inA = backend.create_tensor(shapeA, MLLM_FLOAT32);
    auto inB = backend.create_tensor(shapeB, MLLM_FLOAT32);
    auto outC = backend.create_tensor(shapeC, MLLM_FLOAT32);
    inA->allocate();
    inB->allocate();
    outC->allocate();

    std::vector<float> dataA = {
        // Batch 0, Head 0
        1, 2, 3,
        4, 5, 6,
        // Batch 0, Head 1
        7, 8, 9,
        10, 11, 12,
        // Batch 1, Head 0
        -1, -2, -3,
        -4, -5, -6,
        // Batch 1, Head 1
        -7, -8, -9,
        -10, -11, -12
    };
    std::vector<float> dataB = {
        // Batch 0, Head 0
        1, 2,
        3, 4,
        5, 6,
        // Batch 0, Head 1
        7, 8,
        9, 10,
        11, 12,
        // Batch 1, Head 0
        -1, -2,
        -3, -4,
        -5, -6,
        // Batch 1, Head 1
        -7, -8,
        -9, -10,
        -11, -12
    };
    inA->copy_from_float(dataA);
    inB->copy_from_float(dataB);

    backend.matrix_multiply(inA.get(), inB.get(), outC.get());

    std::vector<float> resultC;
    outC->copy_to_float(resultC);

    std::vector<float> expectedC = {
        // Batch 0, Head 0
        22, 28,
        49, 64,
        // Batch 0, Head 1
        220, 244,
        301, 334,
        // Batch 1, Head 0
        22, 28,
        49, 64,
        // Batch 1, Head 1
        220, 244,
        301, 334
    };

    ASSERT_EQ(resultC.size(), expectedC.size());
    for (size_t i = 0; i < resultC.size(); ++i) {
        EXPECT_FLOAT_EQ(resultC[i], expectedC[i]);
    }
}

TEST(MetalBackendTest, Add) {
    MetalBackend backend;
    int size = 4;
    std::vector<int> shape = {size};

    auto inA = backend.create_tensor(shape, MLLM_FLOAT32);
    auto inB = backend.create_tensor(shape, MLLM_FLOAT32);
    auto outC = backend.create_tensor(shape, MLLM_FLOAT32);
    inA->allocate();
    inB->allocate();
    outC->allocate();

    std::vector<float> dataA = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> dataB = {5.0f, 6.0f, 7.0f, 8.0f};
    inA->copy_from_float(dataA);
    inB->copy_from_float(dataB);

    backend.add(inA.get(), inB.get(), outC.get());

    std::vector<float> resultC;
    outC->copy_to_float(resultC);

    std::vector<float> expectedC = {6.0f, 8.0f, 10.0f, 12.0f};
    ASSERT_EQ(resultC.size(), expectedC.size());
    for (size_t i = 0; i < resultC.size(); ++i) {
        EXPECT_FLOAT_EQ(resultC[i], expectedC[i]);
    }
}

TEST(MetalBackendTest, BroadcastAdd) {
    MetalBackend backend;
    int rows = 2, cols = 3;
    std::vector<int> shapeA = {rows, cols};
    std::vector<int> shapeB_bias = {cols};
    std::vector<int> shapeC = {rows, cols};

    auto inA = backend.create_tensor(shapeA, MLLM_FLOAT32);
    auto inB_bias = backend.create_tensor(shapeB_bias, MLLM_FLOAT32);
    auto outC = backend.create_tensor(shapeC, MLLM_FLOAT32);
    inA->allocate();
    inB_bias->allocate();
    outC->allocate();

    std::vector<float> dataA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> dataB_bias = {10.0f, 20.0f, 30.0f};
    inA->copy_from_float(dataA);
    inB_bias->copy_from_float(dataB_bias);

    backend.broadcast_add(inA.get(), inB_bias.get(), outC.get());

    std::vector<float> resultC;
    outC->copy_to_float(resultC);

    std::vector<float> expectedC = {11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f};
    ASSERT_EQ(resultC.size(), expectedC.size());
    for (size_t i = 0; i < resultC.size(); ++i) {
        EXPECT_FLOAT_EQ(resultC[i], expectedC[i]);
    }
}

TEST(MetalBackendTest, SoftmaxRowwise) {
    MetalBackend backend;
    int rows = 2;
    int cols = 4;
    std::vector<int> shape = {rows, cols};

    auto in = backend.create_tensor(shape, MLLM_FLOAT32);
    auto out = backend.create_tensor(shape, MLLM_FLOAT32);
    in->allocate();
    out->allocate();

    std::vector<float> data = {
        1.0f, 2.0f, 3.0f, 4.0f, // row 1
        -1.0f, 0.0f, 1.0f, 2.0f  // row 2
    };
    in->copy_from_float(data);

    backend.softmax_rowwise(in.get(), out.get());

    std::vector<float> result;
    out->copy_to_float(result);

    // Expected for row 1
    std::vector<float> expected_row1 = {0.032058603f, 0.087144316f, 0.236882818f, 0.643914263f};
    // Expected for row 2
    float sum_exp_row2 = exp(-1.0f) + exp(0.0f) + exp(1.0f) + exp(2.0f);
    std::vector<float> expected_row2(cols);
    expected_row2[0] = exp(-1.0f) / sum_exp_row2;
    expected_row2[1] = exp(0.0f) / sum_exp_row2;
    expected_row2[2] = exp(1.0f) / sum_exp_row2;
    expected_row2[3] = exp(2.0f) / sum_exp_row2;

    std::vector<float> expected;
    expected.insert(expected.end(), expected_row1.begin(), expected_row1.end());
    expected.insert(expected.end(), expected_row2.begin(), expected_row2.end());

    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result[i], expected[i], 1e-6);
    }
}

TEST(MetalBackendTest, Scale) {
    MetalBackend backend;
    int size = 4;
    std::vector<int> shape = {size};

    auto in = backend.create_tensor(shape, MLLM_FLOAT32);
    auto out = backend.create_tensor(shape, MLLM_FLOAT32);
    in->allocate();
    out->allocate();

    std::vector<float> data = {1.0f, 2.0f, -3.0f, 0.0f};
    in->copy_from_float(data);

    float scale_factor = 1.5f;
    backend.scale(in.get(), out.get(), scale_factor);

    std::vector<float> result;
    out->copy_to_float(result);

    std::vector<float> expected = {1.5f, 3.0f, -4.5f, 0.0f};
    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }
}

TEST(MetalBackendTest, Transpose) {
    MetalBackend backend;
    int d0 = 2, d1 = 2, d2 = 3;
    std::vector<int> in_shape = {d0, d1, d2};
    std::vector<int> out_shape = {d1, d0, d2};

    auto in = backend.create_tensor(in_shape, MLLM_FLOAT32);
    auto out = backend.create_tensor(out_shape, MLLM_FLOAT32);
    in->allocate();
    out->allocate();

    std::vector<float> data(in->get_size());
    std::iota(data.begin(), data.end(), 0.0f);
    in->copy_from_float(data);

    backend.transpose(in.get(), out.get(), 0, 1);

    std::vector<float> result;
    out->copy_to_float(result);

    std::vector<float> expected = {0.0f, 1.0f, 2.0f, 6.0f, 7.0f, 8.0f, 3.0f, 4.0f, 5.0f, 9.0f, 10.0f, 11.0f};
    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }
}

TEST(MetalBackendTest, Transpose2D) {
    MetalBackend backend;

    int height = 2;
    int width = 3;

    auto input = backend.create_tensor({height, width}, MLLM_FLOAT32);
    input->allocate();

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    input->copy_from_float(input_data);

    auto output = backend.create_tensor({width, height}, MLLM_FLOAT32);
    output->allocate();

    backend.transpose2d(input.get(), output.get());

    std::vector<float> result;
    output->copy_to_float(result);

    std::vector<float> expected = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};

    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }
}

// Helper function for the GELU approximation used in GPT-2 (updated to match Metal shader)
static float gpt2_gelu(float x) {
    float sigmoid_arg = 1.702f * x;
    float sigmoid_val = 1.0f / (1.0f + expf(-sigmoid_arg)); // Use expf for float
    return x * sigmoid_val;
}

TEST(MetalBackendTest, Gelu) {
    MetalBackend backend;
    int size = 5;
    std::vector<int> shape = {size};

    auto in = backend.create_tensor(shape, MLLM_FLOAT32);
    auto out = backend.create_tensor(shape, MLLM_FLOAT32);
    in->allocate();
    out->allocate();

    std::vector<float> data = {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f};
    in->copy_from_float(data);

    backend.gelu(in.get(), out.get());

    std::vector<float> result;
    out->copy_to_float(result);

    std::vector<float> expected;
    for (float val : data) {
        expected.push_back(gpt2_gelu(val));
    }

    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result[i], expected[i], 1e-6);
    }
}
