#include <gtest/gtest.h>
#include "cpu_backend.h"
#include "cpu_tensor.h"
#include <vector>
#include <numeric>
#include <cmath>

TEST(CpuBackendTest, MatrixMultiply) {
    CpuBackend backend;

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

TEST(CpuBackendTest, Add) {
    CpuBackend backend;
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

TEST(CpuBackendTest, BroadcastAdd) {
    CpuBackend backend;
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

TEST(CpuBackendTest, SoftmaxRowwise) {
    CpuBackend backend;
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

TEST(CpuBackendTest, Scale) {
    CpuBackend backend;
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

TEST(CpuBackendTest, Transpose) {
    CpuBackend backend;
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

TEST(CpuBackendTest, Transpose2D) {
    CpuBackend backend;

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

static float gpt2_gelu(float x) {
    float sigmoid_arg = 1.702f * x;
    float sigmoid_val = 1.0f / (1.0f + expf(-sigmoid_arg));
    return x * sigmoid_val;
}

TEST(CpuBackendTest, Gelu) {
    CpuBackend backend;
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

TEST(CpuBackendTest, LayerNorm) {
    CpuBackend backend;
    int normalized_shape = 3;
    float epsilon = 1e-5f;
    std::vector<int> shape = {2, normalized_shape};

    auto in = backend.create_tensor(shape, MLLM_FLOAT32);
    auto out = backend.create_tensor(shape, MLLM_FLOAT32);
    auto gamma = backend.create_tensor({normalized_shape}, MLLM_FLOAT32);
    auto beta = backend.create_tensor({normalized_shape}, MLLM_FLOAT32);

    in->allocate();
    out->allocate();
    gamma->allocate();
    beta->allocate();

    std::vector<float> in_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> gamma_data = {1.0f, 1.0f, 1.0f};
    std::vector<float> beta_data = {0.0f, 0.0f, 0.0f};

    in->copy_from_float(in_data);
    gamma->copy_from_float(gamma_data);
    beta->copy_from_float(beta_data);

    backend.layernorm(in.get(), out.get(), gamma.get(), beta.get(), epsilon);

    std::vector<float> result;
    out->copy_to_float(result);

    // Expected values for input {1,2,3}
    // Mean = (1+2+3)/3 = 2
    // Variance = ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = (1+0+1)/3 = 2/3
    // InvStd = 1 / sqrt(2/3 + 1e-5) = 1 / sqrt(0.666666 + 0.00001) = 1 / sqrt(0.666676) = 1 / 0.816502 = 1.22473
    // (1-2)*1.22473 = -1.22473
    // (2-2)*1.22473 = 0
    // (3-2)*1.22473 = 1.22473

    // Expected values for input {4,5,6}
    // Mean = (4+5+6)/3 = 5
    // Variance = ((4-5)^2 + (5-5)^2 + (6-5)^2)/3 = (1+0+1)/3 = 2/3
    // InvStd = 1.22473
    // (4-5)*1.22473 = -1.22473
    // (5-5)*1.22473 = 0
    // (6-5)*1.22473 = 1.22473

    std::vector<float> expected = {
        -1.22473f, 0.0f, 1.22473f,
        -1.22473f, 0.0f, 1.22473f
    };

    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result[i], expected[i], 1e-5);
    }
}

TEST(CpuBackendTest, Lookup) {
    CpuBackend backend;
    int num_embeddings = 5;
    int embedding_dim = 4;
    std::vector<int> weights_shape = {num_embeddings, embedding_dim};
    std::vector<int> indices_shape = {3};
    std::vector<int> out_shape = {3, embedding_dim};

    auto weights = backend.create_tensor(weights_shape, MLLM_FLOAT32);
    auto indices = backend.create_tensor(indices_shape, MLLM_INT32);
    auto out = backend.create_tensor(out_shape, MLLM_FLOAT32);

    weights->allocate();
    indices->allocate();
    out->allocate();

    std::vector<float> weights_data = {
        0.0f, 0.1f, 0.2f, 0.3f, // embedding 0
        1.0f, 1.1f, 1.2f, 1.3f, // embedding 1
        2.0f, 2.1f, 2.2f, 2.3f, // embedding 2
        3.0f, 3.1f, 3.2f, 3.3f, // embedding 3
        4.0f, 4.1f, 4.2f, 4.3f  // embedding 4
    };
    std::vector<int> indices_data = {1, 0, 3};

    weights->copy_from_float(weights_data);
    indices->copy_from_int(indices_data);

    backend.lookup(weights.get(), indices.get(), out.get());

    std::vector<float> result;
    out->copy_to_float(result);

    std::vector<float> expected = {
        1.0f, 1.1f, 1.2f, 1.3f,
        0.0f, 0.1f, 0.2f, 0.3f,
        3.0f, 3.1f, 3.2f, 3.3f
    };

    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }
}

TEST(CpuBackendTest, ApplyCausalMask) {
    CpuBackend backend;
    int batch = 1, heads = 1, seq = 4;
    std::vector<int> shape = {batch, heads, seq, seq};

    auto scores = backend.create_tensor(shape, MLLM_FLOAT32);
    scores->allocate();

    std::vector<float> scores_data = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    scores->copy_from_float(scores_data);

    backend.apply_causal_mask(scores.get());

    std::vector<float> result;
    scores->copy_to_float(result);

    std::vector<float> expected = {
        1.0f, -INFINITY, -INFINITY, -INFINITY,
        5.0f, 6.0f, -INFINITY, -INFINITY,
        9.0f, 10.0f, 11.0f, -INFINITY,
        13.0f, 14.0f, 15.0f, 16.0f
    };

    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); ++i) {
        if (std::isinf(expected[i])) {
            ASSERT_TRUE(std::isinf(result[i]));
            ASSERT_EQ(std::signbit(result[i]), std::signbit(expected[i]));
        } else {
            EXPECT_FLOAT_EQ(result[i], expected[i]);
        }
    }
}
