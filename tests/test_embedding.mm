#include <gtest/gtest.h>
#include "backend.h"
#include "metal_backend.h"
#include "cpu_backend.h"
#include "tensor.h"
#include "embedding.h"
#include <vector>
#include <numeric>
#include <memory>

template<typename BackendType>
void TestEmbeddingLayerForward(BackendType& backend) {
    int num_embeddings = 10;
    int embedding_dim = 4;
    int seq_len = 3;

    Embedding layer(&backend, num_embeddings, embedding_dim);
    
    auto input_indices = backend.create_tensor({seq_len}, MLLM_INT32);
    input_indices->allocate();

    std::vector<float> weight_data(num_embeddings * embedding_dim);
    std::iota(weight_data.begin(), weight_data.end(), 0.0f); // Fill with 0, 1, 2, ...
    layer.load_weights(weight_data);

    std::vector<int> index_data = {0, 3, 1};
    input_indices->copy_from_int(index_data);

    auto output = layer.forward(input_indices.get());

    std::vector<float> result;
    output->copy_to_float(result);

    // Expected result (manual lookup)
    // Index 0 -> row 0 of weights -> [0, 1, 2, 3]
    // Index 3 -> row 3 of weights -> [12, 13, 14, 15] (3 * embedding_dim)
    // Index 1 -> row 1 of weights -> [4, 5, 6, 7] (1 * embedding_dim)
    std::vector<float> expected = {
        0.0f, 1.0f, 2.0f, 3.0f,
        12.0f, 13.0f, 14.0f, 15.0f,
        4.0f, 5.0f, 6.0f, 7.0f
    };

    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }
}

TEST(EmbeddingLayerTest, ForwardMetal) {
    MetalBackend backend;
    TestEmbeddingLayerForward(backend);
}

TEST(EmbeddingLayerTest, ForwardCpu) {
    CpuBackend backend;
    TestEmbeddingLayerForward(backend);
}
