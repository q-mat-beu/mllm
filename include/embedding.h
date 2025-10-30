#pragma once

#include "tensor.h"
#include "backend.h"
#include <vector>
#include <memory>

class Embedding {
public:
    Embedding(Backend* backend, int num_embeddings, int embedding_dim);
    ~Embedding();

    std::unique_ptr<Tensor> forward(const Tensor* input_indices);

    void load_weights(const std::vector<float>& weight_data);

private:
    Backend* backend;
    std::unique_ptr<Tensor> weight; // Shape: (num_embeddings, embedding_dim)
    int num_embeddings;
    int embedding_dim;
};