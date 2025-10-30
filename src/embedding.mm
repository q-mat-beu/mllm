#include "embedding.h"
#include "backend.h"
#include <stdexcept>

Embedding::Embedding(Backend* backend, int num_embeddings, int embedding_dim)
    : backend(backend), num_embeddings(num_embeddings), embedding_dim(embedding_dim) {
    
    weight = backend->create_tensor({num_embeddings, embedding_dim}, MLLM_FLOAT32);
    weight->allocate();
}

Embedding::~Embedding() {
}

void Embedding::load_weights(const std::vector<float>& weight_data) {
    if (weight_data.size() != weight->get_size()) {
        throw std::runtime_error("Incorrect size for weight data provided to Embedding layer.");
    }
    weight->copy_from_float(weight_data);
}

std::unique_ptr<Tensor> Embedding::forward(const Tensor* input_indices) {
    if (input_indices->get_dtype() != MLLM_INT32) {
        throw std::runtime_error("Embedding input must be an integer tensor.");
    }

    auto input_shape = input_indices->get_shape();
    std::vector<int> output_shape;
    if (input_shape.size() == 1) {
        output_shape = {input_shape[0], embedding_dim};
    } else if (input_shape.size() == 2) {
        output_shape = {input_shape[0], input_shape[1], embedding_dim};
    } else {
        throw std::runtime_error("Embedding forward only supports 1D or 2D input.");
    }

    auto output = backend->create_tensor(output_shape, MLLM_FLOAT32);
    output->allocate();

    backend->lookup(weight.get(), input_indices, output.get());

    return output;
}
