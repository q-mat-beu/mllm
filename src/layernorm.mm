#include "layernorm.h"
#include "backend.h"
#include <stdexcept>

LayerNorm::LayerNorm(Backend* backend, int normalized_shape, float epsilon)
    : backend(backend), normalized_shape(normalized_shape), epsilon(epsilon) {
    
    gamma = backend->create_tensor({normalized_shape}, MLLM_FLOAT32);
    beta = backend->create_tensor({normalized_shape}, MLLM_FLOAT32);

    gamma->allocate();
    beta->allocate();
}

LayerNorm::~LayerNorm() {
}

void LayerNorm::load_weights(const std::vector<float>& gamma_data, const std::vector<float>& beta_data) {
    if (gamma_data.size() != gamma->get_size()) {
        throw std::runtime_error("Incorrect size for gamma data provided to LayerNorm layer.");
    }
    if (beta_data.size() != beta->get_size()) {
        throw std::runtime_error("Incorrect size for beta data provided to LayerNorm layer.");
    }
    gamma->copy_from_float(gamma_data);
    beta->copy_from_float(beta_data);
}

std::unique_ptr<Tensor> LayerNorm::forward(const Tensor* input) {
    if (input->get_shape().back() != normalized_shape) {
        throw std::runtime_error("LayerNorm input shape mismatch.");
    }

    auto output = backend->create_tensor(input->get_shape(), MLLM_FLOAT32);
    output->allocate();

    backend->layernorm(input, output.get(), gamma.get(), beta.get(), epsilon);

    return output;
}
