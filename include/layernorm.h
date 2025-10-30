#pragma once

#include "tensor.h"
#include "backend.h"
#include <vector>
#include <memory>

class LayerNorm {
public:
    LayerNorm(Backend* backend, int normalized_shape, float epsilon = 1e-5f);
    ~LayerNorm();

    std::unique_ptr<Tensor> forward(const Tensor* input);

    void load_weights(const std::vector<float>& gamma_data, const std::vector<float>& beta_data);

private:
    Backend* backend;
    std::unique_ptr<Tensor> gamma; // gain
    std::unique_ptr<Tensor> beta;  // bias
    int normalized_shape;
    float epsilon;
};