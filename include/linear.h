#pragma once

#include "tensor.h"
#include "backend.h"
#include <vector>
#include <memory>

class Linear {
public:
    Linear(Backend* backend, int in_features, int out_features);
    ~Linear();

    std::unique_ptr<Tensor> forward(const Tensor* input);

    void load_weights(const std::vector<float>& weight_data, const std::vector<float>& bias_data);

private:
    Backend* backend;
    std::unique_ptr<Tensor> weight; // Shape: (in_features, out_features)
    std::unique_ptr<Tensor> bias;   // Shape: (out_features)
    int in_features;
    int out_features;
};
