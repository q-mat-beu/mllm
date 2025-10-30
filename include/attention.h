#pragma once

#include "tensor.h"
#include "backend.h"
#include "linear.h"
#include <vector>
#include <memory>

class MultiHeadAttention {
public:
    MultiHeadAttention(Backend* backend, int embed_dim, int num_heads);
    ~MultiHeadAttention();

    std::unique_ptr<Tensor> forward(const Tensor* input, bool apply_mask = false);

    void load_weights(
        const std::vector<float>& w_q_data, const std::vector<float>& b_q_data,
        const std::vector<float>& w_k_data, const std::vector<float>& b_k_data,
        const std::vector<float>& w_v_data, const std::vector<float>& b_v_data,
        const std::vector<float>& w_o_data, const std::vector<float>& b_o_data
    );

private:
    Backend* backend;
    int embed_dim;
    int num_heads;
    int head_dim;

    std::unique_ptr<Linear> proj_q;
    std::unique_ptr<Linear> proj_k;
    std::unique_ptr<Linear> proj_v;
    std::unique_ptr<Linear> proj_o;
};