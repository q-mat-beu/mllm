#pragma once

#include "tensor.h"
#include "backend.h"
#include "layernorm.h"
#include "attention.h"
#include "linear.h"
#include <vector>
#include <memory>

class TransformerBlock {
public:
    TransformerBlock(Backend* backend, int embed_dim, int num_heads, float epsilon = 1e-5f);
    ~TransformerBlock();

    std::unique_ptr<Tensor> forward(const Tensor* input);

    void load_weights(
        const std::vector<float>& ln_1_gamma_data, const std::vector<float>& ln_1_beta_data,
        const std::vector<float>& attn_w_q_data, const std::vector<float>& attn_b_q_data,
        const std::vector<float>& attn_w_k_data, const std::vector<float>& attn_b_k_data,
        const std::vector<float>& attn_w_v_data, const std::vector<float>& attn_b_v_data,
        const std::vector<float>& attn_w_o_data, const std::vector<float>& attn_b_o_data,
        const std::vector<float>& ln_2_gamma_data, const std::vector<float>& ln_2_beta_data,
        const std::vector<float>& fc_in_weight_data, const std::vector<float>& fc_in_bias_data,
        const std::vector<float>& fc_out_weight_data, const std::vector<float>& fc_out_bias_data
    );

private:
    Backend* backend;
    std::unique_ptr<LayerNorm> ln_1;
    std::unique_ptr<MultiHeadAttention> attn;
    std::unique_ptr<LayerNorm> ln_2;
    std::unique_ptr<Linear> fc_in;
    std::unique_ptr<Linear> fc_out;

    int embed_dim;
    int num_heads;
    float epsilon;
};