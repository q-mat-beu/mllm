#include "transformer_block.h"
#include "backend.h"
#include "debug_utils.h"
#include <memory>
#include <stdexcept>
#ifdef ENABLE_PROFILING
#include <chrono> // Added for timing
#include <iostream> // Added for printing timing
#endif

TransformerBlock::TransformerBlock(Backend* backend, int embed_dim, int num_heads, float epsilon)
    : backend(backend), embed_dim(embed_dim), num_heads(num_heads), epsilon(epsilon) {
    
    ln_1 = std::make_unique<LayerNorm>(backend, embed_dim, epsilon);
    attn = std::make_unique<MultiHeadAttention>(backend, embed_dim, num_heads);
    ln_2 = std::make_unique<LayerNorm>(backend, embed_dim, epsilon);
    
    fc_in = std::make_unique<Linear>(backend, embed_dim, 4 * embed_dim);
    fc_out = std::make_unique<Linear>(backend, 4 * embed_dim, embed_dim);
}

TransformerBlock::~TransformerBlock() {
}

void TransformerBlock::load_weights(
    const std::vector<float>& ln_1_gamma_data, const std::vector<float>& ln_1_beta_data,
    const std::vector<float>& attn_w_q_data, const std::vector<float>& attn_b_q_data,
    const std::vector<float>& attn_w_k_data, const std::vector<float>& attn_b_k_data,
    const std::vector<float>& attn_w_v_data, const std::vector<float>& attn_b_v_data,
    const std::vector<float>& attn_w_o_data, const std::vector<float>& attn_b_o_data,
    const std::vector<float>& ln_2_gamma_data, const std::vector<float>& ln_2_beta_data,
    const std::vector<float>& fc_in_weight_data, const std::vector<float>& fc_in_bias_data,
    const std::vector<float>& fc_out_weight_data, const std::vector<float>& fc_out_bias_data
) {
    ln_1->load_weights(ln_1_gamma_data, ln_1_beta_data);
    attn->load_weights(attn_w_q_data, attn_b_q_data, attn_w_k_data, attn_b_k_data, attn_w_v_data, attn_b_v_data, attn_w_o_data, attn_b_o_data);
    ln_2->load_weights(ln_2_gamma_data, ln_2_beta_data);
    fc_in->load_weights(fc_in_weight_data, fc_in_bias_data);
    fc_out->load_weights(fc_out_weight_data, fc_out_bias_data);
}

std::unique_ptr<Tensor> TransformerBlock::forward(const Tensor* input) {
#ifdef ENABLE_PROFILING
    auto start_block = std::chrono::high_resolution_clock::now();
#endif

#ifdef ENABLE_PROFILING
    auto start_ln1 = std::chrono::high_resolution_clock::now();
#endif
    auto ln1_output = ln_1->forward(input);
#ifdef ENABLE_PROFILING
    auto end_ln1 = std::chrono::high_resolution_clock::now();
    std::cout << "  LN1 took: " << std::chrono::duration<double>(end_ln1 - start_ln1).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    auto start_attn = std::chrono::high_resolution_clock::now();
#endif
    auto attn_output = attn->forward(ln1_output.get(), true);
#ifdef ENABLE_PROFILING
    auto end_attn = std::chrono::high_resolution_clock::now();
    std::cout << "  Attention took: " << std::chrono::duration<double>(end_attn - start_attn).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    auto start_add1 = std::chrono::high_resolution_clock::now();
#endif
    auto add1_output = backend->create_tensor(input->get_shape(), MLLM_FLOAT32);
    add1_output->allocate();
    backend->add(input, attn_output.get(), add1_output.get());
#ifdef ENABLE_PROFILING
    auto end_add1 = std::chrono::high_resolution_clock::now();
    std::cout << "  Add1 took: " << std::chrono::duration<double>(end_add1 - start_add1).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    auto start_ln2 = std::chrono::high_resolution_clock::now();
#endif
    auto ln2_output = ln_2->forward(add1_output.get());
#ifdef ENABLE_PROFILING
    auto end_ln2 = std::chrono::high_resolution_clock::now();
    std::cout << "  LN2 took: " << std::chrono::duration<double>(end_ln2 - start_ln2).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    auto start_fc_in = std::chrono::high_resolution_clock::now();
#endif
    auto fc_in_output = fc_in->forward(ln2_output.get());
#ifdef ENABLE_PROFILING
    auto end_fc_in = std::chrono::high_resolution_clock::now();
    std::cout << "  FC_in took: " << std::chrono::duration<double>(end_fc_in - start_fc_in).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    auto start_gelu = std::chrono::high_resolution_clock::now();
#endif
    auto gelu_output = backend->create_tensor(fc_in_output->get_shape(), MLLM_FLOAT32);
    gelu_output->allocate();
    backend->gelu(fc_in_output.get(), gelu_output.get());
#ifdef ENABLE_PROFILING
    auto end_gelu = std::chrono::high_resolution_clock::now();
    std::cout << "  GELU took: " << std::chrono::duration<double>(end_gelu - start_gelu).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    auto start_fc_out = std::chrono::high_resolution_clock::now();
#endif
    auto fc_out_output = fc_out->forward(gelu_output.get());
#ifdef ENABLE_PROFILING
    auto end_fc_out = std::chrono::high_resolution_clock::now();
    std::cout << "  FC_out took: " << std::chrono::duration<double>(end_fc_out - start_fc_out).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    auto start_add2 = std::chrono::high_resolution_clock::now();
#endif
    auto final_output = backend->create_tensor(input->get_shape(), MLLM_FLOAT32);
    final_output->allocate();
    backend->add(add1_output.get(), fc_out_output.get(), final_output.get());
#ifdef ENABLE_PROFILING
    auto end_add2 = std::chrono::high_resolution_clock::now();
    std::cout << "  Add2 took: " << std::chrono::duration<double>(end_add2 - start_add2).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    auto end_block = std::chrono::high_resolution_clock::now();
    std::cout << "Transformer Block total took: " << std::chrono::duration<double>(end_block - start_block).count() << " s" << std::endl;
#endif

    return final_output;
}
