#include "gpt2_model.h"
#include "backend.h"
#include "debug_utils.h"
#include <stdexcept>
#include <memory>
#include <numeric>
#ifdef ENABLE_PROFILING
#include <chrono>
#include <iostream>
#endif

GPT2Model::GPT2Model(
    Backend* backend,
    int vocab_size,
    int max_seq_len,
    int embed_dim,
    int num_heads,
    int num_layers,
    float epsilon
) : backend(backend),
    vocab_size(vocab_size),
    max_seq_len(max_seq_len),
    embed_dim(embed_dim),
    num_heads(num_heads),
    num_layers(num_layers),
    epsilon(epsilon)
{
    token_embeddings = std::make_unique<Embedding>(backend, vocab_size, embed_dim);
    position_embeddings = std::make_unique<Embedding>(backend, max_seq_len, embed_dim);

    for (int i = 0; i < num_layers; ++i) {
        transformer_blocks.push_back(std::make_unique<TransformerBlock>(backend, embed_dim, num_heads, epsilon));
    }

    ln_f = std::make_unique<LayerNorm>(backend, embed_dim, epsilon);
    lm_head = std::make_unique<Linear>(backend, embed_dim, vocab_size);
}

GPT2Model::~GPT2Model() {
}

void GPT2Model::load_weights(const std::map<std::string, std::vector<float>>& weights_map) {
    if (weights_map.count("wte.weight")) {
        token_embeddings->load_weights(weights_map.at("wte.weight"));
    } else {
        throw std::runtime_error("Missing wte.weight in weights map.");
    }

    if (weights_map.count("wpe.weight")) {
        position_embeddings->load_weights(weights_map.at("wpe.weight"));
    } else {
        throw std::runtime_error("Missing wpe.weight in weights map.");
    }

    for (int i = 0; i < num_layers; ++i) {
        std::string prefix = "h." + std::to_string(i) + ".";

        transformer_blocks[i]->load_weights(
            weights_map.at(prefix + "ln_1.weight"), weights_map.at(prefix + "ln_1.bias"),
            weights_map.at(prefix + "attn.w_q_data"), weights_map.at(prefix + "attn.b_q_data"),
            weights_map.at(prefix + "attn.w_k_data"), weights_map.at(prefix + "attn.b_k_data"),
            weights_map.at(prefix + "attn.w_v_data"), weights_map.at(prefix + "attn.b_v_data"),
            weights_map.at(prefix + "attn.c_proj.weight"), weights_map.at(prefix + "attn.c_proj.bias"),
            weights_map.at(prefix + "ln_2.weight"), weights_map.at(prefix + "ln_2.bias"),
            weights_map.at(prefix + "mlp.c_fc.weight"), weights_map.at(prefix + "mlp.c_fc.bias"),
            weights_map.at(prefix + "mlp.c_proj.weight"), weights_map.at(prefix + "mlp.c_proj.bias")
        );
    }

    if (weights_map.count("ln_f.weight") && weights_map.count("ln_f.bias")) {
        ln_f->load_weights(weights_map.at("ln_f.weight"), weights_map.at("ln_f.bias"));
    } else {
        throw std::runtime_error("Missing ln_f.weight or ln_f.bias in weights map.");
    }

    if (weights_map.count("lm_head.weight")) {
        const auto& lm_head_weight = weights_map.at("lm_head.weight");
        if (weights_map.count("lm_head.bias") && !weights_map.at("lm_head.bias").empty()) {
            lm_head->load_weights(lm_head_weight, weights_map.at("lm_head.bias"));
        } else {
            std::vector<float> lm_head_bias(vocab_size, 0.0f);
            lm_head->load_weights(lm_head_weight, lm_head_bias);
        }
    } else {
        throw std::runtime_error("Missing lm_head.weight in weights map.");
    }
}

std::unique_ptr<Tensor> GPT2Model::forward(const Tensor* input_token_ids) {
    if (input_token_ids->get_dtype() != MLLM_INT32) {
        throw std::runtime_error("Input token IDs must be an integer tensor.");
    }

#ifdef ENABLE_PROFILING
    auto total_start = std::chrono::high_resolution_clock::now();
#endif

    auto input_shape = input_token_ids->get_shape();
    int batch_size = input_shape[0];
    int seq_len = input_shape[1];

#ifdef ENABLE_PROFILING
    auto step_start = std::chrono::high_resolution_clock::now();
#endif
    auto token_embeds = token_embeddings->forward(input_token_ids);
#ifdef ENABLE_PROFILING
    auto step_end = std::chrono::high_resolution_clock::now();
    std::cout << "Token Embeddings took: " << std::chrono::duration<double>(step_end - step_start).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    step_start = std::chrono::high_resolution_clock::now();
#endif
    std::vector<int> position_ids_data(seq_len);
    std::iota(position_ids_data.begin(), position_ids_data.end(), 0);
    auto position_ids = backend->create_tensor({seq_len}, MLLM_INT32);
    position_ids->allocate();
    position_ids->copy_from_int(position_ids_data);
    auto position_embeds = position_embeddings->forward(position_ids.get());
#ifdef ENABLE_PROFILING
    step_end = std::chrono::high_resolution_clock::now();
    std::cout << "Positional Embeddings took: " << std::chrono::duration<double>(step_end - step_start).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    step_start = std::chrono::high_resolution_clock::now();
#endif
    auto hidden_states = backend->create_tensor(token_embeds->get_shape(), MLLM_FLOAT32);
    hidden_states->allocate();
    backend->add(token_embeds.get(), position_embeds.get(), hidden_states.get());
#ifdef ENABLE_PROFILING
    step_end = std::chrono::high_resolution_clock::now();
    std::cout << "Add Embeddings took: " << std::chrono::duration<double>(step_end - step_start).count() << " s" << std::endl;
#endif

    for (int i = 0; i < num_layers; ++i) {
#ifdef ENABLE_PROFILING
        step_start = std::chrono::high_resolution_clock::now();
#endif
        hidden_states = transformer_blocks[i]->forward(hidden_states.get());
#ifdef ENABLE_PROFILING
        step_end = std::chrono::high_resolution_clock::now();
        std::cout << "Transformer Block " << i << " took: " << std::chrono::duration<double>(step_end - step_start).count() << " s" << std::endl;
#endif
    }

#ifdef ENABLE_PROFILING
    step_start = std::chrono::high_resolution_clock::now();
#endif
    auto ln_f_output = ln_f->forward(hidden_states.get());
#ifdef ENABLE_PROFILING
    step_end = std::chrono::high_resolution_clock::now();
    std::cout << "Final LayerNorm took: " << std::chrono::duration<double>(step_end - step_start).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    step_start = std::chrono::high_resolution_clock::now();
#endif
    auto logits = lm_head->forward(ln_f_output.get());
#ifdef ENABLE_PROFILING
    step_end = std::chrono::high_resolution_clock::now();
    std::cout << "Language Model Head took: " << std::chrono::duration<double>(step_end - step_start).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    auto total_end = std::chrono::high_resolution_clock::now();
    std.cout << "Total forward pass took: " << std::chrono::duration<double>(total_end - total_start).count() << " s" << std::endl;
#endif

    return logits;
}
