#pragma once

#include "tensor.h"
#include "backend.h"
#include "embedding.h"
#include "transformer_block.h"
#include "layernorm.h"
#include "linear.h"
#include "inference.h"
#include <vector>
#include <map>
#include <string>
#include <memory>

class GPT2Model : public BaseModel {
public:
    GPT2Model(
        Backend* backend,
        int vocab_size,
        int max_seq_len,
        int embed_dim,
        int num_heads,
        int num_layers,
        float epsilon = 1e-5f
    );
    ~GPT2Model();

    void load_weights(const std::map<std::string, std::vector<float>>& weights_map) override;
    std::unique_ptr<Tensor> forward(const Tensor* input_token_ids) override;
    int get_vocab_size() const override { return vocab_size; }
    int get_max_seq_len() const override { return max_seq_len; }
    Backend* get_backend() const override { return backend; }
    int get_eos_token_id() const override { return 50256; }

private:
    Backend* backend;
    int vocab_size;
    int max_seq_len;
    int embed_dim;
    int num_heads;
    int num_layers;
    float epsilon;

    std::unique_ptr<Embedding> token_embeddings;
    std::unique_ptr<Embedding> position_embeddings;
    std::vector<std::unique_ptr<TransformerBlock>> transformer_blocks;
    std::unique_ptr<LayerNorm> ln_f;
    std::unique_ptr<Linear> lm_head;
};