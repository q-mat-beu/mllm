#pragma once

#include "tensor.h"
#include "tokenizer.h"
#include <string>
#include <vector>
#include <map>
#include <random>
#include <memory>

class Backend; // Forward declaration

// Abstract base class for a generic model
class BaseModel {
public:
    virtual ~BaseModel() = default;
    virtual std::unique_ptr<Tensor> forward(const Tensor* input) = 0;
    virtual void load_weights(const std::map<std::string, std::vector<float>>& weights_map) = 0;
    virtual int get_vocab_size() const = 0;
    virtual int get_max_seq_len() const = 0;
    virtual Backend* get_backend() const = 0;
    virtual int get_eos_token_id() const = 0; // Add this
};

// Function to load weights from a JSON file (moved from main.cpp)
std::map<std::string, std::vector<float>> load_weights_from_json(const std::string& filepath);

// Function to sample the next token from logits (moved from main.cpp)
int sample_next_token(const std::vector<float>& logits, float temperature, int top_k, float top_p, std::mt19937& rng);

// Generic inference function
void run_generic_inference(
    BaseModel* model,
    Tokenizer* tokenizer,
    const std::string& prompt,
    int max_tokens,
    float temperature,
    int top_k,
    float top_p,
    bool use_top_k,
    bool use_top_p
);