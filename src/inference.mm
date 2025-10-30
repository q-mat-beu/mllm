#include "inference.h"
#include "metal_backend.h"
#include "tensor.h"
#include "tokenizer.h"
#include "nlohmann/json.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>

// Function to load weights from a JSON file
std::map<std::string, std::vector<float>> load_weights_from_json(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open weights file: " + filepath);
    }
    nlohmann::json j;
    file >> j;

    std::map<std::string, std::vector<float>> weights_map;
    for (auto it = j.begin(); it != j.end(); ++it) {
        std::vector<float> data;
        if (it.value().is_array()) {
            for (const auto& val : it.value()) {
                data.push_back(val.get<float>());
            }
        } else {
            data.push_back(it.value().get<float>());
        }
        weights_map[it.key()] = data;
    }
    return weights_map;
}

// Function to sample the next token from logits
int sample_next_token(const std::vector<float>& logits, float temperature, int top_k, float top_p, bool use_top_k, bool use_top_p, std::mt19937& rng) {
    if (temperature == 0.0f) { // Greedy sampling
        return std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
    }

    std::vector<std::pair<float, int>> logits_with_indices(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        logits_with_indices[i] = {logits[i] / temperature, (int)i};
    }

    // Sort by logit value in descending order
    std::sort(logits_with_indices.rbegin(), logits_with_indices.rend());

    int k = logits.size();
    if (use_top_k && top_k > 0) {
        k = std::min(top_k, k);
    }

    std::vector<float> probabilities(k);
    float sum_exp = 0.0f;
    for (int i = 0; i < k; ++i) {
        probabilities[i] = std::exp(logits_with_indices[i].first);
        sum_exp += probabilities[i];
    }

    // Normalize to get probabilities
    for (int i = 0; i < k; ++i) {
        probabilities[i] /= sum_exp;
    }

    int nucleus_k = k;
    if (use_top_p && top_p < 1.0f) {
        float cumulative_prob = 0.0f;
        for (int i = 0; i < k; ++i) {
            cumulative_prob += probabilities[i];
            if (cumulative_prob > top_p) {
                nucleus_k = i + 1;
                break;
            }
        }
    }

    // Renormalize the probabilities of the nucleus
    float nucleus_sum_prob = 0.0f;
    for (int i = 0; i < nucleus_k; ++i) {
        nucleus_sum_prob += probabilities[i];
    }
    for (int i = 0; i < nucleus_k; ++i) {
        probabilities[i] /= nucleus_sum_prob;
    }

    // Sample from the truncated and renormalized distribution
    std::vector<float> nucleus_probabilities(probabilities.begin(), probabilities.begin() + nucleus_k);
    std::vector<int> nucleus_indices;
    for (int i = 0; i < nucleus_k; ++i) {
        nucleus_indices.push_back(logits_with_indices[i].second);
    }

    std::discrete_distribution<> dist(nucleus_probabilities.begin(), nucleus_probabilities.end());
    int sampled_index = dist(rng);
    return nucleus_indices[sampled_index];
}

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
) {
    std::cout << "Tokenizing prompt: \"" << prompt << "\"" << std::endl;
    std::vector<int> input_ids = tokenizer->encode(prompt);
    
    if (input_ids.empty()) {
        std::cout << "Prompt tokenized to empty sequence. Exiting." << std::endl;
        return;
    }

    std::cout << "Generated text: " << prompt;
    std::cout.flush();

    std::random_device rd;
    std::mt19937 rng(rd());

    int vocab_size = model->get_vocab_size();
    int max_seq_len = model->get_max_seq_len();
    int eos_token_id = model->get_eos_token_id(); // Get EOS token ID from model

    for (int i = 0; i < max_tokens; ++i) {
        // Create input tensor from current sequence
        // Model expects (batch_size, seq_len)
        auto input_tensor = model->get_backend()->create_tensor({1, (int)input_ids.size()}, MLLM_INT32);
        input_tensor->allocate();
        input_tensor->copy_from_int(input_ids);

        // Forward pass
        std::unique_ptr<Tensor> output_logits(model->forward(input_tensor.get()));

        // Get logits for the last token
        std::vector<float> all_logits;
        output_logits->copy_to_float(all_logits);
        
        // The output_logits shape is (1, current_seq_len, vocab_size)
        // We need the logits for the last token: output_logits[0][current_seq_len-1]
        std::vector<float> last_token_logits(vocab_size);
        size_t last_token_offset = (input_ids.size() - 1) * vocab_size;
        for (int j = 0; j < vocab_size; ++j) {
            last_token_logits[j] = all_logits[last_token_offset + j];
        }

        // Debug: Print top logits
        // std::vector<std::pair<float, int>> debug_logits_with_indices(vocab_size);
        // for (int j = 0; j < vocab_size; ++j) {
        //     debug_logits_with_indices[j] = {last_token_logits[j], j};
        // }
        // std::sort(debug_logits_with_indices.rbegin(), debug_logits_with_indices.rend());

        // std::cout << "\nTop 5 logits for next token:" << std::endl;
        // for (int j = 0; j < std::min(5, vocab_size); ++j) {
        //     std::string token_text = tokenizer->decode({debug_logits_with_indices[j].second});
        //     std::cout << std::fixed << std::setprecision(4)
        //               << "  " << j + 1 << ". Token: \"" << token_text << "\" (ID: "
        //               << debug_logits_with_indices[j].second << "), Logit: "
        //               << debug_logits_with_indices[j].first << std::endl;
        // }

        // Sample next token
        int next_token_id = sample_next_token(last_token_logits, temperature, top_k, top_p, use_top_k, use_top_p, rng);

        // Check for EOS token
        if (next_token_id == eos_token_id) {
            std::cout << "\n[EOS]" << std::endl;
            break;
        }

        // Decode and print
        std::string next_token_text = tokenizer->decode({next_token_id});
        std::cout << next_token_text;
        std::cout.flush();

        // Append to input for next iteration
        input_ids.push_back(next_token_id);

        // Prevent sequence from growing too long for positional embeddings
        if (input_ids.size() >= max_seq_len) {
            std::cout << "\n[Max sequence length reached]" << std::endl;
            break;
        }
    }
    std::cout << std::endl;
}
