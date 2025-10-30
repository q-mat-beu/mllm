#pragma once

#include <string>

void run_gpt2_inference(const std::string& prompt, const std::string& model_path,
                        int max_tokens, float temperature, int top_k, float top_p,
                        bool use_top_k, bool use_top_p,
                        const std::string& vocab_path, const std::string& merges_path,
                        const std::string& backend_type);