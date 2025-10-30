#include "gpt2_inference.h"
#include "inference.h"
#include "gpt2_model.h"
#include "metal_backend.h"
#include "cpu_backend.h"
#include "tokenizer.h"

#include <iostream>
#include <stdexcept>

void run_gpt2_inference(const std::string& prompt, const std::string& model_path,
                        int max_tokens, float temperature, int top_k, float top_p,
                        bool use_top_k, bool use_top_p,
                        const std::string& vocab_path, const std::string& merges_path,
                        const std::string& backend_type) {
    
    std::unique_ptr<Backend> backend;

    if (backend_type == "metal") {
        std::cout << "Initializing Metal Backend..." << std::endl;
        backend = std::make_unique<MetalBackend>();
    } else if (backend_type == "cpu") {
        std::cout << "Initializing CPU Backend..." << std::endl;
        backend = std::make_unique<CpuBackend>();
    } else {
        throw std::runtime_error("Unknown backend type: " + backend_type + ". Supported types are 'metal' and 'cpu'.");
    }

    std::cout << "Loading Tokenizer from " << vocab_path << " and " << merges_path << "..." << std::endl;
    Tokenizer tokenizer(vocab_path, merges_path);

    std::cout << "Loading model weights from " << model_path << "..." << std::endl;
    std::map<std::string, std::vector<float>> weights_map = load_weights_from_json(model_path);

    // Model parameters (hardcoded for GPT-2 small for now)
    int vocab_size = 50257;
    int max_seq_len = 1024;
    int embed_dim = 768;
    int num_heads = 12;
    int num_layers = 12; // GPT-2 small has 12 layers
    float epsilon = 1e-5f;

    std::cout << "Initializing GPT2Model..." << std::endl;
    GPT2Model model(backend.get(), vocab_size, max_seq_len, embed_dim, num_heads, num_layers, epsilon);

    std::cout << "Loading weights into GPT2Model..." << std::endl;
    model.load_weights(weights_map);

    run_generic_inference(&model, &tokenizer, prompt, max_tokens, temperature, top_k, top_p, use_top_k, use_top_p);
}