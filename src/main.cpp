#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <iomanip> // For std::fixed and std::setprecision

#include "nlohmann/json.hpp"
#include "gpt2_inference.h" // Include the header for run_gpt2_inference

// Struct to hold parsed command-line arguments
struct CLIArgs {
    std::string prompt;
    std::string model_path;
    int max_tokens = 50;
    float temperature = 1.0f;
    int top_k = 40;
    float top_p = 0.9f;
    bool use_top_k = true;
    bool use_top_p = true;
    std::string vocab_path = "resources/vocab.json";
    std::string merges_path = "resources/merges.txt";
    std::string backend_type = "metal"; // Default to metal
};

// Function to parse command-line arguments
CLIArgs parse_args(int argc, char* argv[]) {
    CLIArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--prompt" && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (arg == "--max_tokens" && i + 1 < argc) {
            args.max_tokens = std::stoi(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            args.temperature = std::stof(argv[++i]);
        } else if (arg == "--top_k" && i + 1 < argc) {
            args.top_k = std::stoi(argv[++i]);
        } else if (arg == "--top_p" && i + 1 < argc) {
            args.top_p = std::stof(argv[++i]);
        } else if (arg == "--no_top_k") {
            args.use_top_k = false;
        } else if (arg == "--no_top_p") {
            args.use_top_p = false;
        } else if (arg == "--vocab_path" && i + 1 < argc) {
            args.vocab_path = argv[++i];
        } else if (arg == "--merges_path" && i + 1 < argc) {
            args.merges_path = argv[++i];
        } else if (arg == "--backend" && i + 1 < argc) {
            args.backend_type = argv[++i];
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            exit(1);
        }
    }
    if (args.prompt.empty() || args.model_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " --prompt \"Your text\" --model <path_to_weights.json> [--max_tokens N] [--temperature T] [--top_k K] [--top_p P] [--no_top_k] [--no_top_p] [--vocab_path P] [--merges_path M] [--backend <metal|cpu>]" << std::endl;
        exit(1);
    }
    return args;
}

int main(int argc, char* argv[]) {
    CLIArgs args = parse_args(argc, argv);

    // Call the GPT-2 specific inference function
    run_gpt2_inference(args.prompt, args.model_path, args.max_tokens, args.temperature, args.top_k, args.top_p,
                       args.use_top_k, args.use_top_p, args.vocab_path, args.merges_path, args.backend_type);

    return 0;
}