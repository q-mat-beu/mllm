#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <regex>
#include <algorithm>
#include <cstdint>
#include <cassert>

#include "nlohmann/json.hpp"
using json = nlohmann::json;

// UTF-8 helpers
static std::string codepoint_to_utf8(int cp);
static std::vector<std::string> utf8_split(const std::string &s);

/**
 * @brief A C++ implementation of the GPT-2 Byte-Pair Encoding (BPE) tokenizer.
 *
 * This tokenizer is responsible for converting text into a sequence of integer tokens
 * (encoding) and vice-versa (decoding). It is compatible with the GPT-2 tokenizer
 * from Hugging Face and OpenAI.
 */
class Tokenizer {
public:
    /**
     * @brief Constructs a new Tokenizer object.
     *
     * @param vocab_path Path to the vocab.json file.
     * @param merges_path Path to the merges.txt file.
     */
    Tokenizer(const std::string &vocab_path, const std::string &merges_path);

    /**
     * @brief Encodes a string of text into a sequence of token IDs.
     *
     * @param text The text to encode.
     * @param add_prefix_space Whether to add a prefix space to the text before encoding.
     * @return A vector of integer token IDs.
     */
    std::vector<int> encode(const std::string &text, bool add_prefix_space = false);

    /**
     * @brief Decodes a sequence of token IDs back into a string of text.
     *
     * @param ids The vector of integer token IDs to decode.
     * @return The decoded string.
     */
    std::string decode(const std::vector<int> &ids);

private:
    /**
     * @brief Loads the vocabulary from a vocab.json file.
     *
     * @param vocab_path Path to the vocab.json file.
     */
    void load_vocab(const std::string &vocab_path);

    /**
     * @brief Loads the BPE merges from a merges.txt file.
     *
     * @param merges_path Path to the merges.txt file.
     */
    void load_merges(const std::string &merges_path);

    // token -> id mapping
    std::unordered_map<std::string, int> encoder;
    // id -> token mapping
    std::unordered_map<int, std::string> decoder;

    // Struct for hashing pairs of strings (for bpe_ranks)
    struct PairHash {
        size_t operator()(const std::pair<std::string, std::string> &p) const noexcept {
            return std::hash<std::string>()(p.first) ^ (std::hash<std::string>()(p.second) << 1);
        }
    };
    // BPE merge ranks
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> bpe_ranks;

    // BPE cache
    std::unordered_map<std::string, std::string> cache;

    // Byte encoder/decoder
    std::vector<std::string> byte_encoder; // 256 -> utf8 string
    std::unordered_map<std::string, int> byte_decoder; // utf8 string -> original byte

    // Tokenization regex
    std::regex token_pattern;

    /**
     * @brief Builds the byte-to-unicode mapping.
     */
    void build_byte_encoder();

    /**
     * @brief Gets all adjacent pairs of tokens in a word.
     *
     * @param word The vector of tokens.
     * @return A set of pairs of adjacent tokens.
     */
    static std::unordered_set<std::pair<std::string, std::string>, PairHash> get_pairs(const std::vector<std::string> &word);

    /**
     * @brief Applies the BPE algorithm to a token.
     *
     * @param token The token to apply BPE to.
     * @return A string of space-separated BPE tokens.
     */
    std::string bpe(const std::string &token);
};
