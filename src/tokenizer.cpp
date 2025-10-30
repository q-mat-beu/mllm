#include "tokenizer.h"

/**
 * @brief Converts a Unicode codepoint to a UTF-8 string.
 *
 * @param cp The Unicode codepoint.
 * @return The UTF-8 encoded string.
 */
static std::string codepoint_to_utf8(int cp) {
    std::string out;
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

/**
 * @brief Splits a UTF-8 string into a vector of UTF-8 characters.
 *
 * @param s The UTF-8 string to split.
 * @return A vector of strings, where each string is a single UTF-8 character.
 */
static std::vector<std::string> utf8_split(const std::string &s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = 1;
        if ((c & 0x80) == 0x00) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        else {
            len = 1;
        }
        if (i + len > s.size()) len = s.size() - i;
        out.emplace_back(s.substr(i, len));
        i += len;
    }
    return out;
}

Tokenizer::Tokenizer(const std::string &vocab_path, const std::string &merges_path) {
    build_byte_encoder();
    // The tokenization regex is a simplified approximation of the original GPT-2 regex.
    token_pattern = std::regex("'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?\\S+|\\s+", std::regex::ECMAScript);
    load_vocab(vocab_path);
    load_merges(merges_path);
}

void Tokenizer::load_vocab(const std::string &vocab_path) {
    std::ifstream in(vocab_path);
    if (!in) throw std::runtime_error("could not open vocab file: " + vocab_path);
    json j;
    in >> j;
    for (auto it = j.begin(); it != j.end(); ++it) {
        std::string token = it.key();
        int id = it.value();
        encoder[token] = id;
        decoder[id] = token;
    }
}

void Tokenizer::load_merges(const std::string &merges_path) {
    std::ifstream in(merges_path);
    if (!in) throw std::runtime_error("could not open merges file: " + merges_path);
    std::string line;
    int idx = 0;
    while (std::getline(in, line)) {
        if (line.size() == 0) continue;
        if (line.rfind("#", 0) == 0) continue; // skip comments
        std::istringstream iss(line);
        std::string a, b;
        if (!(iss >> a >> b)) continue;
        bpe_ranks[{a, b}] = idx++;
    }
}

std::vector<int> Tokenizer::encode(const std::string &text, bool add_prefix_space) {
    std::string txt = text;
    if (add_prefix_space && (txt.empty() || txt[0] != ' ')) txt = std::string(" ") + txt;

    std::vector<int> out_ids;
    // Pre-tokenize the text using the regex
    auto begin = std::sregex_iterator(txt.begin(), txt.end(), token_pattern);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        std::string token = it->str();
        // Map the bytes of the token to their unicode representation
        std::string mapped;
        for (unsigned char c : token) mapped += byte_encoder[c];
        // Apply BPE to the mapped token
        std::string bpe_res = bpe(mapped);
        // Split the BPE result into individual tokens and get their IDs
        size_t pos = 0;
        while (pos < bpe_res.size()) {
            size_t sp = bpe_res.find(' ', pos);
            std::string piece;
            if (sp == std::string::npos) { piece = bpe_res.substr(pos); pos = bpe_res.size(); }
            else { piece = bpe_res.substr(pos, sp - pos); pos = sp + 1; }
            auto itv = encoder.find(piece);
            if (itv != encoder.end()) out_ids.push_back(itv->second);
            else {
                // Fallback for unknown tokens: split into individual characters
                auto cps = utf8_split(piece);
                for (const auto &cp : cps) {
                    auto it2 = encoder.find(cp);
                    if (it2 != encoder.end()) out_ids.push_back(it2->second);
                    else {
                        // Use a default token ID for unknown characters
                        out_ids.push_back(0);
                    }
                }
            }
        }
    }
    return out_ids;
}

std::string Tokenizer::decode(const std::vector<int> &ids) {
    // Join the tokens into a single string
    std::string text;
    for (int id : ids) {
        auto it = decoder.find(id);
        if (it != decoder.end()) text += it->second;
        else throw std::runtime_error("unknown id in decoder: " + std::to_string(id));
    }
    // Convert the unicode string back to bytes
    std::vector<unsigned char> bytes;
    auto cps = utf8_split(text);
    for (const auto &cpstr : cps) {
        auto it = byte_decoder.find(cpstr);
        if (it == byte_decoder.end()) {
            throw std::runtime_error("byte_decoder missing for codepoint in decode");
        }
        bytes.push_back(static_cast<unsigned char>(it->second));
    }
    return std::string(bytes.begin(), bytes.end());
}

void Tokenizer::build_byte_encoder() {
    byte_encoder.resize(256);
    // The byte-to-unicode mapping is the same as in the original GPT-2 tokenizer
    std::vector<int> bs;
    for (int i = (int)'!'; i <= (int)'~'; ++i) bs.push_back(i);
    for (int i = (int)0xA1; i <= (int)0xAC; ++i) bs.push_back(i);
    for (int i = (int)0xAE; i <= (int)0xFF; ++i) bs.push_back(i);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n += 1;
        }
    }
    for (size_t i = 0; i < bs.size(); ++i) {
        int byte = bs[i];
        int codepoint = cs[i];
        std::string u = codepoint_to_utf8(codepoint);
        if (byte >=0 && byte < 256) byte_encoder[byte] = u;
    }
    // Build the reverse mapping from unicode to bytes
    for (int b = 0; b < 256; ++b) {
        byte_decoder[byte_encoder[b]] = b;
    }
}

std::unordered_set<std::pair<std::string, std::string>, Tokenizer::PairHash> Tokenizer::get_pairs(const std::vector<std::string> &word) {
    std::unordered_set<std::pair<std::string, std::string>, PairHash> pairs;
    if (word.size() < 2) return pairs;
    for (size_t i = 0; i + 1 < word.size(); ++i) {
        pairs.insert({word[i], word[i+1]});
    }
    return pairs;
}

std::string Tokenizer::bpe(const std::string &token) {
    auto it_cache = cache.find(token);
    if (it_cache != cache.end()) return it_cache->second;

    std::vector<std::string> word = utf8_split(token);
    if (word.size() == 0) return std::string("");
    auto pairs = get_pairs(word);
    if (pairs.empty()) {
        std::string res = word[0];
        cache[token] = res;
        return res;
    }

    while (true) {
        // Find the best pair to merge
        int min_rank = INT32_MAX;
        std::pair<std::string, std::string> best_pair;
        bool found = false;
        for (const auto &p : pairs) {
            auto it = bpe_ranks.find(p);
            if (it != bpe_ranks.end()) {
                if (it->second < min_rank) {
                    min_rank = it->second;
                    best_pair = p;
                    found = true;
                }
            }
        }
        if (!found) break;

        // Merge the best pair
        std::vector<std::string> new_word;
        size_t i = 0;
        while (i < word.size()) {
            if (i + 1 < word.size() && word[i] == best_pair.first && word[i+1] == best_pair.second) {
                new_word.push_back(word[i] + word[i+1]);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                ++i;
            }
        }
        word.swap(new_word);
        if (word.size() == 1) break;
        pairs = get_pairs(word);
    }

    // Join the tokens with spaces
    std::string out;
    for (size_t i = 0; i < word.size(); ++i) {
        out += word[i];
        if (i + 1 < word.size()) out += ' ';
    }
    cache[token] = out;
    return out;
}
