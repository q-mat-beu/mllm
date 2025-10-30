#include <gtest/gtest.h>
#include "tokenizer.h"

TEST(TokenizerTest, EncodeDecode) {
    Tokenizer tokenizer("resources/vocab.json", "resources/merges.txt");
    std::string text = "hello world";
    std::vector<int> encoded = tokenizer.encode(text);
    std::string decoded = tokenizer.decode(encoded);
    EXPECT_EQ(text, decoded);
}

TEST(TokenizerTest, KnownTokenization) {
    Tokenizer tokenizer("resources/vocab.json", "resources/merges.txt");
    std::string text = "hello world";
    std::vector<int> encoded = tokenizer.encode(text);
    std::vector<int> expected = {31373, 995};
    EXPECT_EQ(encoded, expected);
}

TEST(TokenizerTest, EmptyString) {
    Tokenizer tokenizer("resources/vocab.json", "resources/merges.txt");
    std::string text = "";
    std::vector<int> encoded = tokenizer.encode(text);
    EXPECT_TRUE(encoded.empty());
    std::string decoded = tokenizer.decode(encoded);
    EXPECT_EQ(text, decoded);
}

TEST(TokenizerTest, SpecialCharacters) {
    Tokenizer tokenizer("resources/vocab.json", "resources/merges.txt");
    std::string text = "a!@#$%^&*()_+-=[]{}|;':,./<>?`~b";
    std::vector<int> encoded = tokenizer.encode(text);
    std::string decoded = tokenizer.decode(encoded);
    EXPECT_EQ(text, decoded);
}





