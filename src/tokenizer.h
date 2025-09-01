#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>

class Tokenizer {
public:
    
    // Special tokens
    struct SpecialTokens {
        int pad_id;
        int unk_id;
        int bos_id;
        int eos_id;
        int query_id;
        int response_id;
        SpecialTokens();
    } token;
    
    int Add(const std::string& w);
    int Get(const std::string& w) const;
    
    
    // Build the special tokens.
    void BuildSpecials();
    
    // Word <-> ID conversion.
    std::unordered_map<std::string, int> word_to_id;
    std::vector<std::string> id_to_word;
};

std::string ToLower(const std::string& s);

std::vector<std::string> WhitespaceTokenize(const std::string& text);

void SortVocabAlphabetically(Tokenizer& vocab);

void FitVocab(Tokenizer& vocab, const std::vector<std::string>& corpus_texts);

std::vector<int> Encode(const Tokenizer& vocab, const std::string& text, bool add_bos, bool add_eos);

#endif
