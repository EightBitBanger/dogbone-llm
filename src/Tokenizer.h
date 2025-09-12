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
    struct SpecialTokens {
        int pad_id;
        int unk_id;
        int bos_id;
        int eos_id;
        int query_id;
        int response_id;
        SpecialTokens();
    } token;
    
    // Add a word to the vocab (if missing) and return its id
    int Add(const std::string& w);
    
    // Get id for a word; returns unk_id if not present
    int Get(const std::string& w) const;
    
    // Get the number of words in the vocabulary
    unsigned int Size() const;
    
    // Clear the vocabulary
    void Clear();
    
    // Reserve a size for an incoming vocabulary of words (for efficiency)
    void Reserve(size_t amount);
    
    // Build the special tokens
    void BuildSpecials();
    
    // Word -> id  (no insertion; falls back to unk_id)
    int operator[](const std::string& w) const;
    
    // id -> word  (safe; returns "<UNK>" if out-of-range)
    const std::string& operator[](int id) const;
    
    void SortVocabAlphabetically();
    
    // Fits the words in a corpus into the vocabulary
    void FitToCorpus(const std::vector<std::string>& corpus_texts);
    
    // Word <-> ID conversion maps
    std::unordered_map<std::string, int> word_to_id;
    std::vector<std::string> id_to_word;
    
};

std::string ToLower(const std::string& s);
std::vector<std::string> WhitespaceTokenize(const std::string& text);

// Note: now uses operator[] internally
std::vector<int> Encode(const Tokenizer& vocab, const std::string& text);

#endif
