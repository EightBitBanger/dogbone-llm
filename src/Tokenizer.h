#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>

struct Vocabulary {
    // Token <-> word conversion
    std::unordered_map<std::string, int> word_to_id;
    std::vector<std::string> id_to_word;
    
    // Special tokens
    int pad_id, unk_id, bos_id, eos_id;
    
    Vocabulary();
    
    int Add(const std::string& w);
    int Get(const std::string& w) const;
    int Size() const;
    
    void BuildSpecials();
};

std::string ToLower(const std::string& s);

std::vector<std::string> WhitespaceTokenize(const std::string& text);

void SortVocabAlphabetically(Vocabulary& vocab);

void FitVocab(Vocabulary& vocab, const std::vector<std::string>& corpus_texts);

std::vector<int> Encode(const Vocabulary& vocab, const std::string& text, bool add_bos, bool add_eos);

#endif
