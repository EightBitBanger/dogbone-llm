#ifndef SEMANTIC_COHERENCE_H
#define SEMANTIC_COHERENCE_H

#include <unordered_set>

#include "sampler.h"
#include "tokenizer.h"
#include "ContextWindow.h"

#include "Transformer/LanguageModel.h"
#include "WeightedMemories.h"


struct SentenceStructure {
    int wordsPerSentenceMax; // Maximum number of tokens to sample per sentence
    int wordsPerSentenceMin; // Minimum number of tokens to sample per sentence
    int sentenceCountMax;    // Max number of sentences
    
    int wordsCounter;        // Current word count
    int sentenceCounter;     // Current sentence count
    
    SentenceStructure() :
        wordsPerSentenceMax(0), 
        wordsPerSentenceMin(0), 
        sentenceCountMax(0),
        wordsCounter(0),
        sentenceCounter(0) {}
};

class SemanticCoherence {
public:
    
    // Sample a token context stream through a model via a given vocabulary and sentence structure.
    bool ProcessTokenStream(LanguageModel& model, Tokenizer& vocab, TokenSampler& Sampler, SamplingParams& samplerParams, 
                            ContextWindow& context, ContextWindow& current, SentenceStructure& sentenceStruct);
    
    std::string lower(std::string& s);
    
    // Uppercase the first alphabetic character, lowercase the rest.
    std::string capitalize(std::string& s);
    
    bool is_alpha(std::string& s);
    bool is_capitalized(std::string& s);
    bool is_punct(std::string& s);
    bool is_plain_punct(std::string& s);
    bool is_end_punct(std::string& s);
    
    bool is_article(std::string& w);
    bool is_preposition(std::string& w);
    bool is_conjunction(std::string& w);
    bool is_pronoun(std::string& w);
    bool is_aux_verb(std::string& w);
    
    bool is_wordish(std::string& t);
    bool is_quote(std::string& t);
    bool is_closing_bracket(std::string& t);
    
    int count_unclosed(std::vector<std::string>& ctx, std::string& opener, std::string& closer);
    int count_unclosed_pair(std::vector<std::string>& ctx, std::string& opener, std::string& closer);
    bool starts_with_vowel_sound(std::string& s);
    bool appeared_recently(std::vector<std::string>& ctx, std::string& nextLower, int lastN);
    bool is_special(int id, Tokenizer& vocab);
    std::string safe_word(Tokenizer& vocab, int id);
    
};

#endif
