#ifndef SEMANTIC_COHERENCE_H
#define SEMANTIC_COHERENCE_H

#include <unordered_set>

#include "Transformer/TransformerLauguageModel.h"
#include "ContextWindow.h"
#include "sampler.h"
#include "tokenizer.h"


struct SentenceStructure {
    int wordsPerSentence;   // Max number of tokens to sample per sentence
    int sentenceCountMax;   // Max number of sentences
    
    int wordsCounter;       // Current word count
    int sentenceCounter;    // Current sentence count
    
    SentenceStructure() :
        wordsPerSentence(3), 
        sentenceCountMax(1),
        wordsCounter(0),
        sentenceCounter(0) {}
};


class SemanticCoherence {
public:
    
    // Sample a token context stream through a model via a given vocabulary and sentence structure.
    bool ProcessTokenStream(TransformerLauguageModel& model, Vocabulary& vocab, SamplingParams& sampler, ContextWindow& context, ContextWindow& current, SentenceStructure& sentenceStruct);
    
    // Print a token to the stream.
    void EmitToken(std::string word);
    
private:
    
    inline std::string lower(std::string& s);
    
    // Uppercase the first alphabetic character, lowercase the rest.
    inline std::string capitalize(std::string& s);
    
    inline bool is_alpha(std::string& s);
    inline bool is_capitalized(std::string& s);
    inline bool is_punct(std::string& s);
    inline bool is_plain_punct(std::string& s);
    inline bool is_end_punct(std::string& s);
    
    inline bool is_article(std::string& w);
    inline bool is_preposition(std::string& w);
    inline bool is_conjunction(std::string& w);
    inline bool is_pronoun(std::string& w);
    inline bool is_aux_verb(std::string& w);
    
    inline bool is_wordish(std::string& t);
    inline bool is_quote(std::string& t);
    inline bool is_closing_bracket(std::string& t);
    
    inline int count_unclosed(std::vector<std::string>& ctx, std::string& opener, std::string& closer);
    inline int count_unclosed_pair(std::vector<std::string>& ctx, std::string& opener, std::string& closer);
    inline bool starts_with_vowel_sound(std::string& s);
    inline bool appeared_recently(std::vector<std::string>& ctx, std::string& nextLower, int lastN);
    inline bool is_special(int id, Vocabulary& vocab);
    inline int sanitize_special(int id, Vocabulary& vocab);
    inline std::string safe_word(Vocabulary& vocab, int id);
    
};


extern SemanticCoherence semantic;
#endif
