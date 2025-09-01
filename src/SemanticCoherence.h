#ifndef SEMANTIC_COHERENCE_H
#define SEMANTIC_COHERENCE_H

#include <unordered_set>

#include "sampler.h"
#include "tokenizer.h"
#include "ContextWindow.h"

#include "Transformer/LauguageModel.h"
#include "GrammaticalCongruence.h"


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

class GrammarReranker {
public:
    // Holds a reference to your vocab so we never go out of sync.
    // id_to_word must outlive this object.
    explicit GrammarReranker(const std::vector<std::string>& id_to_word);
    
    // Re-rank candidates in-place based on cheap grammar rules.
    // - context_ids: recent token IDs (we look back up to 16)
    // - lambda: strength of grammar bonus (added to logits)
    // - hard_select: if true, decisively pick the best one
    void Apply(std::vector<TokenCandidate>& candidates,
               const std::vector<int>& context_ids,
               float lambda = 1.25f,
               bool hard_select = false) const;
    
private:
    // --- Tag bits
    enum GTag : unsigned {
        G_UNKNOWN   = 0,
        G_DET       = 1u<<0,
        G_NOUN_SG   = 1u<<1,
        G_NOUN_PL   = 1u<<2,
        G_VERB_BASE = 1u<<3,
        G_VERB_3SG  = 1u<<4,
        G_VERB_PAST = 1u<<5,
        G_VERB_GER  = 1u<<6,
        G_ADJ       = 1u<<7,
        G_ADV       = 1u<<8,
        G_PRON_SUBJ = 1u<<9,
        G_PRON_OBJ  = 1u<<10,
        G_MODAL     = 1u<<11,
        G_PREP      = 1u<<12,
        G_TO        = 1u<<13,
        G_PUNCT     = 1u<<14
    };
    
    // Vocab + fast membership sets (exact and lowercased)
    const std::vector<std::string>& mIdToWord;
    std::unordered_set<std::string> mVocabExact;
    std::unordered_set<std::string> mVocabLower;
    
    // Tiny lexicon for function words & auxiliaries
    std::unordered_map<std::string, unsigned> mLex;
    
    // Heuristics + helpers
    static std::string LowerASCII(const std::string& s);
    std::string ToSafeLowerIfInVocab(const std::string& w) const;
    unsigned GuessTag(const std::string& token) const;
    static bool StartsWithVowel(const std::string& w);
    static bool IsPastTenseHint(const std::vector<std::string>& ctxLower);
    static int  RecentSubject(const std::vector<std::string>& ctxLower);
    float ScoreGrammar(const std::vector<std::string>& ctxLower,
                    const std::string& candidate) const;
    
    static void SoftmaxOverLogits(std::vector<TokenCandidate>& cands);
};


class SemanticCoherence {
public:
    
    // Sample a token context stream through a model via a given vocabulary and sentence structure.
    bool ProcessTokenStream(LauguageModel& model, Tokenizer& vocab, SamplingParams& sampler, ContextWindow& context, ContextWindow& current, SentenceStructure& sentenceStruct);
    
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


extern SemanticCoherence semantic;
#endif
