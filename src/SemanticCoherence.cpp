#include <iostream>
#include "SemanticCoherence.h"

SemanticCoherence semantic;



void SemanticCoherence::EmitToken(std::string word) {
    
    std::cout << word;
}



bool SemanticCoherence::ProcessTokenStream(TransformerLauguageModel& model,
                                           Vocabulary& vocab,
                                           SamplingParams& sampler,
                                           ContextWindow& context,
                                           ContextWindow& current,
                                           SentenceStructure& sentenceStruct) {
    // Candidate shortlist
    const int   kMaxCandidates = vocab.Size();
    const float kMinProb       = 0.0001f;
    const bool  kRenormalize   = true;
    
    // Find sentence starting point
    const std::vector<Token>& ctx_ids = context.GetContext();
    bool at_sentence_start = ctx_ids.empty();
    if (!at_sentence_start) {
        Token lastId = ctx_ids.back();
        std::string lastWord = vocab.id_to_word[lastId];
        at_sentence_start = semantic.is_end_punct(lastWord);
    }
    
    // Sample next token
    std::vector<TokenCandidate> candidate =
        Sampler.GetProbableTokens(model, context.GetContext(), sampler, kMaxCandidates, kMinProb, kRenormalize);
    
    Token token = candidate[0].id;
    std::string word = vocab.id_to_word[token];
    
    // Avoid non word starting words
    if (current.Size() < 3 && !semantic.is_wordish(word)) {
        
        // Check candidates
        for (unsigned int c=0; c < candidate.size(); c++) {
            token = candidate[c].id;
            word = vocab.id_to_word[token];
            
            if (semantic.is_wordish(word)) 
                break;
        }
    }
    
    // Capitalize first word in stream
    if (current.Size() == 0) 
        at_sentence_start = true;
    
    // Capitalize first word in sentence
    if (at_sentence_start && semantic.is_wordish(word)) {
        word = semantic.capitalize(word);
    }
    
    // Pull punctuation tight (esp. '.'), otherwise add a leading space
    const bool is_tight_punct = semantic.is_end_punct(word) || semantic.is_plain_punct(word) || semantic.is_closing_bracket(word);
    std::string out = word;
    if (!ctx_ids.empty() && !is_tight_punct) {
        out.insert(0, " ");
    }
    // Emit
    EmitToken(out);
    
    // Update contexts
    context.Add(token);
    current.Add(token);
    
    // If we just closed a sentence, bump the counter and possibly stop
    if (semantic.is_end_punct(word)) {
        sentenceStruct.sentenceCounter++;
        current.Clear(); // optional: reset per-sentence buffer
        
        if (sentenceStruct.sentenceCounter >= sentenceStruct.sentenceCountMax) {
            // Completed the desired number of sentences — stop generation
            return false;
        }
    }
    
    // Keep generating
    return true;
    
    
    
    
    
    
    
    
    
    /*
    
    
    // Avoid starting with end punctuation
    for (unsigned int i = 0; i < 4; ++i) {
        if (!semantic.is_end_punct(word)) break;
        candidate = Sampler.GetProbableTokens(model, context.GetContext(), sampler, kMaxCandidates, kMinProb, kRenormalize);
        if (candidate.empty()) break;
        token = candidate[0].id;
        word  = vocab.id_to_word[token];
    }
    
    // Determine start of a sentence
    bool atSentenceStart = false;
    const unsigned int contextSize = context.Size();
    if (contextSize == 0) {
        atSentenceStart = true;
    } else {
        std::string lastWord = vocab.id_to_word[context[contextSize - 1]];
        if (semantic.is_end_punct(lastWord)) {
            atSentenceStart = true;
        } else if ((semantic.is_quote(lastWord) || semantic.is_closing_bracket(lastWord)) && contextSize >= 2) {
            std::string prevWord = vocab.id_to_word[context[contextSize - 2]];
            if (semantic.is_end_punct(prevWord)) atSentenceStart = true;
        }
    }
    
    // Capitalized sentence start
    if (atSentenceStart && semantic.is_wordish(word)) {
        sentenceStruct.sentenceCounter++;
        
        // Build capitalized surface form
        std::string capWord = word;
        capWord = semantic.capitalize(capWord);  // capitalizes first character in-place
        
        // If vocab contains a token for the capitalized form, and it’s in the shortlist, use it.
        int capId = -1;
        
        std::string printed = word;
        printed = semantic.capitalize(printed);
        
        if (semantic.is_end_punct(printed)) {
            std::cout << printed;
        } else {
            std::cout << " " << printed;
        }
        
    }
    
    // Spacing
    if (semantic.is_end_punct(word)) {
        std::cout << word;
    } else {
        std::cout << " " << word;
    }
    
    // Word counter
    sentenceStruct.wordsCounter++;
    if (sentenceStruct.wordsCounter > sentenceStruct.wordsPerSentence) {
        sentenceStruct.wordsCounter=0;
        context.Add(token);
        return true;
    }
    
    // Sentence counter
    if (sentenceStruct.sentenceCounter > sentenceStruct.sentenceCountMax) {
        sentenceStruct.sentenceCounter=0;
        sentenceStruct.wordsCounter=0;
        context.Add(token);
        return false;
    }
    
    context.Add(token);
    return true;
    */
    
    
    
    
}


inline std::string SemanticCoherence::lower(std::string& s) {
    std::string out; 
    out.reserve(s.size());
    for (char c : s) {
        out.push_back((char)std::tolower((unsigned char)c));
    }
    return out;
}

inline std::string SemanticCoherence::capitalize(std::string& s) {
    std::string out;
    out.reserve(s.size());

    bool first_alpha_done = false;
    for (size_t i = 0, n = s.size(); i < n; ++i) {
        unsigned char uc = (unsigned char)s[i];

        if (!first_alpha_done && std::isalpha(uc)) {
            out.push_back((char)std::toupper(uc));
            first_alpha_done = true;
        } else if (first_alpha_done && std::isalpha(uc)) {
            out.push_back((char)std::tolower(uc));
        } else {
            out.push_back((char)uc);
        }
    }
    return out;
}

inline bool SemanticCoherence::is_alpha(std::string& s) {
    if (s.empty()) return false;
    for (char c : s) {
        if (!std::isalpha((unsigned char)c)) return false;
    }
    return true;
}

inline bool SemanticCoherence::is_capitalized(std::string& s) {
    return !s.empty() && std::isupper((unsigned char)s[0]);
}

inline bool SemanticCoherence::is_punct(std::string& s) {
    static const std::unordered_set<std::string> P = {
        ".", ",", ";", ":", "!", "?", "\"", "'", "(", ")", "[", "]", "{", "}", "-", "—", "…"
    };
    return P.count(s) > 0;
}

inline bool SemanticCoherence::is_plain_punct(std::string& s) {
    // Punctuation that usually doesn't start a sentence
    static const std::unordered_set<std::string> P = {",", ";", ":", ")", "]", "}", "-", "—"};
    return P.count(s) > 0;
}

inline bool SemanticCoherence::is_end_punct(std::string& s) {
    static const std::unordered_set<std::string> P = {".", "!", "?"};
    return P.count(s) > 0;
}

inline bool SemanticCoherence::starts_with_vowel_sound(std::string& s) {
    // Heuristic: a/an rule (cheap)
    if (s.empty()) return false;
    char c = (char)std::tolower((unsigned char)s[0]);
    return (c=='a'||c=='e'||c=='i'||c=='o'||c=='u');
}

inline bool SemanticCoherence::is_article(std::string& w) {
    static const std::unordered_set<std::string> S = {"a","an","the"};
    return S.count(w) > 0;
}

inline bool SemanticCoherence::is_preposition(std::string& w) {
    static const std::unordered_set<std::string> S = {
        "of","to","in","for","on","with","at","by","from","about","as","into",
        "like","through","after","over","between","out","against","during",
        "without","before","under","around","among"
    };
    return S.count(w) > 0;
}

inline bool SemanticCoherence::is_conjunction(std::string& w) {
    static const std::unordered_set<std::string> S = {
        "and","but","or","nor","for","so","yet", "because","although","though",
        "since","while", "if","unless","whereas"
    };
    return S.count(w) > 0;
}

inline bool SemanticCoherence::is_pronoun(std::string& w) {
    static const std::unordered_set<std::string> S = {
        "i","you","he","she","it","we","they","me","him","her","us","them","this",
        "that","these","those"
    };
    return S.count(w) > 0;
}

inline bool SemanticCoherence::is_aux_verb(std::string& w) {
    static const std::unordered_set<std::string> S = {
        "am","is","are","was","were","be","been","being",
        "do","does","did",
        "have","has","had",
        "will","would","can","could","should","may","might","must","shall"
    };
    return S.count(w) > 0;
}

inline bool SemanticCoherence::is_wordish(std::string& t) {
    if (t.empty()) return false;

    bool has_alpha = false;
    for (size_t i = 0; i < t.size(); ++i) {
        unsigned char uc = (unsigned char)t[i];
        if (std::isalpha(uc)) { has_alpha = true; continue; }
        if (std::isdigit(uc)) { continue; }
        if (uc == '\'' || uc == '-' || uc == '_') { continue; }
        // Anything else (., !, ?, quotes, brackets, emoji, etc.) => not wordish
        return false;
    }
    return has_alpha;
}

// Quote tokens (handles ASCII, PTB-style ``/'' and common Unicode quotes)
inline bool SemanticCoherence::is_quote(std::string& t) {
    static const std::unordered_set<std::string> Q = {
        "\"", "'", "``", "''", "`",
        "“", "”", "‘", "’",
        "«", "»"
    };
    return Q.count(t) > 0;
}

// Closing brackets you'd attach punctuation to
inline bool SemanticCoherence::is_closing_bracket(std::string& t) {
    static const std::unordered_set<std::string> C = {
        ")", "]", "}", "»"
    };
    return C.count(t) > 0;
}

inline bool SemanticCoherence::appeared_recently(std::vector<std::string>& ctx, std::string& nextLower, int lastN) {
    int count = 0;
    for (int i = (int)ctx.size() - 1; i >= 0 && count < lastN; --i, ++count) {
        if (lower(ctx[(size_t)i]) == nextLower) return true;
    }
    return false;
}

inline int SemanticCoherence::count_unclosed(std::vector<std::string>& ctx, std::string& opener, std::string& closer) {
    int open = 0;
    for (const auto& w : ctx) {
        if (w == opener) ++open;
        if (w == closer && open > 0) --open;
    }
    return open;
}

inline int SemanticCoherence::count_unclosed_pair(std::vector<std::string>& ctx, std::string& opener, std::string& closer) {
    int open = 0;
    for (const auto& w : ctx) {
        if (w == opener) ++open;
        else if (w == closer && open > 0) --open;
    }
    return open;
}

inline bool SemanticCoherence::is_special(int id, Vocabulary& vocab) {
    return id == vocab.eos_id || id == vocab.bos_id ||
           id == vocab.pad_id || id == vocab.unk_id;
}

inline int SemanticCoherence::sanitize_special(int id, Vocabulary& vocab) {
    if (is_special(id, vocab)) return vocab.pad_id;
    return id;
}

inline std::string SemanticCoherence::safe_word(Vocabulary& vocab, int id) {
    if (id < 0 || id >= (int)vocab.id_to_word.size()) return "";
    return vocab.id_to_word[(size_t)id];
}
