#ifndef TOKEN_RASTERIZER_H
#define TOKEN_RASTERIZER_H

#include <iostream>
#include <string>

struct TokenRasterizer {
    bool at_start           = true;   // first token overall
    bool at_sentence_start  = true;   // first token after . ! ?
    bool in_quotes          = false;  // track " state
    std::string last;

    // Counters
    size_t word_count_total     = 0;  // optional: total words (unchanged)
    size_t sentence_count       = 0;  // total sentences completed
    size_t word_count_sentence  = 0;  // <-- words in the current sentence

    // ---- helpers ----
    static inline bool is_sentence_end(const std::string& t) {
        return t == "." || t == "!" || t == "?";
    }
    static inline bool is_closing_punct(const std::string& t) {
        return t == "." || t == "," || t == ";" || t == ":" ||
               t == "!" || t == "?" || t == ")" || t == "]" ||
               t == "}" || t == "…";
    }
    static inline bool is_opening_bracket(const std::string& t) {
        return t == "(" || t == "[" || t == "{";
    }
    static inline bool is_quote(const std::string& t) {
        return t == "\"";
    }
    static inline bool is_wordish(const std::string& t) {
        if (t.empty()) return false;
        for (unsigned char c : t) if (std::isalnum(c)) return true;
        return false;
    }
    static inline std::string capitalize_first(std::string s) {
        for (char& c : s) {
            if (std::isalpha((unsigned char)c)) { c = (char)std::toupper((unsigned char)c); break; }
        }
        return s;
    }

    void emit(const std::string& t) {
        if (t.empty()) return;

        bool addSpace = false;
        if (!at_start) {
            if (is_quote(t))                addSpace = !in_quotes;     // opening vs closing
            else if (is_closing_punct(t))   addSpace = false;          // attach
            else if (is_opening_bracket(last) || is_quote(last)) addSpace = false;
            else                           addSpace = true;
        }

        std::string out = t;
        if (at_sentence_start && is_wordish(out)) out = capitalize_first(out);

        if (addSpace) std::cout << ' ';
        std::cout << out;

        // counters & sentence state
        if (is_sentence_end(t)) {
            sentence_count += 1;
            at_sentence_start  = true;
            word_count_sentence = 0;
        } else if (is_wordish(t)) {
            word_count_total    += 1;
            word_count_sentence += 1;
            at_sentence_start    = false;
        }

        if (is_quote(t)) in_quotes = !in_quotes;

        last = t;
        at_start = false;
    }

    void reset() {
        at_start = true;
        at_sentence_start = true;
        in_quotes = false;
        last.clear();
    }
    void reset_counts() {
        word_count_total = 0;
        sentence_count = 0;
        word_count_sentence = 0;
    }

    // Overall stop (sentences). Word cap is per-sentence, handled outside.
    bool should_stop_overall(size_t max_sentences) const;
    bool sentence_word_cap_reached(size_t max_words_per_sentence) const;
};

extern TokenRasterizer rasterizer;

#endif
