#include "TokenRasterizer.h"
//TokenRasterizer rasterizer;

/*
inline bool TokenRasterizer::is_sentence_end(const std::string& t) {
    return t == "." || t == "!" || t == "?";
}
inline bool TokenRasterizer::is_closing_punct(const std::string& t) {
    return t == "." || t == "," || t == ";" || t == ":" ||
           t == "!" || t == "?" || t == ")" || t == "]" ||
           t == "}" || t == "…";
}
inline bool TokenRasterizer::is_opening_bracket(const std::string& t) {
    return t == "(" || t == "[" || t == "{";
}
bool TokenRasterizer::is_quote(const std::string& t) {
    return t == "\"";
}
inline bool TokenRasterizer::is_wordish(const std::string& t) {
    if (t.empty()) return false;
    for (unsigned char c : t) if (std::isalnum(c)) return true;
    return false;
}
inline std::string TokenRasterizer::capitalize_first(std::string s) {
    for (char& c : s) {
        if (std::isalpha((unsigned char)c)) { c = (char)std::toupper((unsigned char)c); break; }
    }
    return s;
}

void TokenRasterizer::emit(const std::string& t) {
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
        word_count_sentence = 0;              // <-- reset per-sentence count
    } else if (is_wordish(t)) {
        word_count_total    += 1;
        word_count_sentence += 1;             // <-- bump per-sentence count
        at_sentence_start    = false;
    }

    if (is_quote(t)) in_quotes = !in_quotes;

    last = t;
    at_start = false;
}

void TokenRasterizer::reset() {
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
bool should_stop_overall(size_t max_sentences) const {
    return (max_sentences > 0 && sentence_count >= max_sentences);
}
bool sentence_word_cap_reached(size_t max_words_per_sentence) const {
    return (max_words_per_sentence > 0 && word_count_sentence >= max_words_per_sentence);
}
*/
