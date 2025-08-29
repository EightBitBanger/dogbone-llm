#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>

struct Vocabulary {
    std::unordered_map<std::string, int> word_to_id;
    std::vector<std::string> id_to_word;
    int pad_id;
    int unk_id;
    int bos_id;
    int eos_id;

    Vocabulary() : pad_id(-1), unk_id(-1), bos_id(-1), eos_id(-1) {}

    int Add(const std::string& w) {
        std::unordered_map<std::string, int>::const_iterator it = word_to_id.find(w);
        if (it != word_to_id.end()) return it->second;
        int id = (int)id_to_word.size();
        word_to_id[w] = id;
        id_to_word.push_back(w);
        return id;
    }
    int Get(const std::string& w) const {
        std::unordered_map<std::string, int>::const_iterator it = word_to_id.find(w);
        if (it != word_to_id.end()) return it->second;
        return unk_id;
    }
    int Size() const { return (int)id_to_word.size(); }

    void BuildSpecials() {
        pad_id = Add("<PAD>");
        unk_id = Add("<UNK>");
        bos_id = Add("<BOS>");
        eos_id = Add("<EOS>");
    }
};

static std::string ToLower(const std::string& s) {
    std::string out = s;
    for (size_t i = 0; i < out.size(); i++) {
        char c = out[i];
        if (c >= 'A' && c <= 'Z') out[i] = (char)(c - 'A' + 'a');
    }
    return out;
}

static std::vector<std::string> WhitespaceTokenize(const std::string& text) {
    std::vector<std::string> toks;
    std::istringstream iss(ToLower(text));
    for (std::string tok; iss >> tok; ) {
        toks.push_back(tok);
    }
    return toks;
}

// Sort all non-special tokens alphabetically and reindex IDs.
// Specials (pad, unk, bos, eos) are placed at the front in that order.
static void SortVocabAlphabetically(Vocabulary& vocab) {
    const std::vector<int> special_ids = { vocab.pad_id, vocab.unk_id, vocab.bos_id, vocab.eos_id };
    std::vector<std::string> special_tokens; special_tokens.reserve(4);
    for (int sid : special_ids) {
        if (sid >= 0 && sid < (int)vocab.id_to_word.size()) {
            special_tokens.push_back(vocab.id_to_word[(size_t)sid]);
        } else {
            static const char* defs[4] = {"<PAD>","<UNK>","<BOS>","<EOS>"};
            if ((int)special_tokens.size() < 4) special_tokens.push_back(defs[special_tokens.size()]);
        }
    }
    std::vector<std::string> normals;
    normals.reserve(vocab.id_to_word.size());
    for (size_t i=0;i<vocab.id_to_word.size();++i) {
        bool is_spec = false;
        for (int sid : special_ids) if ((int)i == sid) { is_spec = true; break; }
        if (!is_spec) normals.push_back(vocab.id_to_word[i]);
    }
    std::sort(normals.begin(), normals.end());
    vocab.id_to_word.clear();
    vocab.id_to_word.push_back(special_tokens.size()>0?special_tokens[0]:"<PAD>");
    vocab.id_to_word.push_back(special_tokens.size()>1?special_tokens[1]:"<UNK>");
    vocab.id_to_word.push_back(special_tokens.size()>2?special_tokens[2]:"<BOS>");
    vocab.id_to_word.push_back(special_tokens.size()>3?special_tokens[3]:"<EOS>");
    vocab.pad_id = 0; vocab.unk_id = 1; vocab.bos_id = 2; vocab.eos_id = 3;
    vocab.id_to_word.insert(vocab.id_to_word.end(), normals.begin(), normals.end());
    vocab.word_to_id.clear();
    for (size_t i=0;i<vocab.id_to_word.size();++i) vocab.word_to_id[vocab.id_to_word[i]] = (int)i;
}

static void FitVocab(Vocabulary& vocab, const std::vector<std::string>& corpus_texts) {
    vocab.BuildSpecials();
    for (size_t i = 0; i < corpus_texts.size(); i++) {
        std::vector<std::string> toks = WhitespaceTokenize(corpus_texts[i]);
        for (size_t j = 0; j < toks.size(); j++) (void)vocab.Add(toks[j]);
    }
    // Finalize: sort tokens alphabetically and reindex IDs
    SortVocabAlphabetically(vocab);
}

static std::vector<int> Encode(const Vocabulary& vocab, const std::string& text, bool add_bos, bool add_eos) {
    std::vector<int> ids;
    if (add_bos) ids.push_back(vocab.bos_id);
    std::vector<std::string> toks = WhitespaceTokenize(text);
    for (size_t i = 0; i < toks.size(); i++) ids.push_back(vocab.Get(toks[i]));
    if (add_eos) ids.push_back(vocab.eos_id);
    return ids;
}

#endif
