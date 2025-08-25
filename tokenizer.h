#ifndef TOKENIZER_H
#define TOKENIZER_H

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

static void FitVocab(Vocabulary& vocab, const std::vector<std::string>& corpus_texts) {
    vocab.BuildSpecials();
    for (size_t i = 0; i < corpus_texts.size(); i++) {
        std::vector<std::string> toks = WhitespaceTokenize(corpus_texts[i]);
        for (size_t j = 0; j < toks.size(); j++) (void)vocab.Add(toks[j]);
    }
}

static std::vector<int> Encode(const Vocabulary& vocab, const std::string& text, bool add_bos, bool add_eos) {
    std::vector<int> ids;
    if (add_bos) ids.push_back(vocab.bos_id);
    std::vector<std::string> toks = WhitespaceTokenize(text);
    for (size_t i = 0; i < toks.size(); i++) ids.push_back(vocab.Get(toks[i]));
    if (add_eos) ids.push_back(vocab.eos_id);
    return ids;
}

static std::string Decode(const Vocabulary& vocab, const std::vector<int>& ids) {
    std::ostringstream oss;
    for (size_t i = 0; i < ids.size(); i++) {
        int id = ids[i];
        if (id < 0 || id >= (int)vocab.id_to_word.size()) continue;
        if (i > 0) oss << " ";
        oss << vocab.id_to_word[id];
    }
    return oss.str();
}

#endif
