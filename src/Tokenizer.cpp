#include "tokenizer.h"

Tokenizer::SpecialTokens::SpecialTokens() :
    pad_id(-1),
    unk_id(-1),
    bos_id(-1),
    eos_id(-1),
    query_id(-1),
    response_id(-1) {}

int Tokenizer::Add(const std::string& w) {
    std::unordered_map<std::string, int>::const_iterator it = word_to_id.find(w);
    if (it != word_to_id.end()) return it->second;
    int id = (int)id_to_word.size();
    word_to_id[w] = id;
    id_to_word.push_back(w);
    return id;
}

int Tokenizer::Get(const std::string& w) const {
    std::unordered_map<std::string, int>::const_iterator it = word_to_id.find(w);
    if (it != word_to_id.end()) return it->second;
    return token.unk_id;
}

unsigned int Tokenizer::Size() const {
    return id_to_word.size();
}

void Tokenizer::Clear() {
    word_to_id.clear();
    id_to_word.clear();
}

void Tokenizer::Reserve(size_t amount) {
    id_to_word.reserve(amount);
}

// --- operator[] overloads ---
int Tokenizer::operator[](const std::string& w) const {
    // Word -> id (no insertion; ensure we never create a non-existing token)
    return Get(w);
}

const std::string& Tokenizer::operator[](int id) const {
    if (id >= 0 && id < (int)id_to_word.size()) {
        return id_to_word[(size_t)id];
    }
    if (token.unk_id >= 0 && token.unk_id < (int)id_to_word.size()) {
        return id_to_word[(size_t)token.unk_id];
    }
    static const std::string kUNK = "<UNK>";
    return kUNK;
}

void Tokenizer::BuildSpecials() {
    if (token.pad_id != -1)
        return;
    token.pad_id = Add("<PAD>");
    token.unk_id = Add("<UNK>");
    token.bos_id = Add("<BOS>");
    token.eos_id = Add("<EOS>");
    token.query_id    = Add("<QURY>"); // User query
    token.response_id = Add("<RESP>"); // Assistant response
}

std::string ToLower(const std::string& s) {
    std::string out = s;
    for (size_t i = 0; i < out.size(); i++) {
        char c = out[i];
        if (c >= 'A' && c <= 'Z') out[i] = (char)(c - 'A' + 'a');
    }
    return out;
}

std::vector<std::string> WhitespaceTokenize(const std::string& text) {
    std::vector<std::string> toks;
    std::istringstream iss(ToLower(text));
    for (std::string tok; iss >> tok; ) {
        toks.push_back(tok);
    }
    return toks;
}

void Tokenizer::SortVocabAlphabetically() {
    const std::vector<int> special_ids = {
        token.pad_id, token.unk_id, token.bos_id, token.eos_id, token.query_id, token.response_id
    };
    std::vector<std::string> special_tokens; special_tokens.reserve(4);
    for (int sid : special_ids) {
        if (sid >= 0 && sid < (int)id_to_word.size()) {
            special_tokens.push_back(id_to_word[(size_t)sid]);
        } else {
            static const char* defs[4] = {"<PAD>","<UNK>","<BOS>","<EOS>"};
            if ((int)special_tokens.size() < 4) special_tokens.push_back(defs[special_tokens.size()]);
        }
    }
    std::vector<std::string> normals;
    normals.reserve(id_to_word.size());
    for (size_t i=0;i<id_to_word.size();++i) {
        bool is_spec = false;
        for (int sid : special_ids) if ((int)i == sid) { is_spec = true; break; }
        if (!is_spec) normals.push_back(id_to_word[i]);
    }
    std::sort(normals.begin(), normals.end());
    id_to_word.clear();
    id_to_word.push_back(special_tokens.size()>0?special_tokens[0]:"<PAD>");
    id_to_word.push_back(special_tokens.size()>1?special_tokens[1]:"<UNK>");
    id_to_word.push_back(special_tokens.size()>2?special_tokens[2]:"<BOS>");
    id_to_word.push_back(special_tokens.size()>3?special_tokens[3]:"<EOS>");
    id_to_word.push_back(special_tokens.size()>4?special_tokens[4]:"<QUERY>");
    id_to_word.push_back(special_tokens.size()>5?special_tokens[5]:"<RESPONSE>");

    token.pad_id = 0;
    token.unk_id = 1;
    token.bos_id = 2;
    token.eos_id = 3;
    token.query_id = 4;
    token.response_id = 5;

    id_to_word.insert(id_to_word.end(), normals.begin(), normals.end());
    word_to_id.clear();
    for (size_t i=0;i<id_to_word.size();++i) word_to_id[id_to_word[i]] = (int)i;
}

void Tokenizer::FitToCorpus(Tokenizer& vocab, const std::vector<std::string>& corpus_texts) {
    vocab.BuildSpecials();
    for (size_t i = 0; i < corpus_texts.size(); i++) {
        std::vector<std::string> toks = WhitespaceTokenize(corpus_texts[i]);
        for (size_t j = 0; j < toks.size(); j++) (void)vocab.Add(toks[j]);
    }
    vocab.SortVocabAlphabetically();
}

std::vector<int> Encode(const Tokenizer& vocab, const std::string& text) {
    std::vector<int> ids;
    std::vector<std::string> toks = WhitespaceTokenize(text);
    ids.reserve(toks.size());
    for (const std::string& t : toks) {
        ids.push_back(vocab[t]);
    }
    return ids;
}
