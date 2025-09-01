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

void SortVocabAlphabetically(Tokenizer& vocab) {
    const std::vector<int> special_ids = {
        vocab.token.pad_id, vocab.token.unk_id, vocab.token.bos_id, vocab.token.eos_id, vocab.token.query_id, vocab.token.response_id
    };
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
    vocab.id_to_word.push_back(special_tokens.size()>4?special_tokens[4]:"<QUERY>");
    vocab.id_to_word.push_back(special_tokens.size()>5?special_tokens[5]:"<RESPONSE>");
    
    vocab.token.pad_id = 0;
    vocab.token.unk_id = 1;
    vocab.token.bos_id = 2;
    vocab.token.eos_id = 3;
    vocab.token.query_id = 4;
    vocab.token.response_id = 5;
    
    vocab.id_to_word.insert(vocab.id_to_word.end(), normals.begin(), normals.end());
    vocab.word_to_id.clear();
    for (size_t i=0;i<vocab.id_to_word.size();++i) vocab.word_to_id[vocab.id_to_word[i]] = (int)i;
}

void FitVocab(Tokenizer& vocab, const std::vector<std::string>& corpus_texts) {
    vocab.BuildSpecials();
    for (size_t i = 0; i < corpus_texts.size(); i++) {
        std::vector<std::string> toks = WhitespaceTokenize(corpus_texts[i]);
        for (size_t j = 0; j < toks.size(); j++) (void)vocab.Add(toks[j]);
    }
    // Finalize: sort tokens alphabetically and reindex IDs
    SortVocabAlphabetically(vocab);
}

std::vector<int> Encode(const Tokenizer& vocab, const std::string& text, bool add_bos, bool add_eos) {
    std::vector<int> ids;
    std::vector<std::string> toks = WhitespaceTokenize(text);
    for (size_t i = 0; i < toks.size(); i++) 
        ids.push_back(vocab.Get(toks[i]));
    return ids;
}
