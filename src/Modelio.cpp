#include "Modelio.h"

//#include "Transformer/Transformer.h"
//#include "tokenizer.h"

inline bool StreamWriteLE32(std::ostream& out, uint32_t v) {
    unsigned char b[4];
    b[0] = (unsigned char)(v & 0xFFu);
    b[1] = (unsigned char)((v >> 8) & 0xFFu);
    b[2] = (unsigned char)((v >> 16) & 0xFFu);
    b[3] = (unsigned char)((v >> 24) & 0xFFu);
    out.write(reinterpret_cast<const char*>(b), 4);
    return out.good();
}

inline bool StreamReadLE32(std::istream& in, uint32_t& v) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in.good()) return false;
    v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return true;
}

inline bool WriteModelDims(std::ostream& out, const ModelDims& d) {
    return  StreamWriteLE32(out, (uint32_t)d.vocab_size) &&
            StreamWriteLE32(out, (uint32_t)d.d_model)     &&
            StreamWriteLE32(out, (uint32_t)d.n_heads)     &&
            StreamWriteLE32(out, (uint32_t)d.d_ff)        &&
            StreamWriteLE32(out, (uint32_t)d.n_layers)    &&
            StreamWriteLE32(out, (uint32_t)d.max_T);
}

inline bool ReadModelDims(std::istream& in, ModelDims& d) {
    uint32_t v=0;
    if (!StreamReadLE32(in, v)) {return false;} d.vocab_size = (int32_t)v;
    if (!StreamReadLE32(in, v)) {return false;} d.d_model    = (int32_t)v;
    if (!StreamReadLE32(in, v)) {return false;} d.n_heads    = (int32_t)v;
    if (!StreamReadLE32(in, v)) {return false;} d.d_ff       = (int32_t)v;
    if (!StreamReadLE32(in, v)) {return false;} d.n_layers   = (int32_t)v;
    if (!StreamReadLE32(in, v)) {return false;} d.max_T      = (int32_t)v;
    return true;
}

inline bool StreamWriteLE64(std::ostream& out, uint64_t v) {
    unsigned char b[8];
    b[0] = (unsigned char)(v & 0xFFu);
    b[1] = (unsigned char)((v >> 8) & 0xFFu);
    b[2] = (unsigned char)((v >> 16) & 0xFFu);
    b[3] = (unsigned char)((v >> 24) & 0xFFu);
    b[4] = (unsigned char)((v >> 32) & 0xFFu);
    b[5] = (unsigned char)((v >> 40) & 0xFFu);
    b[6] = (unsigned char)((v >> 48) & 0xFFu);
    b[7] = (unsigned char)((v >> 56) & 0xFFu);
    out.write(reinterpret_cast<const char*>(b), 8);
    return out.good();
}

inline bool StreamReadLE64(std::istream& in, uint64_t& v) {
    unsigned char b[8];
    in.read(reinterpret_cast<char*>(b), 8);
    if (!in.good()) return false;
    v =  (uint64_t)b[0]
       | ((uint64_t)b[1] << 8)
       | ((uint64_t)b[2] << 16)
       | ((uint64_t)b[3] << 24)
       | ((uint64_t)b[4] << 32)
       | ((uint64_t)b[5] << 40)
       | ((uint64_t)b[6] << 48)
       | ((uint64_t)b[7] << 56);
    return true;
}

bool SaveVocabBinaryToStream(const Tokenizer& vocab, std::ostream& out) {
    const char magic[4] = { 'V','O','C','B' };
    out.write(magic, 4);
    if (!out.good()) return false;

    const uint32_t version = 1u;
    if (!StreamWriteLE32(out, version)) return false;

    const uint32_t count = (uint32_t)vocab.Size();
    if (!StreamWriteLE32(out, count)) return false;

    if (!StreamWriteLE32(out, (uint32_t)vocab.token.pad_id)) return false;
    if (!StreamWriteLE32(out, (uint32_t)vocab.token.unk_id)) return false;
    if (!StreamWriteLE32(out, (uint32_t)vocab.token.bos_id)) return false;
    if (!StreamWriteLE32(out, (uint32_t)vocab.token.eos_id)) return false;
    if (!StreamWriteLE32(out, (uint32_t)vocab.token.query_id)) return false;
    if (!StreamWriteLE32(out, (uint32_t)vocab.token.response_id)) return false;

    for (size_t i = 0; i < vocab.Size(); ++i) {
        const std::string& tok = vocab[i];
        const uint32_t len = (uint32_t)tok.size();
        if (!StreamWriteLE32(out, len)) return false;
        if (len > 0) {
            out.write(tok.data(), (std::streamsize)len);
            if (!out.good()) return false;
        }
    }
    return out.good();
}

bool LoadVocabBinaryFromStream(Tokenizer& vocab, std::istream& in) {
    char magic[4] = {0,0,0,0};
    in.read(magic, 4);
    if (!in.good()) return false;
    if (!(magic[0]=='V' && magic[1]=='O' && magic[2]=='C' && magic[3]=='B')) return false;

    uint32_t version = 0u;
    if (!StreamReadLE32(in, version)) return false;
    if (version != 1u) return false;

    uint32_t count = 0u;
    if (!StreamReadLE32(in, count)) return false;

    uint32_t pad_u=0, unk_u=0, bos_u=0, eos_u=0, qry_u=0, rsp_u=0;
    if (!StreamReadLE32(in, pad_u)) return false;
    if (!StreamReadLE32(in, unk_u)) return false;
    if (!StreamReadLE32(in, bos_u)) return false;
    if (!StreamReadLE32(in, eos_u)) return false;
    if (!StreamReadLE32(in, qry_u)) return false;
    if (!StreamReadLE32(in, rsp_u)) return false;

    vocab.Clear();
    vocab.Reserve(count);
    
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t len = 0u;
        if (!StreamReadLE32(in, len)) return false;
        if (len > (1024u * 1024u)) return false; // sanity

        std::string tok;
        tok.resize(len);
        if (len > 0) {
            in.read(&tok[0], (std::streamsize)len);
            if (!in.good()) return false;
        }
        vocab.id_to_word.push_back(tok);
        vocab.word_to_id[tok] = (int)i;
    }

    vocab.token.pad_id = (int32_t)pad_u;
    vocab.token.unk_id = (int32_t)unk_u;
    vocab.token.bos_id = (int32_t)bos_u;
    vocab.token.eos_id = (int32_t)eos_u;
    vocab.token.query_id    = (int32_t)qry_u;
    vocab.token.response_id = (int32_t)rsp_u;
    return true;
}

void EnsureAdamSize(AdamState& s, size_t n) {
    if (s.m.size() != n) s.m.assign(n, 0.0f);
    if (s.v.size() != n) s.v.assign(n, 0.0f);
}

bool WriteAdamState(std::ostream& out, const AdamState& s, size_t expected_len) {
    uint32_t len = (uint32_t)expected_len;
    if (!StreamWriteLE32(out, len)) return false;
    if (len) {
        out.write((const char*)s.m.data(), (std::streamsize)(sizeof(float)*len));
        out.write((const char*)s.v.data(), (std::streamsize)(sizeof(float)*len));
    }
    return out.good();
}

bool ReadAdamState(std::istream& in, AdamState& s, size_t expected_len) {
    uint32_t len = 0u;
    if (!StreamReadLE32(in, len)) return false;

    std::vector<float> mtmp((size_t)len), vtmp((size_t)len);
    if (len) {
        in.read((char*)mtmp.data(), (std::streamsize)(sizeof(float)*len));
        if (!in.good()) return false;
        in.read((char*)vtmp.data(), (std::streamsize)(sizeof(float)*len));
        if (!in.good()) return false;
    }

    EnsureAdamSize(s, expected_len);
    size_t copy = std::min((size_t)len, expected_len);
    if (copy) {
        std::memcpy(s.m.data(), mtmp.data(), sizeof(float)*copy);
        std::memcpy(s.v.data(), vtmp.data(), sizeof(float)*copy);
    }
    // pad remainder with zeros (already zeroed by EnsureAdamSize)
    return true;
}

// Save model + vocab + optimizer state into a single package.
bool SaveModelPackage(const std::string& path,
                             const LauguageModel& model,
                             const Tokenizer& vocab,
                             const NeuralNetwork& trainer,
                             uint64_t epoch,
                             float current_lr,
                             float last_avg_loss) {
    std::ofstream out(path.c_str(), std::ios::binary);
    if (!out.is_open()) return false;

    const char magic[8] = { 'M','O','D','E','L','P','K','G' };
    out.write(magic, 8);
    if (!out.good()) return false;

    const uint32_t version       = 2u;
    const uint32_t flags         = 0u;
    const uint32_t section_count = 4u; // VOCB, HPRM, MODEL, OPTS

    if (!StreamWriteLE32(out, version))       return false;
    if (!StreamWriteLE32(out, flags))         return false;
    if (!StreamWriteLE32(out, section_count)) return false;

    // reserved
    char hdr_resv[44]; std::memset(hdr_resv, 0, sizeof(hdr_resv));
    out.write(hdr_resv, (std::streamsize)sizeof(hdr_resv));
    if (!out.good()) return false;

    // directory placeholder
    const std::streampos dir_pos = out.tellp();
    ModelPkgSection zero_entry; std::memset(&zero_entry, 0, sizeof(zero_entry));
    for (uint32_t i = 0; i < section_count; ++i) {
        out.write(reinterpret_cast<const char*>(&zero_entry), (std::streamsize)sizeof(ModelPkgSection));
    }
    if (!out.good()) return false;

    std::vector<ModelPkgSection> entries(section_count);

    // ---- 1) VOCB first (put strings near the top) ----
    std::strncpy(entries[0].id, "VOCB", 4);
    entries[0].offset = (uint64_t)out.tellp();
    if (!SaveVocabBinaryToStream(vocab, out)) return false;
    entries[0].size = (uint64_t)((uint64_t)out.tellp() - entries[0].offset);

    // ---- 2) HPRM (small header summary) ----
    std::strncpy(entries[1].id, "HPRM", 4);
    entries[1].offset = (uint64_t)out.tellp();
    {
        ModelDims dims;
        dims.vocab_size = model.vocab_size;
        dims.d_model    = model.d_model;
        dims.n_heads    = model.n_heads;
        dims.d_ff       = model.d_ff;
        dims.n_layers   = model.n_layers;
        dims.max_T      = model.n_ctx;
        if (!WriteModelDims(out, dims)) return false;
    }
    entries[1].size = (uint64_t)((uint64_t)out.tellp() - entries[1].offset);

    // ---- 3) MODEL (weights) ----
    std::strncpy(entries[2].id, "MODE", 4);
    entries[2].offset = (uint64_t)out.tellp();
    if (!model.SaveToStream(out)) return false;
    entries[2].size = (uint64_t)((uint64_t)out.tellp() - entries[2].offset);

    // ---- 4) OPTS (optimizer state) ----
    std::strncpy(entries[3].id, "OPTS", 4);
    entries[3].offset = (uint64_t)out.tellp();
    {
        const uint32_t opt_version = 1u;
        if (!StreamWriteLE32(out, opt_version)) return false;
    
        // Adam hyperparams + step + epoch + avg loss
        out.write((const char*)&trainer.opt.learning_rate,    sizeof(float));
        out.write((const char*)&trainer.opt.beta_m, sizeof(float));
        out.write((const char*)&trainer.opt.beta_v, sizeof(float));
        out.write((const char*)&trainer.opt.epsilon,   sizeof(float));
        int32_t t_i32 = trainer.opt.step;
        out.write((const char*)&t_i32, sizeof(int32_t));
        out.write((const char*)&epoch, sizeof(uint64_t));
        out.write((const char*)&last_avg_loss, sizeof(float));
    
        // ---- top-level moments (write from local copies; do not const_cast) ----
        AdamState lmW  = trainer.lmW_s;
        AdamState lmb  = trainer.lmb_s;
        AdamState posP = trainer.posP_s;
        AdamState tokW = trainer.tokW_s;
    
        EnsureAdamSize(lmW,  model.lm_head.W.data.size());
        EnsureAdamSize(lmb,  model.lm_head.b.size());
        EnsureAdamSize(posP, model.pos.P.data.size());
        EnsureAdamSize(tokW, model.tok.W.data.size());
    
        if (!WriteAdamState(out, lmW,  model.lm_head.W.data.size())) return false;
        if (!WriteAdamState(out, lmb,  model.lm_head.b.size()))       return false;
        if (!WriteAdamState(out, posP, model.pos.P.data.size()))      return false;
        if (!WriteAdamState(out, tokW, model.tok.W.data.size()))      return false;
    
        // ---- per-layer moments (work on a local vector that we size safely) ----
        const size_t L = (size_t)model.n_layers;
        auto LS = trainer.layer_states;           // local copy
        if (LS.size() < L) LS.resize(L);          // ensure indexing is safe
    
        for (size_t l = 0; l < L; ++l) {
            const TransformerBlock& B = model.layers[l];
            auto &S = LS[l];                      // local/mutable
    
            EnsureAdamSize(S.WoW_s,  B.attn.Wo.W.data.size());
            EnsureAdamSize(S.Wob_s,  B.attn.Wo.b.size());
            EnsureAdamSize(S.WqW_s,  B.attn.Wq.W.data.size());
            EnsureAdamSize(S.Wqb_s,  B.attn.Wq.b.size());
            EnsureAdamSize(S.WkW_s,  B.attn.Wk.W.data.size());
            EnsureAdamSize(S.Wkb_s,  B.attn.Wk.b.size());
            EnsureAdamSize(S.WvW_s,  B.attn.Wv.W.data.size());
            EnsureAdamSize(S.Wvb_s,  B.attn.Wv.b.size());
            EnsureAdamSize(S.fc1W_s, B.ffn.fc1.W.data.size());
            EnsureAdamSize(S.fc1b_s, B.ffn.fc1.b.size());
            EnsureAdamSize(S.fc2W_s, B.ffn.fc2.W.data.size());
            EnsureAdamSize(S.fc2b_s, B.ffn.fc2.b.size());
            EnsureAdamSize(S.ln1g_s, B.ln1.gamma.size());
            EnsureAdamSize(S.ln1b_s, B.ln1.beta.size());
            EnsureAdamSize(S.ln2g_s, B.ln2.gamma.size());
            EnsureAdamSize(S.ln2b_s, B.ln2.beta.size());
    
            if (!WriteAdamState(out, S.WoW_s,  B.attn.Wo.W.data.size())) return false;
            if (!WriteAdamState(out, S.Wob_s,  B.attn.Wo.b.size()))       return false;
            if (!WriteAdamState(out, S.WqW_s,  B.attn.Wq.W.data.size()))  return false;
            if (!WriteAdamState(out, S.Wqb_s,  B.attn.Wq.b.size()))       return false;
            if (!WriteAdamState(out, S.WkW_s,  B.attn.Wk.W.data.size()))  return false;
            if (!WriteAdamState(out, S.Wkb_s,  B.attn.Wk.b.size()))       return false;
            if (!WriteAdamState(out, S.WvW_s,  B.attn.Wv.W.data.size()))  return false;
            if (!WriteAdamState(out, S.Wvb_s,  B.attn.Wv.b.size()))       return false;
            if (!WriteAdamState(out, S.fc1W_s, B.ffn.fc1.W.data.size()))  return false;
            if (!WriteAdamState(out, S.fc1b_s, B.ffn.fc1.b.size()))       return false;
            if (!WriteAdamState(out, S.fc2W_s, B.ffn.fc2.W.data.size()))  return false;
            if (!WriteAdamState(out, S.fc2b_s, B.ffn.fc2.b.size()))       return false;
            if (!WriteAdamState(out, S.ln1g_s, B.ln1.gamma.size()))       return false;
            if (!WriteAdamState(out, S.ln1b_s, B.ln1.beta.size()))        return false;
            if (!WriteAdamState(out, S.ln2g_s, B.ln2.gamma.size()))       return false;
            if (!WriteAdamState(out, S.ln2b_s, B.ln2.beta.size()))        return false;
        }
    }
    entries[3].size = (uint64_t)((uint64_t)out.tellp() - entries[3].offset);

    // ---- patch directory ----
    out.seekp(dir_pos, std::ios::beg);
    for (uint32_t i = 0; i < section_count; ++i) {
        out.write(reinterpret_cast<const char*>(&entries[i]), (std::streamsize)sizeof(ModelPkgSection));
    }
    out.flush();
    return out.good();
}

// Load model + vocab + optimizer if present.
// Returns true on success. If OPTS missing, model+vocab still loaded and trainer remains default.
bool LoadModelPackage(const std::string& path, LauguageModel& model, Tokenizer& vocab, NeuralNetwork& trainer,
                             uint64_t& epoch_out, float& current_lr_out, float& last_loss_out) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in.is_open()) return false;

    // header
    char magic[8]={0};
    in.read(magic,8);
    if (!in.good()) return false;
    if (!(magic[0]=='M'&&magic[1]=='O'&&magic[2]=='D'&&magic[3]=='E'&&magic[4]=='L'&&magic[5]=='P'&&magic[6]=='K'&&magic[7]=='G')) return false;

    uint32_t version=0, flags=0, section_count=0;
    if (!StreamReadLE32(in, version)) return false;
    if (version != 2u) return false; // our writer uses v2
    if (!StreamReadLE32(in, flags)) return false;
    if (!StreamReadLE32(in, section_count)) return false;
    if (section_count==0 || section_count > 64) return false;

    in.seekg(44, std::ios::cur); // reserved

    std::vector<ModelPkgSection> entries(section_count);
    for (uint32_t i=0;i<section_count;++i) {
        in.read(reinterpret_cast<char*>(&entries[i]), (std::streamsize)sizeof(ModelPkgSection));
        if (!in.good()) return false;
    }

    const ModelPkgSection* secMODEL=nullptr;
    const ModelPkgSection* secVOCB =nullptr;
    const ModelPkgSection* secOPTS =nullptr;
    for (uint32_t i=0;i<section_count;++i) {
        const ModelPkgSection& e = entries[i];
        if (e.id[0]=='M'&&e.id[1]=='O'&&e.id[2]=='D'&&e.id[3]=='E') secMODEL=&e;
        else if (e.id[0]=='V'&&e.id[1]=='O'&&e.id[2]=='C'&&e.id[3]=='B') secVOCB=&e;
        else if (e.id[0]=='O'&&e.id[1]=='P'&&e.id[2]=='T'&&e.id[3]=='S') secOPTS=&e;
    }
    if (!secMODEL || !secVOCB) return false;

    // MODEL
    in.seekg((std::streamoff)secMODEL->offset, std::ios::beg);
    if (!model.LoadFromStream(in)) return false;

    // VOCB
    in.seekg((std::streamoff)secVOCB->offset, std::ios::beg);
    if (!LoadVocabBinaryFromStream(vocab, in)) return false;

    // OPTS (optional)
    epoch_out = 0ull;
    float last_loss = -1.0f;

    current_lr_out = trainer.opt.learning_rate;

    if (secOPTS) {
        in.seekg((std::streamoff)secOPTS->offset, std::ios::beg);
        uint32_t opt_ver=0;
        if (!StreamReadLE32(in, opt_ver)) return false;
        if (opt_ver != 1u) return false;

        float lr=0, b1=0, b2=0, eps=0;
        int32_t t_i32=0;
        uint64_t epoch=0;

        in.read((char*)&lr, sizeof(float));
        in.read((char*)&b1, sizeof(float));
        in.read((char*)&b2, sizeof(float));
        in.read((char*)&eps,sizeof(float));
        in.read((char*)&t_i32,sizeof(int32_t));
        in.read((char*)&epoch ,sizeof(uint64_t));
        in.read((char*)&last_loss, sizeof(float));

        if (!in.good()) return false;

        trainer.opt.learning_rate = lr;
        trainer.opt.beta_m = b1;
        trainer.opt.beta_v = b2;
        trainer.opt.epsilon = eps;
        trainer.opt.step = t_i32;
        epoch_out = epoch;
        current_lr_out = lr;
        last_loss_out = last_loss;

        // Top-level
        if (!ReadAdamState(in, trainer.lmW_s,  model.lm_head.W.data.size())) return false;
        if (!ReadAdamState(in, trainer.lmb_s,  model.lm_head.b.size()))       return false;
        if (!ReadAdamState(in, trainer.posP_s, model.pos.P.data.size()))      return false;
        if (!ReadAdamState(in, trainer.tokW_s, model.tok.W.data.size()))      return false;

        // Per-layer
        trainer.layer_states.resize((size_t)model.n_layers);
        for (int l=0; l<model.n_layers; ++l) {
            TransformerBlock& B = model.layers[(size_t)l];
            NeuralNetwork::BlockStates& S = trainer.layer_states[(size_t)l];

            if (!ReadAdamState(in, S.WoW_s,  B.attn.Wo.W.data.size())) return false;
            if (!ReadAdamState(in, S.Wob_s,  B.attn.Wo.b.size()))       return false;
            if (!ReadAdamState(in, S.WqW_s,  B.attn.Wq.W.data.size()))  return false;
            if (!ReadAdamState(in, S.Wqb_s,  B.attn.Wq.b.size()))       return false;
            if (!ReadAdamState(in, S.WkW_s,  B.attn.Wk.W.data.size()))  return false;
            if (!ReadAdamState(in, S.Wkb_s,  B.attn.Wk.b.size()))       return false;
            if (!ReadAdamState(in, S.WvW_s,  B.attn.Wv.W.data.size()))  return false;
            if (!ReadAdamState(in, S.Wvb_s,  B.attn.Wv.b.size()))       return false;
            if (!ReadAdamState(in, S.fc1W_s, B.ffn.fc1.W.data.size()))  return false;
            if (!ReadAdamState(in, S.fc1b_s, B.ffn.fc1.b.size()))       return false;
            if (!ReadAdamState(in, S.fc2W_s, B.ffn.fc2.W.data.size()))  return false;
            if (!ReadAdamState(in, S.fc2b_s, B.ffn.fc2.b.size()))       return false;
            if (!ReadAdamState(in, S.ln1g_s, B.ln1.gamma.size()))       return false;
            if (!ReadAdamState(in, S.ln1b_s, B.ln1.beta.size()))        return false;
            if (!ReadAdamState(in, S.ln2g_s, B.ln2.gamma.size()))       return false;
            if (!ReadAdamState(in, S.ln2b_s, B.ln2.beta.size()))        return false;
        }
    }

    return true;
}

// Persist-only LR update: loads package, updates trainer.opt.lr, re-saves package.
// Leaves weights, epoch, Adam moments, etc. untouched.
bool UpdateModelLROnDisk(const std::string& modelPath, float newLR,
                                uint64_t* outEpoch, float* outOldLR) {
    LauguageModel  m;
    Tokenizer     v;
    // seed trainer with something; loader will overwrite states
    NeuralNetwork    t(newLR);

    uint64_t epoch = 0;
    float oldLR = 0.0f;
    float avgLoss = 0.0f;

    if (!LoadModelPackage(modelPath, m, v, t, epoch, oldLR, avgLoss)) return false;

    t.opt.learning_rate = newLR; // change only LR
    bool ok = SaveModelPackage(modelPath, m, v, t, epoch, newLR, avgLoss);

    if (ok) {
        if (outEpoch)  *outEpoch  = epoch;
        if (outOldLR)  *outOldLR  = oldLR;
    }
    return ok;
}


bool FileTextLoad(std::string& path, std::string& out) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    
    std::ostringstream buffer;
    buffer << in.rdbuf();
    const std::string bufferString = buffer.str();
    
    out.clear();
    out.reserve(bufferString.size());
    
    for (size_t i = 0; i < bufferString.size(); ++i) {
        char c = bufferString[i];
        if (c != '\n' && c != '\r' && c != ',' && c != '`') {
            out.push_back(c);
        }
    }
    return true;
}

bool FileExists(std::string filename) {
    std::ofstream fStream(filename, std::fstream::in | std::fstream::binary);
    if (!fStream.is_open()) {
        fStream.close();
        return false;
    }
    fStream.close();
    return true;
}

bool FileDelete(std::string filename) {
    if (remove( filename.c_str() ) == 0) 
        return true;
    return false;
}
