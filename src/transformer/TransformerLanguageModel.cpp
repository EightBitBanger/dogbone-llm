#include "TransformerLauguageModel.h"

TransformerLauguageModel::TransformerLauguageModel() :
    vocab_size(0), d_model(0), n_heads(0), d_ff(0), n_layers(0), max_T(0) {}

TransformerLauguageModel::TransformerLauguageModel(int vocab, int dmodel, int heads, int ff, int layers_count, int maxT)
    : vocab_size(vocab), d_model(dmodel), n_heads(heads),
      d_ff(ff), n_layers(layers_count), max_T(maxT),
      tok(vocab, dmodel), pos(maxT, dmodel), lm_head(dmodel, vocab) {
    layers.reserve((size_t)n_layers);
    for (int i = 0; i < n_layers; i++) {
        layers.push_back(TransformerBlock(d_model, n_heads, d_ff));
    }
}

Tensor2D TransformerLauguageModel::Forward(const std::vector<int>& ids) const {
    std::vector<float> tmp((size_t)max_T);
    return Forward(ids, tmp.data());
}

Tensor2D TransformerLauguageModel::Forward(const std::vector<int>& ids, float* scratch) const {
    Tensor2D x = tok.Forward(ids);
    pos.AddInPlace(x);
    for (int i = 0; i < n_layers; i++) {
        x = layers[(size_t)i].Forward(x, scratch);
    }
    Tensor2D logits = lm_head.Forward(x);
    return logits;
}

bool TransformerLauguageModel::SaveToStream(std::ostream& out) const {
    if (!out.good()) return false;
    char magic[8] = {'T','R','A','N','S','M','L','P'};
    out.write(magic, 8);
    int version = 1;
    out.write((const char*)&version, sizeof(int));
    out.write((const char*)&vocab_size, sizeof(int));
    out.write((const char*)&d_model, sizeof(int));
    out.write((const char*)&n_heads, sizeof(int));
    out.write((const char*)&d_ff, sizeof(int));
    out.write((const char*)&n_layers, sizeof(int));
    out.write((const char*)&max_T, sizeof(int));

    struct W {
        static void tensor(std::ostream& o, const Tensor2D& t) {
            o.write((const char*)&t.R, sizeof(int));
            o.write((const char*)&t.C, sizeof(int));
            if (t.R > 0 && t.C > 0)
                o.write((const char*)t.data.data(), sizeof(float) * t.data.size());
        }
        static void vec(std::ostream& o, const std::vector<float>& v) {
            int n = (int)v.size();
            o.write((const char*)&n, sizeof(int));
            if (n) o.write((const char*)v.data(), sizeof(float) * n);
        }
    };

    W::tensor(out, tok.W);
    W::tensor(out, pos.P);
    for (int i = 0; i < n_layers; i++) {
        const TransformerBlock& b = layers[(size_t)i];
        W::vec(out, b.ln1.gamma); W::vec(out, b.ln1.beta);
        W::vec(out, b.ln2.gamma); W::vec(out, b.ln2.beta);
        W::tensor(out, b.attn.Wq.W); W::vec(out, b.attn.Wq.b);
        W::tensor(out, b.attn.Wk.W); W::vec(out, b.attn.Wk.b);
        W::tensor(out, b.attn.Wv.W); W::vec(out, b.attn.Wv.b);
        W::tensor(out, b.attn.Wo.W); W::vec(out, b.attn.Wo.b);
        W::tensor(out, b.ffn.fc1.W); W::vec(out, b.ffn.fc1.b);
        W::tensor(out, b.ffn.fc2.W); W::vec(out, b.ffn.fc2.b);
    }
    W::tensor(out, lm_head.W);
    W::vec(out, lm_head.b);
    return out.good();
}

bool TransformerLauguageModel::LoadFromStream(std::istream& in) {
    if (!in.good()) return false;
    char magic[8];
    in.read(magic, 8);
    int version = 0;
    in.read((char*)&version, sizeof(int));
    int V, DM, H, DFF, L, MT;
    in.read((char*)&V, sizeof(int));
    in.read((char*)&DM, sizeof(int));
    in.read((char*)&H, sizeof(int));
    in.read((char*)&DFF, sizeof(int));
    in.read((char*)&L, sizeof(int));
    in.read((char*)&MT, sizeof(int));
    *this = TransformerLauguageModel(V, DM, H, DFF, L, MT);

    struct R {
        static void tensor(std::istream& i, Tensor2D& t) {
            int r, c;
            i.read((char*)&r, sizeof(int));
            i.read((char*)&c, sizeof(int));
            t = Tensor2D(r, c);
            if (r > 0 && c > 0)
                i.read((char*)t.data.data(), sizeof(float) * t.data.size());
        }
        static void vec(std::istream& i, std::vector<float>& v) {
            int n = 0;
            i.read((char*)&n, sizeof(int));
            v.assign((size_t)n, 0.0f);
            if (n) i.read((char*)v.data(), sizeof(float) * n);
        }
    };

    R::tensor(in, tok.W);
    R::tensor(in, pos.P);
    for (int i = 0; i < n_layers; i++) {
        TransformerBlock& b = layers[(size_t)i];
        R::vec(in, b.ln1.gamma); R::vec(in, b.ln1.beta);
        R::vec(in, b.ln2.gamma); R::vec(in, b.ln2.beta);
        R::tensor(in, b.attn.Wq.W); R::vec(in, b.attn.Wq.b);
        R::tensor(in, b.attn.Wk.W); R::vec(in, b.attn.Wk.b);
        R::tensor(in, b.attn.Wv.W); R::vec(in, b.attn.Wv.b);
        R::tensor(in, b.attn.Wo.W); R::vec(in, b.attn.Wo.b);
        R::tensor(in, b.ffn.fc1.W); R::vec(in, b.ffn.fc1.b);
        R::tensor(in, b.ffn.fc2.W); R::vec(in, b.ffn.fc2.b);
    }
    R::tensor(in, lm_head.W);
    R::vec(in, lm_head.b);
    return in.good();
}

bool TransformerLauguageModel::Save(const std::string& path) const {
    std::ofstream out(path.c_str(), std::ios::binary);
    if (!out.is_open()) return false;
    bool ok = SaveToStream(out);
    out.close();
    return ok;
}

bool TransformerLauguageModel::Load(const std::string& path) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in.is_open()) return false;
    bool ok = LoadFromStream(in);
    in.close();
    return ok;
}
