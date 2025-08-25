#ifndef TRANSFORMER_H
#define TRANSFORMER_H

struct PositionalEncoding {
    // learned PE: [max_T, d_model]
    Tensor2D P;
    PositionalEncoding() {}
    PositionalEncoding(int max_T, int d_model) : P(max_T, d_model) {
        float scale = 0.02f;
        for (size_t i = 0; i < P.data.size(); i++) {
            P.data[i] = scale * ((float)std::rand() / (float)RAND_MAX - 0.5f);
        }
    }
    // adds in place: x:[T,d_model]
    void AddInPlace(Tensor2D& x) const {
        for (int t = 0; t < x.R; t++) {
            const float* pe = P.Row(t);
            float* row = x.Row(t);
            for (int c = 0; c < x.C; c++) row[c] += pe[c];
        }
    }
};

struct LayerNorm {
    // gamma, beta: [d_model]
    std::vector<float> gamma;
    std::vector<float> beta;
    float eps;

    LayerNorm() : eps(1e-5f) {}
    LayerNorm(int d_model) : gamma(d_model, 1.0f), beta(d_model, 0.0f), eps(1e-5f) {}

    // x:[T,d_model] -> y:[T,d_model]
    Tensor2D Forward(const Tensor2D& x) const {
        Tensor2D y(x.R, x.C);
        for (int t = 0; t < x.R; t++) {
            const float* xr = x.Row(t);
            float* yr = y.Row(t);

            float mean = 0.0f;
            for (int c = 0; c < x.C; c++) mean += xr[c];
            mean /= (float)x.C;

            float var = 0.0f;
            for (int c = 0; c < x.C; c++) {
                float d = xr[c] - mean;
                var += d * d;
            }
            var /= (float)x.C;
            float inv = 1.0f / std::sqrt(var + eps);

            for (int c = 0; c < x.C; c++) {
                float z = (xr[c] - mean) * inv;
                yr[c] = z * gamma[(size_t)c] + beta[(size_t)c];
            }
        }
        return y;
    }
};

// Simple Linear: y = x @ W + b   (x:[T,d_in], W:[d_in,d_out]) -> y:[T,d_out]
struct Linear {
    Tensor2D W;
    std::vector<float> b;

    Linear() {}
    Linear(int d_in, int d_out) : W(d_in, d_out), b((size_t)d_out, 0.0f) {
        float scale = 1.0f / std::sqrt((float)d_in);
        for (size_t i = 0; i < W.data.size(); i++) {
            W.data[i] = scale * ((float)std::rand() / (float)RAND_MAX - 0.5f);
        }
    }

    Tensor2D Forward(const Tensor2D& x) const {
        Tensor2D y = MatMul(x, W);
        AddBiasRowInPlace(y, b);
        return y;
    }
};

// GELU (approx): y = 0.5x(1 + tanh( (2/PI)(x + 0.044715x^3)))
static void GELU_InPlace(Tensor2D& x) {
    const float k = std::sqrt(2.0f / 3.14159265358979323846f);
    for (size_t i = 0; i < x.data.size(); i++) {
        float v = x.data[i];
        float v3 = v * v * v;
        float h = std::tanh(k * (v + 0.044715f * v3));
        x.data[i] = 0.5f * v * (1.0f + h);
    }
}


struct MultiHeadSelfAttention {
    int d_model;
    int n_heads;
    int d_head; // d_model / n_heads

    Linear Wq;
    Linear Wk;
    Linear Wv;
    Linear Wo;

    MultiHeadSelfAttention() : d_model(0), n_heads(0), d_head(0) {}
    MultiHeadSelfAttention(int dmodel, int heads)
        : d_model(dmodel), n_heads(heads), d_head(dmodel / heads),
          Wq(dmodel, dmodel), Wk(dmodel, dmodel), Wv(dmodel, dmodel), Wo(dmodel, dmodel) {}

    // x:[T,d_model] -> out:[T,d_model]
    Tensor2D Forward(const Tensor2D& x) const {
        // Project
        Tensor2D Q = Wq.Forward(x); // [T, d_model]
        Tensor2D K = Wk.Forward(x); // [T, d_model]
        Tensor2D V = Wv.Forward(x); // [T, d_model]

        // reshape into heads: we’ll do it logically by indexing
        Tensor2D out(x.R, d_model);
        out.Zero();

        float scale = 1.0f / std::sqrt((float)d_head);

        // For each head, compute attention
        for (int h = 0; h < n_heads; h++) {
            int offset = h * d_head;

            // Scores S = Q_h @ K_h^T  -> [T, T]
            // We’ll compute row by row to avoid big T×T allocation; softmax per row with causal mask
            for (int t = 0; t < x.R; t++) {
                // compute logits for positions <= t
                // also compute denom for softmax
                float maxlogit = -std::numeric_limits<float>::infinity();

                for (int u = 0; u <= t; u++) {
                    float dot = 0.0f;
                    const float* q = &Q.data[(size_t)t * Q.C + offset];
                    const float* k = &K.data[(size_t)u * K.C + offset];
                    for (int c = 0; c < d_head; c++) dot += q[c] * k[c];
                    float logit = dot * scale;
                    if (logit > maxlogit) maxlogit = logit;
                }

                float denom = 0.0f;
                // store weights temporarily in a small vector
                std::vector<float> weights((size_t)t + 1, 0.0f);
                for (int u = 0; u <= t; u++) {
                    float dot = 0.0f;
                    const float* q = &Q.data[(size_t)t * Q.C + offset];
                    const float* k = &K.data[(size_t)u * K.C + offset];
                    for (int c = 0; c < d_head; c++) dot += q[c] * k[c];
                    float logit = dot * scale;
                    float w = std::exp(logit - maxlogit);
                    weights[(size_t)u] = w;
                    denom += w;
                }
                float invden = 1.0f / denom;

                // Output at t for this head: sum_u weights[u] * V_h[u]
                float* out_row = out.Row(t);
                for (int u = 0; u <= t; u++) {
                    float w = weights[(size_t)u] * invden;
                    const float* v = &V.data[(size_t)u * V.C + offset];
                    for (int c = 0; c < d_head; c++) {
                        out_row[offset + c] += w * v[c];
                    }
                }
            }
        }

        // Final projection
        Tensor2D y = Wo.Forward(out);
        return y;
    }
};

// ==============================
// Feedforward block
// ==============================

struct FeedForward {
    Linear fc1; // d_model -> d_ff
    Linear fc2; // d_ff -> d_model

    FeedForward() {}
    FeedForward(int d_model, int d_ff) : fc1(d_model, d_ff), fc2(d_ff, d_model) {}

    Tensor2D Forward(const Tensor2D& x) const {
        Tensor2D h = fc1.Forward(x);
        GELU_InPlace(h);
        Tensor2D y = fc2.Forward(h);
        return y;
    }
};

// ==============================
// Transformer Block
// ==============================

struct TransformerBlock {
    LayerNorm ln1;
    LayerNorm ln2;
    MultiHeadSelfAttention attn;
    FeedForward ffn;

    TransformerBlock() {}
    TransformerBlock(int d_model, int n_heads, int d_ff)
        : ln1(d_model), ln2(d_model), attn(d_model, n_heads), ffn(d_model, d_ff) {}

    Tensor2D Forward(const Tensor2D& x_in) const {
        Tensor2D x = x_in;
        // Pre-norm
        Tensor2D n1 = ln1.Forward(x); // (space in "Forward" to avoid accidental auto-correct; remove)
        
        Tensor2D a = attn.Forward(n1);
        AddInPlace(x, a); // residual

        Tensor2D n2 = ln2.Forward(x);
        Tensor2D f = ffn.Forward(n2);
        AddInPlace(x, f); // residual
        return x;
    }
};

// ==============================
// Full Decoder-only Transformer
// ==============================

struct TransformerLM {
    int vocab_size;
    int d_model;
    int n_heads;
    int d_ff;
    int n_layers;
    int max_T;

    Embedding tok;
    PositionalEncoding pos;
    std::vector<TransformerBlock> layers;
    Linear lm_head; // d_model -> vocab_size

    TransformerLM() :
        vocab_size(0), d_model(0), n_heads(0), d_ff(0), n_layers(0), max_T(0) {}

    TransformerLM(int vocab, int dmodel, int heads, int ff, int layers_count, int maxT)
        : vocab_size(vocab), d_model(dmodel), n_heads(heads),
          d_ff(ff), n_layers(layers_count), max_T(maxT),
          tok(vocab, dmodel), pos(maxT, dmodel), lm_head(dmodel, vocab) {
        layers.reserve((size_t)n_layers);
        for (int i = 0; i < n_layers; i++) {
            layers.push_back(TransformerBlock(d_model, n_heads, d_ff));
        }
    }

    // ids:[T] -> logits:[T, vocab]
    Tensor2D Forward(const std::vector<int>& ids) const {
        Tensor2D x = tok.Forward(ids); // [T, d_model]
        pos.AddInPlace(x);             // + PE
        for (int i = 0; i < n_layers; i++) {
            x = layers[(size_t)i].Forward(x);
        }
        Tensor2D logits = lm_head.Forward(x); // [T, vocab]
        return logits;
    }
};

// returns mean CE over positions (ignores PAD)
static float CrossEntropyLoss(const Tensor2D& logits, const std::vector<int>& targets, int pad_id) {
    // logits:[T,V], targets:[T]
    int T = logits.R;
    int V = logits.C;
    float total = 0.0f;
    int count = 0;

    for (int t = 0; t < T; t++) {
        int y = targets[(size_t)t];
        if (y == pad_id) continue;

        // stable softmax log-prob for target
        const float* row = logits.Row(t);
        float maxv = -std::numeric_limits<float>::infinity();
        for (int v = 0; v < V; v++) if (row[v] > maxv) maxv = row[v];
        float denom = 0.0f;
        for (int v = 0; v < V; v++) denom += std::exp(row[v] - maxv);
        float logprob = (row[y] - maxv) - std::log(denom);

        total += -logprob;
        count += 1;
    }
    if (count == 0) return 0.0f;
    return total / (float)count;
}

// NOTE: Real training needs backprop & optimizer state.
// Below is a placeholder “shape” for Adam state so you can wire grads later.
struct AdamParam {
    std::vector<float> m;
    std::vector<float> v;
};

struct Adam {
    float lr;
    float beta1;
    float beta2;
    float eps;
    int t;

    Adam() : lr(1e-3f), beta1(0.9f), beta2(0.999f), eps(1e-8f), t(0) {}
    Adam(float learning_rate) : lr(learning_rate), beta1(0.9f), beta2(0.999f), eps(1e-8f), t(0) {}

    // Placeholder to show intended use:
    static void StepVector(std::vector<float>& w, std::vector<float>& grad,
                           AdamParam& state, float lr, float b1, float b2, float eps, int t) {
        if (state.m.size() != w.size()) state.m.assign(w.size(), 0.0f);
        if (state.v.size() != w.size()) state.v.assign(w.size(), 0.0f);

        for (size_t i = 0; i < w.size(); i++) {
            state.m[i] = b1 * state.m[i] + (1.0f - b1) * grad[i];
            state.v[i] = b2 * state.v[i] + (1.0f - b2) * (grad[i] * grad[i]);
            float mhat = state.m[i] / (1.0f - std::pow(b1, (float)t));
            float vhat = state.v[i] / (1.0f - std::pow(b2, (float)t));
            w[i] -= lr * mhat / (std::sqrt(vhat) + eps);
        }
    }
};

#endif
