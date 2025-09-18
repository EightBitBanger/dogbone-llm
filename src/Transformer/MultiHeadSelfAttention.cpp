#include "MultiHeadSelfAttention.h"
#include "LinearLayer.h"

#include <algorithm>
#include <cmath>
#include <limits>

MultiHeadSelfAttention::MultiHeadSelfAttention() : d_model(0), n_heads(0), d_head(0) {}

MultiHeadSelfAttention::MultiHeadSelfAttention(int dmodel, int heads)
    : d_model(dmodel), n_heads(heads), d_head(dmodel / heads),
      Wq(dmodel, dmodel), Wk(dmodel, dmodel), Wv(dmodel, d_model), Wo(d_model, d_model) {}

Tensor2D MultiHeadSelfAttention::Forward(const Tensor2D& x, float* scratch) const {
    return Forward(x, nullptr, scratch);
}

Tensor2D MultiHeadSelfAttention::Forward(const Tensor2D& x, const std::vector<uint8_t>* key_keep, float* /*scratch*/) const {
    Tensor2D Q = Wq.Forward(x);
    Tensor2D K = Wk.Forward(x);
    Tensor2D V = Wv.Forward(x);

    Tensor2D out(x.R, d_model);
    out.Zero();

    const float scale = 1.0f / std::sqrt((float)d_head);
    std::vector<float> logits; // reused per (h,t)
    std::vector<float> w;

    for (int h = 0; h < n_heads; ++h) {
        const int off = h * d_head;
        for (int t = 0; t < x.R; ++t) {
            logits.assign((size_t)t + 1, -INFINITY);

            const float* q = &Q.data[(size_t)t * Q.C + off];

            // 1) compute masked logits once
            float maxlogit = -std::numeric_limits<float>::infinity();
            for (int u = 0; u <= t; ++u) {
                if (key_keep && (*key_keep)[(size_t)u] == 0) continue; // mask PAD key
                const float* k = &K.data[(size_t)u * K.C + off];
                float dot = 0.0f;
                for (int c = 0; c < d_head; ++c) dot += q[c] * k[c];
                float logit = dot * scale;
                logits[(size_t)u] = logit;
                if (logit > maxlogit) maxlogit = logit;
            }
            if (maxlogit == -std::numeric_limits<float>::infinity()) continue;

            // 2) softmax with stability
            double denom = 0.0;
            w.assign((size_t)t + 1, 0.0f);
            for (int u = 0; u <= t; ++u) {
                float logit = logits[(size_t)u];
                if (!std::isfinite(logit)) continue;
                float e = std::exp(logit - maxlogit);
                w[(size_t)u] = e;
                denom += e;
            }
            if (denom == 0.0) continue;
            float invden = 1.0f / (float)denom;

            // 3) weighted sum of V
            float* out_row = out.Row(t);
            for (int u = 0; u <= t; ++u) {
                float ww = w[(size_t)u] * invden;
                if (ww == 0.0f) continue;
                const float* v = &V.data[(size_t)u * V.C + off];
                for (int c = 0; c < d_head; ++c) {
                    out_row[off + c] += ww * v[c];
                }
            }
        }
    }

    Tensor2D y = Wo.Forward(out);
    return y;
}
