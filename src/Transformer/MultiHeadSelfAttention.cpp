#include "MultiHeadSelfAttention.h"
#include "LinearLayer.h"

#include <algorithm>
#include <cmath>

MultiHeadSelfAttention::MultiHeadSelfAttention() : d_model(0), n_heads(0), d_head(0) {}

MultiHeadSelfAttention::MultiHeadSelfAttention(int dmodel, int heads)
    : d_model(dmodel), n_heads(heads), d_head(dmodel / heads),
    Wq(dmodel, dmodel), Wk(dmodel, dmodel), Wv(dmodel, dmodel), Wo(dmodel, dmodel) {}

Tensor2D MultiHeadSelfAttention::Forward(const Tensor2D& x, float* scratch) const {
    Tensor2D Q = Wq.Forward(x);
    Tensor2D K = Wk.Forward(x);
    Tensor2D V = Wv.Forward(x);
    
    Tensor2D out(x.R, d_model);
    out.Zero();
    
    float scale = 1.0f / std::sqrt((float)d_head);
    
    for (int h = 0; h < n_heads; h++) {
        int off = h * d_head;
        for (int t = 0; t < x.R; t++) {
            // find max logit for stability
            float maxlogit = -std::numeric_limits<float>::infinity();
            for (int u = 0; u <= t; u++) {
                float dot = 0.0f;
                const float* q = &Q.data[(size_t)t * Q.C + off];
                const float* k = &K.data[(size_t)u * K.C + off];
                for (int c = 0; c < d_head; c++) dot += q[c] * k[c];
                float logit = dot * scale;
                if (logit > maxlogit) maxlogit = logit;
            }
            // compute weights
            float denom = 0.0f;
            std::vector<float> w((size_t)t + 1, 0.0f);
            for (int u = 0; u <= t; u++) {
                float dot = 0.0f;
                const float* q = &Q.data[(size_t)t * Q.C + off];
                const float* k = &K.data[(size_t)u * K.C + off];
                for (int c = 0; c < d_head; c++) dot += q[c] * k[c];
                float logit = dot * scale;
                float e = std::exp(logit - maxlogit);
                w[(size_t)u] = e;
                denom += e;
            }
            float invden = 1.0f / denom;
            
            float* out_row = out.Row(t);
            for (int u = 0; u <= t; u++) {
                float ww = w[(size_t)u] * invden;
                const float* v = &V.data[(size_t)u * V.C + off];
                for (int c = 0; c < d_head; c++) {
                    out_row[off + c] += ww * v[c];
                }
            }
        }
    }
    
    Tensor2D y = Wo.Forward(out);
    return y;
}
