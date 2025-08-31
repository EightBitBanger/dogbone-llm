#include "NeuralNetwork.h"
#include "CrossEntropyLoss.h"
#include <cstring>
#include "ShaderTensor.h"
#include <cstdint>
#include <string>

// === GPU linear forward via ShaderTensor (X [T x IN] * W [IN x OUT] + b [OUT]) ===
static bool gpu_linear_forward(ShaderTensor* gpu, const Tensor2D& X, const LinearLayer& lin, Tensor2D& Y) {
    const int T = X.R, IN = X.C, OUT = lin.W.C;
    if (lin.W.R != IN || (int)lin.b.size() != OUT) return false;
    
    // buffers
    std::ptrdiff_t bytesX = (std::ptrdiff_t)T * IN * sizeof(float);
    std::ptrdiff_t bytesW = (std::ptrdiff_t)IN * OUT * sizeof(float);
    std::ptrdiff_t bytesB = (std::ptrdiff_t)OUT * sizeof(float);
    std::ptrdiff_t bytesY = (std::ptrdiff_t)T * OUT * sizeof(float);
    int meta[3] = { T, IN, OUT };
    
    // correct order: (name, sizeBytes, bindingIndex)
    gpu->ensureSSBO("X",   bytesX, 0);
    gpu->ensureSSBO("W",   bytesW, 1);
    gpu->ensureSSBO("B",   bytesB, 2);
    gpu->ensureSSBO("Y",   bytesY, 3);
    gpu->ensureSSBO("Meta", (std::ptrdiff_t)sizeof(meta), 4);
    
    gpu->upload("X", X.data.data(), bytesX);
    gpu->upload("W", lin.W.data.data(), bytesW);
    gpu->upload("B", lin.b.data(), bytesB);
    gpu->upload("Meta", meta, (std::ptrdiff_t)sizeof(meta));

    // Dispatch
    unsigned gx = (unsigned)((OUT + 15) / 16);
    unsigned gy = (unsigned)((T   + 15) / 16);
    gpu->use();
    gpu->dispatch(gx, gy, 1);
    gpu->sync(); // ensure completion

    // Readback
    std::vector<float> hostY((size_t)T * (size_t)OUT);
    gpu->downloadSync("Y", hostY.data(), bytesY);

    Y = Tensor2D(T, OUT);
    for (int r = 0; r < T; ++r) {
        float* yr = Y.Row(r);
        std::memcpy(yr, hostY.data() + (size_t)r * OUT, (size_t)OUT * sizeof(float));
    }
    return true;
}



// === GPU linear forward using resident W/B (weights already uploaded) ===
static bool gpu_linear_forward_resident(ShaderTensor* gpu, const Tensor2D& X, int IN, int OUT, Tensor2D& Y) {
    if (!gpu) return false;
    const int T = X.R;
    if (X.C != IN) return false;

    std::ptrdiff_t bytesX = (std::ptrdiff_t)T * IN * sizeof(float);
    std::ptrdiff_t bytesY = (std::ptrdiff_t)T * OUT * sizeof(float);
    int meta[3] = { T, IN, OUT };

    gpu->ensureSSBO("X",   bytesX, 0);
    gpu->ensureSSBO("Y",   bytesY, 3);
    // meta is tiny; keep as SSBO binding 4 for parity with existing shader
    gpu->ensureSSBO("Meta", (std::ptrdiff_t)sizeof(meta), 4);

    gpu->upload("X", X.data.data(), bytesX);
    gpu->upload("Meta", meta, (std::ptrdiff_t)sizeof(meta));

    unsigned gx = (unsigned)((OUT + 15) / 16);
    unsigned gy = (unsigned)((T   + 15) / 16);
    gpu->use();
    gpu->dispatch(gx, gy, 1);

    // blocking readback for now; can be swapped to async later
    std::vector<float> hostY((size_t)T * (size_t)OUT);
    gpu->downloadSync("Y", hostY.data(), bytesY);

    Y = Tensor2D(T, OUT);
    for (int r = 0; r < T; ++r) {
        float* yr = Y.Row(r);
        std::memcpy(yr, hostY.data() + (size_t)r * OUT, (size_t)OUT * sizeof(float));
    }
    return true;
}
void NeuralNetwork::InitScratch(int max_T) {
    attnScratch_p.resize((size_t)max_T);
    attnScratch_gs.resize((size_t)max_T);
}

NeuralNetwork::NeuralNetwork(float lr) : opt(lr) {}

void NeuralNetwork::LinearBackward(const Tensor2D& x, const LinearLayer& lin, const Tensor2D& dy, Tensor2D& dW, std::vector<float>& db, Tensor2D& dx) {
    // Compute Weight/Bias/Input Gradients: dW = X Transposed Times dY; dB = Row Sum Of dY; dX = dY Times W Transposed.
    dW = Tensor2D(lin.W.R, lin.W.C);
    db.assign(lin.b.size(), 0.0f);
    dx = Tensor2D(x.R, x.C);
    // db
    for (int r = 0; r < dy.R; r++) {
        for (int c = 0; c < dy.C; c++) {
            db[(size_t)c] += dy.data[(size_t)r * dy.C + c];
        }
    }
    // dW
    for (int i = 0; i < x.R; i++) {
        const float* xi = x.Row(i);
        const float* dyi = dy.Row(i);
        for (int k = 0; k < lin.W.R; k++) {
            float xv = xi[k];
            for (int j = 0; j < lin.W.C; j++) {
                dW.data[(size_t)k * lin.W.C + j] += xv * dyi[j];
            }
        }
    }
    // Compute Input Gradient As dY Times The Transposed Weight Matrix.
    for (int i = 0; i < x.R; i++) {
        float* dxi = dx.Row(i);
        const float* dyi = dy.Row(i);
        for (int k = 0; k < lin.W.R; k++) {
            float sum = 0.0f;
            for (int j = 0; j < lin.W.C; j++) sum += dyi[j] * lin.W.data[(size_t)k * lin.W.C + j];
            dxi[k] = sum;
        }
    }
}

void NeuralNetwork::LayerNormBackward(const Tensor2D& x, const LayerNorm& ln, const Tensor2D& dy, 
                                             std::vector<float>& dgamma, std::vector<float>& dbeta, Tensor2D& dx) {
    int T = x.R, C = x.C;
    dgamma.assign(ln.gamma.size(), 0.0f);
    dbeta.assign(ln.beta.size(), 0.0f);
    dx = Tensor2D(T, C);

    for (int t = 0; t < T; t++) {
        const float* xr = x.Row(t);
        const float* dyr = dy.Row(t);
        // Compute Mean Of Features (Per Row).
        float mean = 0.0f;
        for (int c = 0; c < C; c++) mean += xr[c];
        mean /= (float)C;
        // Compute Variance Of Features (Per Row).
        float var = 0.0f;
        for (int c = 0; c < C; c++) {
            float d = xr[c] - mean;
            var += d * d;
        }
        var /= (float)C;
        float inv = 1.0f / std::sqrt(var + ln.eps);

        // xhat and grads
        float s1 = 0.0f; // sum(dy * gamma)
        float s2 = 0.0f; // sum(dy * gamma * xhat)
        for (int c = 0; c < C; c++) {
            float xhat = (xr[c] - mean) * inv;
            float g = dyr[c];
            dbeta[(size_t)c] += g;
            dgamma[(size_t)c] += g * xhat;
            s1 += g * ln.gamma[(size_t)c];
            s2 += g * ln.gamma[(size_t)c] * xhat;
        }
        for (int c = 0; c < C; c++) {
            float xhat = (xr[c] - mean) * inv;
            float term = (float)C * (dyr[c] * ln.gamma[(size_t)c]) - s1 - xhat * s2;
            dx.data[(size_t)t * C + c] = (inv / (float)C) * term;
        }
    }
}

// One step on a single (inputs, targets)
float NeuralNetwork::Step(LauguageModel& model, const std::vector<int>& inputs, const std::vector<int>& targets, 
                          int pad_id, GradientAccumulator* acc, bool apply_updates) {
    if (apply_updates) { if ((int)layer_states.size() != model.n_layers) layer_states.resize((size_t)model.n_layers); }

    // --------- Forward with caches ---------
    Tensor2D x0 = model.tok.Forward(inputs); // [T, d_model]
    model.pos.AddInPlace(x0);

    // caches per layer
    struct Cache {
        Tensor2D x_in;
        Tensor2D n1;
        Tensor2D Q, K, V;
        // We will recompute softmax probs on the fly in backward for memory economy.
        Tensor2D attn_concat; // before Wo
        Tensor2D attn_out;    // after Wo (to be added residually)
        Tensor2D x_attn_res;  // x after attention residual
        Tensor2D n2;
        Tensor2D ff1_out;
        Tensor2D ff1_act;
        Tensor2D ff2_out;
    };
    std::vector<Cache> Caches((size_t)model.n_layers);

    Tensor2D x = x0;
    for (int l = 0; l < model.n_layers; l++) {
        const TransformerBlock& B = model.layers[(size_t)l];
        Cache& C = Caches[(size_t)l];
        
        C.x_in = x;
        C.n1 = B.ln1.Forward(C.x_in);                 // pre norm to stabilize before attention
        
        // attention: project inputs to q, k, v for all heads (packed along feature dim)
        C.Q = B.attn.Wq.Forward(C.n1);
        C.K = B.attn.Wk.Forward(C.n1);
        C.V = B.attn.Wv.Forward(C.n1);
        
        C.attn_concat = Tensor2D(C.n1.R, model.d_model);
        C.attn_concat.Zero();                         // init buffer that will hold all head outputs
        float scale = 1.0f / std::sqrt((float)B.attn.d_head);  // use 1/sqrt(head dim) for stable scaling
        
        // per head and per token: causal attention over keys u <= t, then weighted sum of v
        for (int h = 0; h < B.attn.n_heads; h++) {
            int off = h * B.attn.d_head;              // slice offset for this head
            for (int t = 0; t < C.n1.R; t++) {
                float maxlogit = -std::numeric_limits<float>::infinity();
                // pass 1: find row wise max logit for stable softmax
                for (int u = 0; u <= t; u++) {
                    float dot = 0.0f;
                    const float* q = &C.Q.data[(size_t)t * C.Q.C + off];
                    const float* k = &C.K.data[(size_t)u * C.K.C + off];
                    for (int c = 0; c < B.attn.d_head; c++) dot += q[c] * k[c];
                    float logit = dot * scale;
                    if (logit > maxlogit) maxlogit = logit;
                }
        
                float denom = 0.0f;
                std::vector<float> w((size_t)t + 1, 0.0f);
                // pass 2: compute unnormalized weights exp(logit - maxlogit) and their sum
                for (int u = 0; u <= t; u++) {
                    float dot = 0.0f;
                    const float* q = &C.Q.data[(size_t)t * C.Q.C + off];
                    const float* k = &C.K.data[(size_t)u * C.K.C + off];
                    for (int c = 0; c < B.attn.d_head; c++) dot += q[c] * k[c];
                    float logit = dot * scale;
                    float e = std::exp(logit - maxlogit);
                    w[(size_t)u] = e;
                    denom += e;
                }
        
                float invden = 1.0f / denom;          // softmax normalization factor
                float* out_row = C.attn_concat.Row(t);
                // pass 3: write this head output as sum over u of weight times v into its slice
                for (int u = 0; u <= t; u++) {
                    float ww = w[(size_t)u] * invden; // attention weight for key u at time t
                    const float* v = &C.V.data[(size_t)u * C.V.C + off];
                    for (int c = 0; c < B.attn.d_head; c++) {
                        out_row[off + c] += ww * v[c];
                    }
                }
            }
        }
        
        // mix heads and add residual
        C.attn_out = B.attn.Wo.Forward(C.attn_concat); // output projection to merge heads
        C.x_attn_res = C.x_in;
        AddInPlace(C.x_attn_res, C.attn_out);          // residual add for attention block
        
        // mlp activation block with pre norm and activation, then residual add
        C.n2       = B.ln2.Forward(C.x_attn_res);
        C.ff1_out  = B.ffn.fc1.Forward(C.n2);       // [T, d_ff_mul * d_ff]
        C.ff1_act  = Activation.Forward(C.ff1_out); // [T, d_ff]
        C.ff2_out  = B.ffn.fc2.Forward(C.ff1_act);
        x = C.x_attn_res;
        AddInPlace(x, C.ff2_out);                   // residual add for mlp block
    }

    Tensor2D logits = model.lm_head.Forward(x);
    float loss = CrossEntropyLoss(logits, targets, pad_id);

    // Backward propagation
    int T = logits.R;
    int V = logits.C;
    // dlogits
    Tensor2D dlogits(T, V);
    for (int t = 0; t < T; t++) {
        int y = targets[(size_t)t];
        if (y == pad_id) {
            std::fill(dlogits.Row(t), dlogits.Row(t) + V, 0.0f);
            continue;
        }
        const float* row = logits.Row(t);
        float maxv = -std::numeric_limits<float>::infinity();
        for (int v = 0; v < V; v++) if (row[v] > maxv) maxv = row[v];
        float denom = 0.0f;
        for (int v = 0; v < V; v++) denom += std::exp(row[v] - maxv);
        float invden = 1.0f / denom;
        float* dlr = dlogits.Row(t);
        for (int v = 0; v < V; v++) {
            float p = std::exp(row[v] - maxv) * invden;
            dlr[v] = p - ((v == y) ? 1.0f : 0.0f);
        }
    }
    // average over non-pad count
    int count = 0;
    for (int t = 0; t < T; t++) if (targets[(size_t)t] != pad_id) count++;
    if (count > 0) {
        float inv = 1.0f / (float)count;
        for (size_t i = 0; i < dlogits.data.size(); i++) dlogits.data[i] *= inv;
    }

    // Grad lm_head
    Tensor2D dW_lm, dx_last;
    std::vector<float> db_lm;
    LinearBackward(x, model.lm_head, dlogits, dW_lm, db_lm, dx_last);

    // Backprop through blocks (reverse order)
    Tensor2D dx = dx_last;
    // init grads accumulators
    Tensor2D d_tokW(model.tok.W.R, model.tok.W.C); std::fill(d_tokW.data.begin(), d_tokW.data.end(), 0.0f);
    Tensor2D d_posP(model.pos.P.R, model.pos.P.C); std::fill(d_posP.data.begin(), d_posP.data.end(), 0.0f);

    std::vector<Tensor2D> d_ln1_gamma(model.n_layers), d_ln2_gamma(model.n_layers);
    std::vector<Tensor2D> dummy; // not used

    for (int l = model.n_layers - 1; l >= 0; l--) {
        TransformerBlock& B = model.layers[(size_t)l];
        Cache& C = Caches[(size_t)l];

        // Split gradient over x = x_attn_res + ff2_out
        Tensor2D d_ff2_out = dx;
        Tensor2D d_x_attn_res = dx;

        // Back through fc2
        Tensor2D dW_fc2, dx_fc2;
        std::vector<float> db_fc2;
        LinearBackward(C.ff1_act, B.ffn.fc2, d_ff2_out, dW_fc2, db_fc2, dx_fc2);

        // back through activation (ff1_out is [T, 2*d_ff], dx_fc2 is [T, d_ff])
        Tensor2D d_ff1_out;
        Activation.Backward(C.ff1_out, dx_fc2, d_ff1_out);

        // back through fc1
        Tensor2D dW_fc1, dx_fc1;
        std::vector<float> db_fc1;
        LinearBackward(C.n2, B.ffn.fc1, d_ff1_out, dW_fc1, db_fc1, dx_fc1);

        // Add dx from this path to d(n2)
        Tensor2D d_n2 = dx_fc1;

        // LayerNorm 2 backward (inputs C.x_attn_res, output C.n2)
        std::vector<float> dgamma2, dbeta2;
        Tensor2D d_x_attn_res_from_ln2;
        LayerNormBackward(C.x_attn_res, B.ln2, d_n2, dgamma2, dbeta2, d_x_attn_res_from_ln2);

        // Sum into d_x_attn_res
        for (size_t i = 0; i < d_x_attn_res.data.size(); i++) d_x_attn_res.data[i] += d_x_attn_res_from_ln2.data[i];

        // Back through attention residual: x_attn_res = x_in + attn_out
        Tensor2D d_attn_out = d_x_attn_res; // part going into attn_out
        Tensor2D d_x_in_after_attn = d_x_attn_res; // part to x_in

        // Back through Wo
        Tensor2D dW_Wo, dx_attn_concat;
        std::vector<float> db_Wo;
        LinearBackward(C.attn_concat, B.attn.Wo, d_attn_out, dW_Wo, db_Wo, dx_attn_concat);

        // Back through attention combine into Q,K,V
        Tensor2D dQ(C.Q.R, C.Q.C); std::fill(dQ.data.begin(), dQ.data.end(), 0.0f);
        Tensor2D dK(C.K.R, C.K.C); std::fill(dK.data.begin(), dK.data.end(), 0.0f);
        Tensor2D dV(C.V.R, C.V.C); std::fill(dV.data.begin(), dV.data.end(), 0.0f);

        float scale = 1.0f / std::sqrt((float)B.attn.d_head);
        for (int h = 0; h < B.attn.n_heads; h++) {
            int off = h * B.attn.d_head;
            for (int t = 0; t < C.n1.R; t++) {
                // Recompute softmax probs for row t, head h
                float maxlogit = -std::numeric_limits<float>::infinity();
                for (int u = 0; u <= t; u++) {
                    float dot = 0.0f;
                    const float* q = &C.Q.data[(size_t)t * C.Q.C + off];
                    const float* k = &C.K.data[(size_t)u * C.K.C + off];
                    for (int c = 0; c < B.attn.d_head; c++) dot += q[c] * k[c];
                    float logit = dot * scale;
                    if (logit > maxlogit) maxlogit = logit;
                }
                float denom = 0.0f;
                std::vector<float> p((size_t)t + 1, 0.0f);
                for (int u = 0; u <= t; u++) {
                    float dot = 0.0f;
                    const float* q = &C.Q.data[(size_t)t * C.Q.C + off];
                    const float* k = &C.K.data[(size_t)u * C.K.C + off];
                    for (int c = 0; c < B.attn.d_head; c++) dot += q[c] * k[c];
                    float logit = dot * scale;
                    float e = std::exp(logit - maxlogit);
                    p[(size_t)u] = e;
                    denom += e;
                }
                float invden = 1.0f / denom;
                for (int u = 0; u <= t; u++) p[(size_t)u] *= invden;

                // grad wrt concat output part for this head at row t
                const float* gout = &dx_attn_concat.data[(size_t)t * dx_attn_concat.C + off];

                // 1) dV: sum over t of p[t,u] * gout[t]
                for (int u = 0; u <= t; u++) {
                    float w = p[(size_t)u];
                    float* dVrow = &dV.data[(size_t)u * dV.C + off];
                    for (int c = 0; c < B.attn.d_head; c++) dVrow[c] += w * gout[c];
                }

                // 2) g_s = dot(gout, V[u]); then dz via softmax Jacobian
                std::vector<float> g_s((size_t)t + 1, 0.0f);
                for (int u = 0; u <= t; u++) {
                    const float* vrow = &C.V.data[(size_t)u * C.V.C + off];
                    float s = 0.0f;
                    for (int c = 0; c < B.attn.d_head; c++) s += gout[c] * vrow[c];
                    g_s[(size_t)u] = s;
                }
                float sum_ps = 0.0f;
                for (int u = 0; u <= t; u++) sum_ps += p[(size_t)u] * g_s[(size_t)u];
                // dz = p * (g_s - sum(p*g_s))
                for (int u = 0; u <= t; u++) {
                    float gz = p[(size_t)u] * (g_s[(size_t)u] - sum_ps);
                    // z = scale * dot(Q[t], K[u])
                    float* dQt = &dQ.data[(size_t)t * dQ.C + off];
                    const float* Ku = &C.K.data[(size_t)u * C.K.C + off];
                    for (int c = 0; c < B.attn.d_head; c++) dQt[c] += scale * gz * Ku[c];

                    float* dKu = &dK.data[(size_t)u * dK.C + off];
                    const float* Qt = &C.Q.data[(size_t)t * C.Q.C + off];
                    for (int c = 0; c < B.attn.d_head; c++) dKu[c] += scale * gz * Qt[c];
                }
            }
        }

        // Back through Wq/Wk/Wv
        Tensor2D dWq, dWk, dWv, dx_q, dx_k, dx_v;
        std::vector<float> dbq, dbk, dbv;
        LinearBackward(C.n1, B.attn.Wq, dQ, dWq, dbq, dx_q);
        LinearBackward(C.n1, B.attn.Wk, dK, dWk, dbk, dx_k);
        LinearBackward(C.n1, B.attn.Wv, dV, dWv, dbv, dx_v);

        // Sum dx to n1 grad plus the x_in residual from attention path
        Tensor2D d_n1(dx_q.R, dx_q.C);
        for (int i = 0; i < d_n1.R * d_n1.C; i++) d_n1.data[(size_t)i] = dx_q.data[(size_t)i] + dx_k.data[(size_t)i] + dx_v.data[(size_t)i];

        // LayerNorm 1 backward
        std::vector<float> dgamma1, dbeta1;
        Tensor2D dx_ln1;
        LayerNormBackward(C.x_in, B.ln1, d_n1, dgamma1, dbeta1, dx_ln1);

        // Combine paths into dx for next lower layer:
        // dx comes from: residual to x_in after attn (d_x_in_after_attn) plus dx_ln1
        Tensor2D dx_next = d_x_in_after_attn;
        for (int i = 0; i < dx_next.R * dx_next.C; i++) dx_next.data[(size_t)i] += dx_ln1.data[(size_t)i];
        
        // Accumulate per-layer grads if requested
        if (acc) {
            GradientAccumulator::BlockGrads& GG = acc->layers[(size_t)l];
            // Wo
            for (size_t i=0;i<GG.dWo.data.size();++i) GG.dWo.data[i] += dW_Wo.data[i];
            for (size_t i=0;i<GG.dbo.size();++i)      GG.dbo[i]      += db_Wo[i];
            // Wq/Wk/Wv
            for (size_t i=0;i<GG.dWq.data.size();++i) GG.dWq.data[i] += dWq.data[i];
            for (size_t i=0;i<GG.dbq.size();++i)      GG.dbq[i]      += dbq[i];
            for (size_t i=0;i<GG.dWk.data.size();++i) GG.dWk.data[i] += dWk.data[i];
            for (size_t i=0;i<GG.dbk.size();++i)      GG.dbk[i]      += dbk[i];
            for (size_t i=0;i<GG.dWv.data.size();++i) GG.dWv.data[i] += dWv.data[i];
            for (size_t i=0;i<GG.dbv.size();++i)      GG.dbv[i]      += dbv[i];
            // fc1/fc2
            for (size_t i=0;i<GG.d_fc1W.data.size();++i) GG.d_fc1W.data[i] += dW_fc1.data[i];
            for (size_t i=0;i<GG.d_fc1b.size();++i)      GG.d_fc1b[i]      += db_fc1[i];
            for (size_t i=0;i<GG.d_fc2W.data.size();++i) GG.d_fc2W.data[i] += dW_fc2.data[i];
            for (size_t i=0;i<GG.d_fc2b.size();++i)      GG.d_fc2b[i]      += db_fc2[i];
            // LN
            for (size_t i=0;i<GG.d_ln1g.size();++i) GG.d_ln1g[i] += dgamma1[i];
            for (size_t i=0;i<GG.d_ln1b.size();++i) GG.d_ln1b[i] += dbeta1[i];
            for (size_t i=0;i<GG.d_ln2g.size();++i) GG.d_ln2g[i] += dgamma2[i];
            for (size_t i=0;i<GG.d_ln2b.size();++i) GG.d_ln2b[i] += dbeta2[i];
        }

        if (apply_updates) {
            // Adam updates for layer l parameters
            opt.t += 1; // one "step" per layer update; simple schedule

            // Wo
            Adam::StepInPlace(B.attn.Wo.W.data, dW_Wo.data, layer_states[(size_t)l].WoW_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.attn.Wo.b, db_Wo, layer_states[(size_t)l].Wob_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            // Wq/Wk/Wv
            Adam::StepInPlace(B.attn.Wq.W.data, dWq.data, layer_states[(size_t)l].WqW_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.attn.Wq.b, dbq, layer_states[(size_t)l].Wqb_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.attn.Wk.W.data, dWk.data, layer_states[(size_t)l].WkW_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.attn.Wk.b, dbk, layer_states[(size_t)l].Wkb_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.attn.Wv.W.data, dWv.data, layer_states[(size_t)l].WvW_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.attn.Wv.b, dbv, layer_states[(size_t)l].Wvb_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            // fc1/fc2
            Adam::StepInPlace(B.ffn.fc1.W.data, dW_fc1.data, layer_states[(size_t)l].fc1W_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.ffn.fc1.b, db_fc1, layer_states[(size_t)l].fc1b_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.ffn.fc2.W.data, dW_fc2.data, layer_states[(size_t)l].fc2W_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.ffn.fc2.b, db_fc2, layer_states[(size_t)l].fc2b_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            // LN params
            Adam::StepInPlace(B.ln1.gamma, dgamma1, layer_states[(size_t)l].ln1g_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.ln1.beta, dbeta1, layer_states[(size_t)l].ln1b_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.ln2.gamma, dgamma2, layer_states[(size_t)l].ln2g_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.ln2.beta, dbeta2, layer_states[(size_t)l].ln2b_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
        }

        dx = dx_next;

    }

    // Back to embeddings and positional enc
    // Accumulate grads for pos.P and tok.W
    // dx is gradient wrt x0 (tok + pos)
    // dpos is simply dx (clamped by max_T)
    int limit = (x0.R <= model.pos.P.R) ? x0.R : model.pos.P.R;
    for (int t = 0; t < limit; t++) {
        float* dprow = &d_posP.data[(size_t)t * d_posP.C];
        const float* dxrow = dx.Row(t);
        for (int c = 0; c < d_posP.C; c++) dprow[c] += dxrow[c];
    }
    // dTokW: scatter-add into rows by ids
    for (int t = 0; t < (int)inputs.size(); t++) {
        int id = inputs[(size_t)t];
        if (id < 0 || id >= model.tok.W.R) continue;
        const float* dxrow = dx.Row(t);
        for (int c = 0; c < model.tok.W.C; c++) d_tokW.data[(size_t)id * model.tok.W.C + c] += dxrow[c];
    }

    // Accumulate top-level grads
    if (acc) {
        for (size_t i=0;i<acc->d_lmW.data.size();++i) acc->d_lmW.data[i] += dW_lm.data[i];
        for (size_t i=0;i<acc->d_lmb.size();++i)      acc->d_lmb[i]      += db_lm[i];
        for (size_t i=0;i<acc->d_posP.data.size();++i) acc->d_posP.data[i] += d_posP.data[i];
        for (size_t i=0;i<acc->d_tokW.data.size();++i) acc->d_tokW.data[i] += d_tokW.data[i];
    }

    if (apply_updates) {
        // Top-level Adam steps (lm_head and pos/tok)
        opt.t += 1;
        Adam::StepInPlace(model.lm_head.W.data, dW_lm.data, lmW_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
        Adam::StepInPlace(model.lm_head.b, db_lm, lmb_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
        Adam::StepInPlace(model.pos.P.data, d_posP.data, posP_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
        Adam::StepInPlace(model.tok.W.data, d_tokW.data, tokW_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
    }

    return loss;
}

float NeuralNetwork::StepGPU(LauguageModel& model, const std::vector<int>& inputs, const std::vector<int>& targets,
                             int pad_id, GradientAccumulator* acc, bool apply_updates) {
    if (mGPUResident.enabled && mGPUResident.map.empty()) { UploadWeightsToGPU(model); }

    // If no GPU is attached, just run the CPU path.
    if (!mGpu) return Step(model, inputs, targets, pad_id, acc, apply_updates);

    if (apply_updates) {
        if ((int)layer_states.size() != model.n_layers) layer_states.resize((size_t)model.n_layers);
    }

    // Forward with caches (GPU-accelerated linears)
    Tensor2D x0 = model.tok.Forward(inputs); // [T, d_model]  (emb lookup remains on CPU)
    model.pos.AddInPlace(x0);

    struct Cache {
        Tensor2D x_in;
        Tensor2D n1;
        Tensor2D Q, K, V;
        Tensor2D attn_concat; // before Wo
        Tensor2D attn_out;    // after Wo (to be added residually)
        Tensor2D x_attn_res;  // x after attention residual
        Tensor2D n2;
        Tensor2D ff1_out;
        Tensor2D ff1_act;
        Tensor2D ff2_out;
    };
    std::vector<Cache> caches((size_t)model.n_layers);

    // prefer GPU for linears; fallback to CPU if the kernel declines
    auto LforwardGPU = [&](const LinearLayer& L, const Tensor2D& Xin) -> Tensor2D {
        Tensor2D Y;
        // If resident weights are enabled and present for this linear, bind them and skip W/B uploads.
        if (mGpu && mGPUResident.enabled) {
            auto it = mGPUResident.map.find(&L);
            if (it != mGPUResident.map.end()) {
                const auto& wb = it->second;
                // Bind resident buffers to expected bindings (1=W, 2=B)
                mGpu->adoptSSBO("W", wb.w, wb.wBytes, 1);
                mGpu->adoptSSBO("B", wb.b, wb.bBytes, 2);
                if (gpu_linear_forward_resident(mGpu, Xin, wb.IN, wb.OUT, Y)) {
                    return Y;
                }
            }
        }
        // Fallback: upload W/B ad-hoc for this call
        if (gpu_linear_forward(mGpu, Xin, L, Y)) {
            return Y;
        }
        // CPU fallback
        return L.Forward(Xin);
    };

    Tensor2D x = x0;
    for (int l = 0; l < model.n_layers; l++) {
        const TransformerBlock& B = model.layers[(size_t)l];
        Cache& C = caches[(size_t)l];

        C.x_in = x;
        C.n1 = B.ln1.Forward(C.x_in); // pre-norm

        // attention projections on GPU when possible
        C.Q = LforwardGPU(B.attn.Wq, C.n1);
        C.K = LforwardGPU(B.attn.Wk, C.n1);
        C.V = LforwardGPU(B.attn.Wv, C.n1);

        // causal attention on CPU (loop)  safe and deterministic for now
        C.attn_concat = Tensor2D(C.n1.R, model.d_model);
        C.attn_concat.Zero();
        float scale = 1.0f / std::sqrt((float)B.attn.d_head);

        for (int h = 0; h < B.attn.n_heads; h++) {
            int off = h * B.attn.d_head;
            for (int t = 0; t < C.n1.R; t++) {
                float maxlogit = -std::numeric_limits<float>::infinity();
                for (int u = 0; u <= t; u++) {
                    float dot = 0.0f;
                    const float* q = &C.Q.data[(size_t)t * C.Q.C + off];
                    const float* k = &C.K.data[(size_t)u * C.K.C + off];
                    for (int c = 0; c < B.attn.d_head; c++) dot += q[c] * k[c];
                    float logit = dot * scale;
                    if (logit > maxlogit) maxlogit = logit;
                }
                float denom = 0.0f;
                std::vector<float> w((size_t)t + 1, 0.0f);
                for (int u = 0; u <= t; u++) {
                    float dot = 0.0f;
                    const float* q = &C.Q.data[(size_t)t * C.Q.C + off];
                    const float* k = &C.K.data[(size_t)u * C.K.C + off];
                    for (int c = 0; c < B.attn.d_head; c++) dot += q[c] * k[c];
                    float logit = dot * scale;
                    float e = std::exp(logit - maxlogit);
                    w[(size_t)u] = e;
                    denom += e;
                }
                float invden = 1.0f / denom;
                float* out_row = C.attn_concat.Row(t);
                for (int u = 0; u <= t; u++) {
                    float ww = w[(size_t)u] * invden;
                    const float* v = &C.V.data[(size_t)u * C.V.C + off];
                    for (int c = 0; c < B.attn.d_head; c++) {
                        out_row[off + c] += ww * v[c];
                    }
                }
            }
        }

        // Wo on GPU, then residual
        C.attn_out     = LforwardGPU(B.attn.Wo, C.attn_concat);
        C.x_attn_res   = C.x_in;
        AddInPlace(C.x_attn_res, C.attn_out);

        // mlp (fc1/act/fc2); fc1/fc2 try GPU
        C.n2      = B.ln2.Forward(C.x_attn_res);
        C.ff1_out = LforwardGPU(B.ffn.fc1, C.n2);
        C.ff1_act = Activation.Forward(C.ff1_out);
        C.ff2_out = LforwardGPU(B.ffn.fc2, C.ff1_act);

        x = C.x_attn_res;
        AddInPlace(x, C.ff2_out);
    }

    // lm head on GPU if possible
    Tensor2D logits;
    {
        Tensor2D tmp;
        if (gpu_linear_forward(mGpu, x, model.lm_head, tmp)) {
            logits = std::move(tmp);
        } else {
            logits = model.lm_head.Forward(x);
        }
    }

    // ensure GPU writes are visible before CPU consumes results (conservative)
    /* sync skipped: downloads already synchronize */

    float loss = CrossEntropyLoss(logits, targets, pad_id);

    // ---------- backward (CPU) ----------
    int T = logits.R;
    int V = logits.C;

    Tensor2D dlogits(T, V);
    for (int t = 0; t < T; t++) {
        int y = targets[(size_t)t];
        if (y == pad_id) {
            std::fill(dlogits.Row(t), dlogits.Row(t) + V, 0.0f);
            continue;
        }
        const float* row = logits.Row(t);
        float maxv = -std::numeric_limits<float>::infinity();
        for (int v = 0; v < V; v++) if (row[v] > maxv) maxv = row[v];
        float denom = 0.0f;
        for (int v = 0; v < V; v++) denom += std::exp(row[v] - maxv);
        float invden = 1.0f / denom;
        float* dlr = dlogits.Row(t);
        for (int v = 0; v < V; v++) {
            float p = std::exp(row[v] - maxv) * invden;
            dlr[v] = p - ((v == y) ? 1.0f : 0.0f);
        }
    }
    int count = 0;
    for (int t = 0; t < T; t++) if (targets[(size_t)t] != pad_id) count++;
    if (count > 0) {
        float inv = 1.0f / (float)count;
        for (size_t i = 0; i < dlogits.data.size(); i++) dlogits.data[i] *= inv;
    }

    Tensor2D dW_lm, dx_last;
    std::vector<float> db_lm;
    LinearBackward(x, model.lm_head, dlogits, dW_lm, db_lm, dx_last);

    Tensor2D dx = dx_last;

    Tensor2D d_tokW(model.tok.W.R, model.tok.W.C); std::fill(d_tokW.data.begin(), d_tokW.data.end(), 0.0f);
    Tensor2D d_posP(model.pos.P.R, model.pos.P.C); std::fill(d_posP.data.begin(), d_posP.data.end(), 0.0f);

    for (int l = model.n_layers - 1; l >= 0; l--) {
        TransformerBlock& B = model.layers[(size_t)l];
        Cache& C = caches[(size_t)l];

        Tensor2D d_ff2_out = dx;
        Tensor2D d_x_attn_res = dx;

        Tensor2D dW_fc2, dx_fc2;
        std::vector<float> db_fc2;
        LinearBackward(C.ff1_act, B.ffn.fc2, d_ff2_out, dW_fc2, db_fc2, dx_fc2);

        Tensor2D d_ff1_out;
        Activation.Backward(C.ff1_out, dx_fc2, d_ff1_out);

        Tensor2D dW_fc1, dx_fc1;
        std::vector<float> db_fc1;
        LinearBackward(C.n2, B.ffn.fc1, d_ff1_out, dW_fc1, db_fc1, dx_fc1);

        Tensor2D d_n2 = dx_fc1;

        std::vector<float> dgamma2, dbeta2;
        Tensor2D d_x_attn_res_from_ln2;
        LayerNormBackward(C.x_attn_res, B.ln2, d_n2, dgamma2, dbeta2, d_x_attn_res_from_ln2);

        for (size_t i = 0; i < d_x_attn_res.data.size(); i++) d_x_attn_res.data[i] += d_x_attn_res_from_ln2.data[i];

        Tensor2D d_attn_out = d_x_attn_res;
        Tensor2D d_x_in_after_attn = d_x_attn_res;

        Tensor2D dWo, dx_attn_concat;
        std::vector<float> dbo;
        LinearBackward(C.attn_concat, B.attn.Wo, d_attn_out, dWo, dbo, dx_attn_concat);

        Tensor2D dQ(C.Q.R, C.Q.C); std::fill(dQ.data.begin(), dQ.data.end(), 0.0f);
        Tensor2D dK(C.K.R, C.K.C); std::fill(dK.data.begin(), dK.data.end(), 0.0f);
        Tensor2D dV(C.V.R, C.V.C); std::fill(dV.data.begin(), dV.data.end(), 0.0f);

        float scale = 1.0f / std::sqrt((float)B.attn.d_head);
        for (int h = 0; h < B.attn.n_heads; h++) {
            int off = h * B.attn.d_head;
            for (int t = 0; t < C.n1.R; t++) {
                float maxlogit = -std::numeric_limits<float>::infinity();
                for (int u = 0; u <= t; u++) {
                    float dot = 0.0f;
                    const float* q = &C.Q.data[(size_t)t * C.Q.C + off];
                    const float* k = &C.K.data[(size_t)u * C.K.C + off];
                    for (int c = 0; c < B.attn.d_head; c++) dot += q[c] * k[c];
                    float logit = dot * scale;
                    if (logit > maxlogit) maxlogit = logit;
                }
                float denom = 0.0f;
                std::vector<float> p((size_t)t + 1, 0.0f);
                for (int u = 0; u <= t; u++) {
                    float dot = 0.0f;
                    const float* q = &C.Q.data[(size_t)t * C.Q.C + off];
                    const float* k = &C.K.data[(size_t)u * C.K.C + off];
                    for (int c = 0; c < B.attn.d_head; c++) dot += q[c] * k[c];
                    float logit = dot * scale;
                    float e = std::exp(logit - maxlogit);
                    p[(size_t)u] = e;
                    denom += e;
                }
                float invden = 1.0f / denom;
                for (int u = 0; u <= t; u++) p[(size_t)u] *= invden;

                const float* gout = &dx_attn_concat.data[(size_t)t * dx_attn_concat.C + off];

                for (int u = 0; u <= t; u++) {
                    float w = p[(size_t)u];
                    float* dVrow = &dV.data[(size_t)u * dV.C + off];
                    for (int c = 0; c < B.attn.d_head; c++) dVrow[c] += w * gout[c];
                }

                std::vector<float> g_s((size_t)t + 1, 0.0f);
                for (int u = 0; u <= t; u++) {
                    const float* vrow = &C.V.data[(size_t)u * C.V.C + off];
                    float s = 0.0f;
                    for (int c = 0; c < B.attn.d_head; c++) s += gout[c] * vrow[c];
                    g_s[(size_t)u] = s;
                }
                float sum_ps = 0.0f;
                for (int u = 0; u <= t; u++) sum_ps += p[(size_t)u] * g_s[(size_t)u];

                for (int u = 0; u <= t; u++) {
                    float gz = p[(size_t)u] * (g_s[(size_t)u] - sum_ps);
                    float* dQt = &dQ.data[(size_t)t * dQ.C + off];
                    const float* Ku = &C.K.data[(size_t)u * C.K.C + off];
                    for (int c = 0; c < B.attn.d_head; c++) dQt[c] += scale * gz * Ku[c];

                    float* dKu = &dK.data[(size_t)u * dK.C + off];
                    const float* Qt = &C.Q.data[(size_t)t * C.Q.C + off];
                    for (int c = 0; c < B.attn.d_head; c++) dKu[c] += scale * gz * Qt[c];
                }
            }
        }

        Tensor2D dWq, dWk, dWv, dx_q, dx_k, dx_v;
        std::vector<float> dbq, dbk, dbv;
        LinearBackward(C.n1, B.attn.Wq, dQ, dWq, dbq, dx_q);
        LinearBackward(C.n1, B.attn.Wk, dK, dWk, dbk, dx_k);
        LinearBackward(C.n1, B.attn.Wv, dV, dWv, dbv, dx_v);

        Tensor2D d_n1(dx_q.R, dx_q.C);
        for (int i = 0; i < d_n1.R * d_n1.C; i++) {
            d_n1.data[(size_t)i] = dx_q.data[(size_t)i] + dx_k.data[(size_t)i] + dx_v.data[(size_t)i];
        }

        std::vector<float> dgamma1, dbeta1;
        Tensor2D dx_ln1;
        LayerNormBackward(C.x_in, B.ln1, d_n1, dgamma1, dbeta1, dx_ln1);

        Tensor2D dx_next = d_x_in_after_attn;
        for (int i = 0; i < dx_next.R * dx_next.C; i++) dx_next.data[(size_t)i] += dx_ln1.data[(size_t)i];

        if (acc) {
            GradientAccumulator::BlockGrads& GG = acc->layers[(size_t)l];
            for (size_t i=0;i<GG.dWo.data.size();++i) GG.dWo.data[i] += dWo.data[i];
            for (size_t i=0;i<GG.dbo.size();++i)      GG.dbo[i]      += dbo[i];

            for (size_t i=0;i<GG.dWq.data.size();++i) GG.dWq.data[i] += dWq.data[i];
            for (size_t i=0;i<GG.dbq.size();++i)      GG.dbq[i]      += dbq[i];
            for (size_t i=0;i<GG.dWk.data.size();++i) GG.dWk.data[i] += dWk.data[i];
            for (size_t i=0;i<GG.dbk.size();++i)      GG.dbk[i]      += dbk[i];
            for (size_t i=0;i<GG.dWv.data.size();++i) GG.dWv.data[i] += dWv.data[i];
            for (size_t i=0;i<GG.dbv.size();++i)      GG.dbv[i]      += dbv[i];

            for (size_t i=0;i<GG.d_fc1W.data.size();++i) GG.d_fc1W.data[i] += dW_fc1.data[i];
            for (size_t i=0;i<GG.d_fc1b.size();++i)      GG.d_fc1b[i]      += db_fc1[i];
            for (size_t i=0;i<GG.d_fc2W.data.size();++i) GG.d_fc2W.data[i] += dW_fc2.data[i];
            for (size_t i=0;i<GG.d_fc2b.size();++i)      GG.d_fc2b[i]      += db_fc2[i];

            for (size_t i=0;i<GG.d_ln1g.size();++i) GG.d_ln1g[i] += dgamma1[i];
            for (size_t i=0;i<GG.d_ln1b.size();++i) GG.d_ln1b[i] += dbeta1[i];
            for (size_t i=0;i<GG.d_ln2g.size();++i) GG.d_ln2g[i] += dgamma2[i];
            for (size_t i=0;i<GG.d_ln2b.size();++i) GG.d_ln2b[i] += dbeta2[i];
        }

        if (apply_updates) {
            opt.t += 1;

            Adam::StepInPlace(B.attn.Wo.W.data, dWo.data, layer_states[(size_t)l].WoW_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.attn.Wo.b, dbo, layer_states[(size_t)l].Wob_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);

            Adam::StepInPlace(B.attn.Wq.W.data, dWq.data, layer_states[(size_t)l].WqW_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.attn.Wq.b, dbq, layer_states[(size_t)l].Wqb_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.attn.Wk.W.data, dWk.data, layer_states[(size_t)l].WkW_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.attn.Wk.b, dbk, layer_states[(size_t)l].Wkb_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.attn.Wv.W.data, dWv.data, layer_states[(size_t)l].WvW_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.attn.Wv.b, dbv, layer_states[(size_t)l].Wvb_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);

            Adam::StepInPlace(B.ffn.fc1.W.data, dW_fc1.data, layer_states[(size_t)l].fc1W_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.ffn.fc1.b, db_fc1, layer_states[(size_t)l].fc1b_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.ffn.fc2.W.data, dW_fc2.data, layer_states[(size_t)l].fc2W_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.ffn.fc2.b, db_fc2, layer_states[(size_t)l].fc2b_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);

            Adam::StepInPlace(B.ln1.gamma, dgamma1, layer_states[(size_t)l].ln1g_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.ln1.beta,  dbeta1, layer_states[(size_t)l].ln1b_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.ln2.gamma, dgamma2, layer_states[(size_t)l].ln2g_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
            Adam::StepInPlace(B.ln2.beta,  dbeta2, layer_states[(size_t)l].ln2b_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
        }

        dx = dx_next;
    }

    // embeddings & positional
    int limit = (x0.R <= model.pos.P.R) ? x0.R : model.pos.P.R;
    for (int t = 0; t < limit; t++) {
        float* dprow = &d_posP.data[(size_t)t * d_posP.C];
        const float* dxrow = dx.Row(t);
        for (int c = 0; c < d_posP.C; c++) dprow[c] += dxrow[c];
    }
    for (int t = 0; t < (int)inputs.size(); t++) {
        int id = inputs[(size_t)t];
        if (id < 0 || id >= model.tok.W.R) continue;
        const float* dxrow = dx.Row(t);
        for (int c = 0; c < model.tok.W.C; c++) d_tokW.data[(size_t)id * model.tok.W.C + c] += dxrow[c];
    }

    if (acc) {
        for (size_t i=0;i<acc->d_lmW.data.size();++i) acc->d_lmW.data[i] += dW_lm.data[i];
        for (size_t i=0;i<acc->d_lmb.size();++i)      acc->d_lmb[i]      += db_lm[i];
        for (size_t i=0;i<acc->d_posP.data.size();++i) acc->d_posP.data[i] += d_posP.data[i];
        for (size_t i=0;i<acc->d_tokW.data.size();++i) acc->d_tokW.data[i] += d_tokW.data[i];
    }

    if (apply_updates) {
        opt.t += 1;
        Adam::StepInPlace(model.lm_head.W.data, dW_lm.data, lmW_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
        Adam::StepInPlace(model.lm_head.b,      db_lm,      lmb_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
        Adam::StepInPlace(model.pos.P.data,     d_posP.data, posP_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
        Adam::StepInPlace(model.tok.W.data,     d_tokW.data, tokW_s, opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t);
    }

    return loss;
}


void NeuralNetwork::ApplyGradients(LauguageModel& model, const GradientAccumulator& G, float scale) {
    auto apply_vec = [&](std::vector<float>& w, const std::vector<float>& g, AdamState& s) {
        std::vector<float> gs(g.size());
        for (size_t i=0;i<g.size();++i) gs[i] = g[i] * scale;
        Adam::StepInPlace(w, gs, s, opt.lr, opt.beta1, opt.beta2, opt.eps, ++opt.t);
    };
    auto apply_tensor = [&](std::vector<float>& w, const std::vector<float>& g, AdamState& s) {
        std::vector<float> gs(g.size());
        for (size_t i=0;i<g.size();++i) gs[i] = g[i] * scale;
        Adam::StepInPlace(w, gs, s, opt.lr, opt.beta1, opt.beta2, opt.eps, ++opt.t);
    };

    // Top-level
    apply_tensor(model.lm_head.W.data, G.d_lmW.data, lmW_s);
    apply_vec(model.lm_head.b, G.d_lmb, lmb_s);
    apply_tensor(model.pos.P.data, G.d_posP.data, posP_s);
    apply_tensor(model.tok.W.data, G.d_tokW.data, tokW_s);

    // Per-layer
    if ((int)layer_states.size() != model.n_layers) layer_states.resize((size_t)model.n_layers);
    for (int l=0; l<model.n_layers; ++l) {
        TransformerBlock& B = model.layers[(size_t)l];
        const GradientAccumulator::BlockGrads& GG = G.layers[(size_t)l];
        NeuralNetwork::BlockStates& S = layer_states[(size_t)l];

        // attn
        apply_tensor(B.attn.Wo.W.data, GG.dWo.data, S.WoW_s);
        apply_vec(B.attn.Wo.b, GG.dbo, S.Wob_s);

        apply_tensor(B.attn.Wq.W.data, GG.dWq.data, S.WqW_s);
        apply_vec(B.attn.Wq.b, GG.dbq, S.Wqb_s);
        apply_tensor(B.attn.Wk.W.data, GG.dWk.data, S.WkW_s);
        apply_vec(B.attn.Wk.b, GG.dbk, S.Wkb_s);
        apply_tensor(B.attn.Wv.W.data, GG.dWv.data, S.WvW_s);
        apply_vec(B.attn.Wv.b, GG.dbv, S.Wvb_s);

        // ffn
        apply_tensor(B.ffn.fc1.W.data, GG.d_fc1W.data, S.fc1W_s);
        apply_vec(B.ffn.fc1.b, GG.d_fc1b, S.fc1b_s);
        apply_tensor(B.ffn.fc2.W.data, GG.d_fc2W.data, S.fc2W_s);
        apply_vec(B.ffn.fc2.b, GG.d_fc2b, S.fc2b_s);

        // norms
        {
            std::vector<float> gs = GG.d_ln1g; for (size_t i=0;i<gs.size();++i) gs[i] *= scale;
            Adam::StepInPlace(B.ln1.gamma, gs, S.ln1g_s, opt.lr, opt.beta1, opt.beta2, opt.eps, ++opt.t);
        }
        {
            std::vector<float> gs = GG.d_ln1b; for (size_t i=0;i<gs.size();++i) gs[i] *= scale;
            Adam::StepInPlace(B.ln1.beta, gs, S.ln1b_s, opt.lr, opt.beta1, opt.beta2, opt.eps, ++opt.t);
        }
        {
            std::vector<float> gs = GG.d_ln2g; for (size_t i=0;i<gs.size();++i) gs[i] *= scale;
            Adam::StepInPlace(B.ln2.gamma, gs, S.ln2g_s, opt.lr, opt.beta1, opt.beta2, opt.eps, ++opt.t);
        }
        {
            std::vector<float> gs = GG.d_ln2b; for (size_t i=0;i<gs.size();++i) gs[i] *= scale;
            Adam::StepInPlace(B.ln2.beta, gs, S.ln2b_s, opt.lr, opt.beta1, opt.beta2, opt.eps, ++opt.t);
        }
    }
    // Keep resident GPU buffers in sync with CPU weights after optimizer step
    if (mGpu && mGPUResident.enabled) { RefreshGPUWeightsFromModel(model); }

}

void NeuralNetwork::UseGPU(ShaderTensor* gpu) { mGpu = gpu; }


// Build required shaders.
void NeuralNetwork::BuildShaders() {
    
    static const std::string kMatMulBiasCS = R"(#version 430
    layout(std430, binding=0) buffer XBuf { float X[]; };
    layout(std430, binding=1) buffer WBuf { float W[]; };
    layout(std430, binding=2) buffer BBuf { float B[]; };
    layout(std430, binding=3) buffer YBuf { float Y[]; };
    layout(std430, binding=4) buffer Meta { int T; int IN; int OUT; };
    
    layout(local_size_x=16, local_size_y=16, local_size_z=1) in;
    
    shared float sX[16][16];
    shared float sW[16][16];
    
    void main() {
        uint col = gl_GlobalInvocationID.x;
        uint row = gl_GlobalInvocationID.y;
    
        float sum = 0.0;
    
        uint tiles = uint((IN + 15) / 16);
        for (uint tile = 0u; tile < tiles; ++tile) {
            uint kx = tile * 16u + gl_LocalInvocationID.x;
            uint ky = tile * 16u + gl_LocalInvocationID.y;
    
            // load a 16x16 tile of X (row-major) and W (row-major) with bounds checks
            if (row < uint(T) && kx < uint(IN)) {
                sX[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = X[int(row) * IN + int(kx)];
            } else {
                sX[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 0.0;
            }
    
            if (ky < uint(IN) && col < uint(OUT)) {
                sW[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = W[int(ky) * OUT + int(col)];
            } else {
                sW[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 0.0;
            }
    
            barrier();
    
            // 16-wide inner product from the cached tiles
            for (uint k = 0u; k < 16u; ++k) {
                sum += sX[gl_LocalInvocationID.y][k] * sW[k][gl_LocalInvocationID.x];
            }
    
            barrier();
        }
    
        if (row < uint(T) && col < uint(OUT)) {
            Y[int(row) * OUT + int(col)] = sum + B[int(col)];
        }
    }
    )";
    
    if (!mGpu || g_matmul_built) return;
    if (mGpu->buildComputeFromSource(kMatMulBiasCS.c_str(), "NN_MatMulBias")) {
        g_matmul_built = true;
    }
}
void NeuralNetwork::EnableResidentWeights(bool enable) {
    mGPUResident.enabled = enable;
}

void NeuralNetwork::ReleaseGPUWeights() {
    if (!mGpu) { mGPUResident.map.clear(); mGPUResident.enabled=false; return; }
    // We only drop our bookkeeping; actual GL buffers are owned by ShaderTensor's buffer map.
    mGPUResident.map.clear();
    mGPUResident.enabled = false;
}


static void UploadOneLinear(ShaderTensor* gpu, const LinearLayer& L,
                            NeuralNetwork::GPUResident::WeightBuf& out) {
    const int IN = L.W.R;
    const int OUT = L.W.C;
    std::ptrdiff_t wBytes = (std::ptrdiff_t)IN * OUT * sizeof(float);
    std::ptrdiff_t bBytes = (std::ptrdiff_t)OUT * sizeof(float);

    // Use unique names per LinearLayer address so ShaderTensor tracks them separately
    uintptr_t key = reinterpret_cast<uintptr_t>(&L);
    std::string wName = "W_resident_" + std::to_string(key);
    std::string bName = "B_resident_" + std::to_string(key);

    unsigned wid = gpu->createSSBO(wName, wBytes, /*binding*/99, 0, L.W.data.data());
    unsigned bid = gpu->createSSBO(bName, bBytes, /*binding*/99, 0, L.b.data());

    out.w = wid; out.b = bid; out.wBytes = wBytes; out.bBytes = bBytes; out.IN = IN; out.OUT = OUT;
}


void NeuralNetwork::UploadWeightsToGPU(LauguageModel& model) {
    if (!mGpu) return;
    mGPUResident.map.clear();

    auto add = [&](const LinearLayer& L){
        GPUResident::WeightBuf wb;
        UploadOneLinear(mGpu, L, wb);
        mGPUResident.map[&L] = wb;
    };

    // per-layer linears
    for (int l=0; l<model.n_layers; ++l) {
        const TransformerBlock& B = model.layers[(size_t)l];
        add(B.attn.Wq);
        add(B.attn.Wk);
        add(B.attn.Wv);
        add(B.attn.Wo);
        add(B.ffn.fc1);
        add(B.ffn.fc2);
    }
    // output head
    add(model.lm_head);

    mGPUResident.enabled = true;
}

void NeuralNetwork::RefreshGPUWeightsFromModel(const LauguageModel& model) {
    if (!mGpu) return;
    if (!mGPUResident.enabled) return;

    auto update = [&](const LinearLayer& L){
        auto it = mGPUResident.map.find(&L);
        if (it == mGPUResident.map.end()) return;
        auto& wb = it->second;
        if (wb.w && wb.wBytes) mGpu->uploadRawSSBO(wb.w, 0, L.W.data.data(), wb.wBytes);
        if (wb.b && wb.bBytes) mGpu->uploadRawSSBO(wb.b, 0, L.b.data(), wb.bBytes);
    };

    for (int l=0; l<model.n_layers; ++l) {
        const TransformerBlock& B = model.layers[(size_t)l];
        update(B.attn.Wq);
        update(B.attn.Wk);
        update(B.attn.Wv);
        update(B.attn.Wo);
        update(B.ffn.fc1);
        update(B.ffn.fc2);
    }
    update(model.lm_head);
}

