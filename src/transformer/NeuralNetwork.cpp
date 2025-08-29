#include "NeuralNetwork.h"
#include "CrossEntropyLoss.h"

void NeuralNetwork::InitScratch(int max_T) {
    attnScratch_p.resize((size_t)max_T);
    attnScratch_gs.resize((size_t)max_T);
}

NeuralNetwork::NeuralNetwork(float lr) : opt(lr) {}

void NeuralNetwork::LinearBackward(const Tensor2D& x, const LinearLayer& lin, const Tensor2D& dy, Tensor2D& dW, std::vector<float>& db, Tensor2D& dx) {
    // dW = x^T @ dy ; db = sum_rows(dy) ; dx = dy @ W^T
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
    // dx = dy @ W^T
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
        // mean
        float mean = 0.0f;
        for (int c = 0; c < C; c++) mean += xr[c];
        mean /= (float)C;
        // var
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
float NeuralNetwork::Step(TransformerLauguageModel& model, const std::vector<int>& inputs, const std::vector<int>& targets, 
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
        C.n1 = B.ln1.Forward(C.x_in);
        // Attention forward with explicit Q/K/V and concat cache
        C.Q = B.attn.Wq.Forward(C.n1);
        C.K = B.attn.Wk.Forward(C.n1);
        C.V = B.attn.Wv.Forward(C.n1);

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
        // After heads: Wo
        C.attn_out = B.attn.Wo.Forward(C.attn_concat);
        C.x_attn_res = C.x_in;
        AddInPlace(C.x_attn_res, C.attn_out);

        C.n2 = B.ln2.Forward(C.x_attn_res);
        C.ff1_out = B.ffn.fc1.Forward(C.n2);
        C.ff1_act = C.ff1_out; // duplicate then GELU
        GELU_InPlace(C.ff1_act);
        C.ff2_out = B.ffn.fc2.Forward(C.ff1_act);

        x = C.x_attn_res;
        AddInPlace(x, C.ff2_out);
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
        Tensor2D dW_fc2, db_fc2_dx;
        std::vector<float> db_fc2;
        Tensor2D dx_fc2;
        LinearBackward(C.ff1_act, B.ffn.fc2, d_ff2_out, dW_fc2, db_fc2, dx_fc2);

        // Back through GELU and fc1
        Tensor2D d_ff1_act = dx_fc2;
        Tensor2D d_ff1_out;
        GELU_Backward(C.ff1_out, d_ff1_act, d_ff1_out);

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
        //float* row = model.tok.W.Row(id);
        const float* dxrow = dx.Row(t);
        // for Adam we need gradient storage separately
        // accumulate into temp tensor
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

void NeuralNetwork::ApplyGradients(TransformerLauguageModel& model, const GradientAccumulator& G, float scale) {
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
}
