#include "GradientAccumulator.h"

void GradientAccumulator::InitLike(const LanguageModel& model) {
    d_tokW = Tensor2D(model.tok.W.R, model.tok.W.C); std::fill(d_tokW.data.begin(), d_tokW.data.end(), 0.0f);
    d_posP = Tensor2D(model.pos.P.R, model.pos.P.C); std::fill(d_posP.data.begin(), d_posP.data.end(), 0.0f);
    d_lmW  = Tensor2D(model.lm_head.W.R, model.lm_head.W.C); std::fill(d_lmW.data.begin(), d_lmW.data.end(), 0.0f);
    d_lmb.assign(model.lm_head.b.size(), 0.0f);
    
    layers.resize((size_t)model.n_layers);
    for (int l = 0; l < model.n_layers; ++l) {
        BlockGrads& G = layers[(size_t)l];
        G.d_ln1g.assign((size_t)model.d_model, 0.0f);
        G.d_ln1b.assign((size_t)model.d_model, 0.0f);
        G.d_ln2g.assign((size_t)model.d_model, 0.0f);
        G.d_ln2b.assign((size_t)model.d_model, 0.0f);
        
        G.dWq = Tensor2D(model.d_model, model.d_model); std::fill(G.dWq.data.begin(), G.dWq.data.end(), 0.0f);
        G.dbq.assign((size_t)model.d_model, 0.0f);
        G.dWk = Tensor2D(model.d_model, model.d_model); std::fill(G.dWk.data.begin(), G.dWk.data.end(), 0.0f);
        G.dbk.assign((size_t)model.d_model, 0.0f);
        G.dWv = Tensor2D(model.d_model, model.d_model); std::fill(G.dWv.data.begin(), G.dWv.data.end(), 0.0f);
        G.dbv.assign((size_t)model.d_model, 0.0f);
        G.dWo = Tensor2D(model.d_model, model.d_model); std::fill(G.dWo.data.begin(), G.dWo.data.end(), 0.0f);
        G.dbo.assign((size_t)model.d_model, 0.0f);
        
        G.d_fc1W = Tensor2D(model.d_model, Activation.d_ff_mul * model.d_ff); std::fill(G.d_fc1W.data.begin(), G.d_fc1W.data.end(), 0.0f);
        G.d_fc1b.assign((size_t)(Activation.d_ff_mul * model.d_ff), 0.0f);
        G.d_fc2W = Tensor2D(model.d_ff, model.d_model); std::fill(G.d_fc2W.data.begin(), G.d_fc2W.data.end(), 0.0f);
        G.d_fc2b.assign((size_t)model.d_model, 0.0f);
    }
}

void GradientAccumulator::Add(const GradientAccumulator& other) {
    for (size_t i = 0; i < d_tokW.data.size(); ++i) d_tokW.data[i] += other.d_tokW.data[i];
    for (size_t i = 0; i < d_posP.data.size(); ++i) d_posP.data[i] += other.d_posP.data[i];
    for (size_t i = 0; i < d_lmW.data.size();  ++i) d_lmW.data[i]  += other.d_lmW.data[i];
    for (size_t i = 0; i < d_lmb.size();       ++i) d_lmb[i]       += other.d_lmb[i];
    
    size_t L = layers.size();
    for (size_t l = 0; l < L; ++l) {
        const BlockGrads& A = other.layers[l];
        BlockGrads& B = layers[l];
        for (size_t i=0;i<B.d_ln1g.size();++i) B.d_ln1g[i] += A.d_ln1g[i];
        for (size_t i=0;i<B.d_ln1b.size();++i) B.d_ln1b[i] += A.d_ln1b[i];
        for (size_t i=0;i<B.d_ln2g.size();++i) B.d_ln2g[i] += A.d_ln2g[i];
        for (size_t i=0;i<B.d_ln2b.size();++i) B.d_ln2b[i] += A.d_ln2b[i];
        
        for (size_t i=0;i<B.dWq.data.size();++i) B.dWq.data[i] += A.dWq.data[i];
        for (size_t i=0;i<B.dbq.size();++i)      B.dbq[i]      += A.dbq[i];
        for (size_t i=0;i<B.dWk.data.size();++i) B.dWk.data[i] += A.dWk.data[i];
        for (size_t i=0;i<B.dbk.size();++i)      B.dbk[i]      += A.dbk[i];
        for (size_t i=0;i<B.dWv.data.size();++i) B.dWv.data[i] += A.dWv.data[i];
        for (size_t i=0;i<B.dbv.size();++i)      B.dbv[i]      += A.dbv[i];
        for (size_t i=0;i<B.dWo.data.size();++i) B.dWo.data[i] += A.dWo.data[i];
        for (size_t i=0;i<B.dbo.size();++i)      B.dbo[i]      += A.dbo[i];
        
        for (size_t i=0;i<B.d_fc1W.data.size();++i) B.d_fc1W.data[i] += A.d_fc1W.data[i];
        for (size_t i=0;i<B.d_fc1b.size();++i)      B.d_fc1b[i]      += A.d_fc1b[i];
        for (size_t i=0;i<B.d_fc2W.data.size();++i) B.d_fc2W.data[i] += A.d_fc2W.data[i];
        for (size_t i=0;i<B.d_fc2b.size();++i)      B.d_fc2b[i]      += A.d_fc2b[i];
    }
}

void GradientAccumulator::Scale(float s) {
    if (s == 1.0f) return;
    for (size_t i = 0; i < d_tokW.data.size(); ++i) d_tokW.data[i] *= s;
    for (size_t i = 0; i < d_posP.data.size(); ++i) d_posP.data[i] *= s;
    for (size_t i = 0; i < d_lmW.data.size();  ++i) d_lmW.data[i]  *= s;
    for (size_t i = 0; i < d_lmb.size();       ++i) d_lmb[i]       *= s;
    
    for (size_t l=0;l<layers.size();++l) {
        BlockGrads& B = layers[l];
        for (size_t i=0;i<B.d_ln1g.size();++i) B.d_ln1g[i] *= s;
        for (size_t i=0;i<B.d_ln1b.size();++i) B.d_ln1b[i] *= s;
        for (size_t i=0;i<B.d_ln2g.size();++i) B.d_ln2g[i] *= s;
        for (size_t i=0;i<B.d_ln2b.size();++i) B.d_ln2b[i] *= s;
        
        for (size_t i=0;i<B.dWq.data.size();++i) B.dWq.data[i] *= s;
        for (size_t i=0;i<B.dbq.size();++i)      B.dbq[i]      *= s;
        for (size_t i=0;i<B.dWk.data.size();++i) B.dWk.data[i] *= s;
        for (size_t i=0;i<B.dbk.size();++i)      B.dbk[i]      *= s;
        for (size_t i=0;i<B.dWv.data.size();++i) B.dWv.data[i] *= s;
        for (size_t i=0;i<B.dbv.size();++i)      B.dbv[i]      *= s;
        for (size_t i=0;i<B.dWo.data.size();++i) B.dWo.data[i] *= s;
        for (size_t i=0;i<B.dbo.size();++i)      B.dbo[i]      *= s;
        
        for (size_t i=0;i<B.d_fc1W.data.size();++i) B.d_fc1W.data[i] *= s;
        for (size_t i=0;i<B.d_fc1b.size();++i)      B.d_fc1b[i]      *= s;
        for (size_t i=0;i<B.d_fc2W.data.size();++i) B.d_fc2W.data[i] *= s;
        for (size_t i=0;i<B.d_fc2b.size();++i)      B.d_fc2b[i]      *= s;
    }
}
