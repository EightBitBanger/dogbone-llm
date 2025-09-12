#include "test.h"
#include "../Transformer/Transformer.h"

#include <iostream>
#include <cmath>
#include <cstdint>

bool TestGradientAccumulator() {
    GradientAccumulator A, B;
    
    // ---- Top-level tensors/vectors ----
    A.d_tokW = Tensor2D(2, 3); std::fill(A.d_tokW.data.begin(), A.d_tokW.data.end(), 1.0f);
    B.d_tokW = Tensor2D(2, 3); std::fill(B.d_tokW.data.begin(), B.d_tokW.data.end(), 2.0f);
    
    A.d_posP = Tensor2D(2, 4); std::fill(A.d_posP.data.begin(), A.d_posP.data.end(), 1.0f);
    B.d_posP = Tensor2D(2, 4); std::fill(B.d_posP.data.begin(), B.d_posP.data.end(), 2.0f);
    
    A.d_lmW  = Tensor2D(3, 2); std::fill(A.d_lmW.data.begin(),  A.d_lmW.data.end(),  1.0f);
    B.d_lmW  = Tensor2D(3, 2); std::fill(B.d_lmW.data.begin(),  B.d_lmW.data.end(),  2.0f);
    
    A.d_lmb.assign(2, 1.0f);
    B.d_lmb.assign(2, 2.0f);
    
    // ---- Per-layer grads (make two layers; arbitrary but consistent sizes) ----
    const int M = 4;     // d_model-like width for vectors and square mats
    const int K = 6;     // fc1 width
    const int K2 = 3;    // fc2 width
    const size_t L = 2;  // number of layers
    
    A.layers.resize(L);
    B.layers.resize(L);
    for (size_t l = 0; l < L; ++l) {
        auto &a = A.layers[l], &b = B.layers[l];
        
        // LayerNorm grads
        a.d_ln1g.assign(M, 1.0f); b.d_ln1g.assign(M, 2.0f);
        a.d_ln1b.assign(M, 1.0f); b.d_ln1b.assign(M, 2.0f);
        a.d_ln2g.assign(M, 1.0f); b.d_ln2g.assign(M, 2.0f);
        a.d_ln2b.assign(M, 1.0f); b.d_ln2b.assign(M, 2.0f);
        
        // Attention linear layers
        a.dWq = Tensor2D(M, M); std::fill(a.dWq.data.begin(), a.dWq.data.end(), 1.0f);
        b.dWq = Tensor2D(M, M); std::fill(b.dWq.data.begin(), b.dWq.data.end(), 2.0f);
        a.dbq.assign(M, 1.0f);  b.dbq.assign(M, 2.0f);
        
        a.dWk = Tensor2D(M, M); std::fill(a.dWk.data.begin(), a.dWk.data.end(), 1.0f);
        b.dWk = Tensor2D(M, M); std::fill(b.dWk.data.begin(), b.dWk.data.end(), 2.0f);
        a.dbk.assign(M, 1.0f);  b.dbk.assign(M, 2.0f);
        
        a.dWv = Tensor2D(M, M); std::fill(a.dWv.data.begin(), a.dWv.data.end(), 1.0f);
        b.dWv = Tensor2D(M, M); std::fill(b.dWv.data.begin(), b.dWv.data.end(), 2.0f);
        a.dbv.assign(M, 1.0f);  b.dbv.assign(M, 2.0f);
        
        a.dWo = Tensor2D(M, M); std::fill(a.dWo.data.begin(), a.dWo.data.end(), 1.0f);
        b.dWo = Tensor2D(M, M); std::fill(b.dWo.data.begin(), b.dWo.data.end(), 2.0f);
        a.dbo.assign(M, 1.0f);  b.dbo.assign(M, 2.0f);
        
        // Feedforward layers
        a.d_fc1W = Tensor2D(M, K);  std::fill(a.d_fc1W.data.begin(), a.d_fc1W.data.end(), 1.0f);
        b.d_fc1W = Tensor2D(M, K);  std::fill(b.d_fc1W.data.begin(), b.d_fc1W.data.end(), 2.0f);
        a.d_fc1b.assign(K, 1.0f);   b.d_fc1b.assign(K, 2.0f);
        
        a.d_fc2W = Tensor2D(K2, M); std::fill(a.d_fc2W.data.begin(), a.d_fc2W.data.end(), 1.0f);
        b.d_fc2W = Tensor2D(K2, M); std::fill(b.d_fc2W.data.begin(), b.d_fc2W.data.end(), 2.0f);
        a.d_fc2b.assign(M, 1.0f);   b.d_fc2b.assign(M, 2.0f);
    }
    
    // ---- A += B ----
    A.Add(B);
    
    auto check_tensor_all = [](const Tensor2D& T, float expect, float eps) {
        for (size_t i = 0; i < T.data.size(); ++i)
            if (std::fabs(T.data[i] - expect) > eps) return false;
        return true;
    };
    auto check_vec_all = [](const std::vector<float>& V, float expect, float eps) {
        for (size_t i = 0; i < V.size(); ++i)
            if (std::fabs(V[i] - expect) > eps) return false;
        return true;
    };
    
    const float eps = 1e-6f;
    
    // After add: expect 3 everywhere
    if (!check_tensor_all(A.d_tokW, 3.0f, eps)) return false;
    if (!check_tensor_all(A.d_posP, 3.0f, eps)) return false;
    if (!check_tensor_all(A.d_lmW,  3.0f, eps)) return false;
    if (!check_vec_all(A.d_lmb,     3.0f, eps)) return false;
    
    for (size_t l = 0; l < L; ++l) {
        auto &g = A.layers[l];
        if (!check_vec_all(g.d_ln1g, 3.0f, eps)) return false;
        if (!check_vec_all(g.d_ln1b, 3.0f, eps)) return false;
        if (!check_vec_all(g.d_ln2g, 3.0f, eps)) return false;
        if (!check_vec_all(g.d_ln2b, 3.0f, eps)) return false;
        
        if (!check_tensor_all(g.dWq, 3.0f, eps) || !check_vec_all(g.dbq, 3.0f, eps)) return false;
        if (!check_tensor_all(g.dWk, 3.0f, eps) || !check_vec_all(g.dbk, 3.0f, eps)) return false;
        if (!check_tensor_all(g.dWv, 3.0f, eps) || !check_vec_all(g.dbv, 3.0f, eps)) return false;
        if (!check_tensor_all(g.dWo, 3.0f, eps) || !check_vec_all(g.dbo, 3.0f, eps)) return false;
        
        if (!check_tensor_all(g.d_fc1W, 3.0f, eps) || !check_vec_all(g.d_fc1b, 3.0f, eps)) return false;
        if (!check_tensor_all(g.d_fc2W, 3.0f, eps) || !check_vec_all(g.d_fc2b, 3.0f, eps)) return false;
    }
    
    // ---- Scale by 0.5 -> expect 1.5 everywhere ----
    A.Scale(0.5f);
    
    if (!check_tensor_all(A.d_tokW, 1.5f, eps)) return false;
    if (!check_tensor_all(A.d_posP, 1.5f, eps)) return false;
    if (!check_tensor_all(A.d_lmW,  1.5f, eps)) return false;
    if (!check_vec_all(A.d_lmb,     1.5f, eps)) return false;
    
    for (size_t l = 0; l < L; ++l) {
        auto &g = A.layers[l];
        if (!check_vec_all(g.d_ln1g, 1.5f, eps)) return false;
        if (!check_vec_all(g.d_ln1b, 1.5f, eps)) return false;
        if (!check_vec_all(g.d_ln2g, 1.5f, eps)) return false;
        if (!check_vec_all(g.d_ln2b, 1.5f, eps)) return false;
        
        if (!check_tensor_all(g.dWq, 1.5f, eps) || !check_vec_all(g.dbq, 1.5f, eps)) return false;
        if (!check_tensor_all(g.dWk, 1.5f, eps) || !check_vec_all(g.dbk, 1.5f, eps)) return false;
        if (!check_tensor_all(g.dWv, 1.5f, eps) || !check_vec_all(g.dbv, 1.5f, eps)) return false;
        if (!check_tensor_all(g.dWo, 1.5f, eps) || !check_vec_all(g.dbo, 1.5f, eps)) return false;
        
        if (!check_tensor_all(g.d_fc1W, 1.5f, eps) || !check_vec_all(g.d_fc1b, 1.5f, eps)) return false;
        if (!check_tensor_all(g.d_fc2W, 1.5f, eps) || !check_vec_all(g.d_fc2b, 1.5f, eps)) return false;
    }
    
    return true;
}

