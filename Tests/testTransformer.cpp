#include "test.h"
#include "../Transformer/Transformer.h"

#include <iostream>
#include <cmath>
#include <cstdint>

static bool vec_close(const std::vector<float>& a,
                      const std::vector<float>& b,
                      float eps) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > eps) return false;
    return true;
}

bool TestTensor2D() {
    Tensor2D A(2, 3);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;
    
    Tensor2D B(3, 2);
    B(0,0)=7;  B(0,1)=8;
    B(1,0)=9;  B(1,1)=10;
    B(2,0)=11; B(2,1)=12;
    
    Tensor2D C = MatMul(A, B);
    
    // Expected results
    const int expected[2][2] = {
        {58, 64},
        {139, 154}
    };
    
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            if (C(r,c) != expected[r][c]) {
                return false;
            }
        }
    }
    return true;
}


bool TestLinearLayer() {
    // x: [2,3]
    Tensor2D x(2, 3);
    x(0,0)=1; x(0,1)=2; x(0,2)=3;
    x(1,0)=4; x(1,1)=5; x(1,2)=6;
    
    // Layer: d_in=3, d_out=2
    LinearLayer lin(3, 2);
    // Overwrite randomized init with fixed values
    lin.W(0,0)=1;  lin.W(0,1)=2;
    lin.W(1,0)=0;  lin.W(1,1)=-1;
    lin.W(2,0)=3;  lin.W(2,1)=1;
    lin.b.resize(2); lin.b[0]=0.5f; lin.b[1]=-1.0f;
    
    // y = xW + b
    Tensor2D y = lin.Forward(x);
    
    // Expected:
    // xW =
    //  [10,  3]
    //  [22,  9]
    // + b -> add to each row: [ +0.5,  -1 ]
    const float exp_[2][2] = { {10.5f, 2.0f}, {22.5f, 8.0f} };
    
    const float eps = 1e-6f;
    for (int r=0;r<2;++r) {
        for (int c=0;c<2;++c) {
            if (std::fabs(y(r,c) - exp_[r][c]) > eps) return false;
        }
    }
    return true;
}


bool TestLayerNorm() {
    // x: 3 rows, 4 features
    Tensor2D x(3, 4);
    // Row 0: varied (non-constant)
    x(0,0)=1; x(0,1)=2; x(0,2)=3; x(0,3)=4;
    // Row 1: constant (edge case -> zeros after norm)
    x(1,0)=2; x(1,1)=2; x(1,2)=2; x(1,3)=2;
    // Row 2: another varied row
    x(2,0)=-1; x(2,1)=0; x(2,2)=1; x(2,3)=2;
    
    LayerNorm ln(4); // gamma=1, beta=0 by default
    Tensor2D y = ln.Forward(x);
    
    // Check: per-row mean ~ 0, var ~ 1 (biased variance, tiny slack for eps)
    const float mean_tol = 1e-4f;
    const float var_tol  = 1e-3f;
    
    for (int r=0;r<x.R;++r) {
        // Compute row mean/var of y
        float m = 0.0f;
        for (int c=0;c<x.C;++c) m += y(r,c);
        m /= (float)x.C;
    
        float v = 0.0f;
        for (int c=0;c<x.C;++c) { float d = y(r,c) - m; v += d*d; }
        v /= (float)x.C;
    
        // For constant input row, output should be ~all zeros -> mean ~0, var ~0
        bool constant_row = true;
        for (int c=1;c<x.C;++c) if (x(r,c)!=x(r,0)) { constant_row=false; break; }
    
        if (std::fabs(m) > mean_tol) return false;
        if (constant_row) {
            if (v > var_tol) return false; // should be ~0
        } else {
            if (std::fabs(v - 1.0f) > var_tol) return false; // ~1
        }
    }
    
    // Now verify gamma/beta effect: y2 == y*2 - 1
    for (size_t i=0;i<ln.gamma.size();++i) { ln.gamma[i]=2.0f; ln.beta[i]=-1.0f; }
    Tensor2D y2 = ln.Forward(x);
    
    const float aff_tol = 1e-5f;
    for (int r=0;r<x.R;++r) {
        for (int c=0;c<x.C;++c) {
            float expect = y(r,c)*2.0f - 1.0f;
            if (std::fabs(y2(r,c) - expect) > aff_tol) return false;
        }
    }
    return true;
}


bool TestMultiHeadSelfAttention() {
    const int d_model = 4;
    const int heads   = 2;
    MultiHeadSelfAttention attn(d_model, heads); // d_head = 2
    
    // Make Wq, Wk, Wv, Wo act like identity; zero bias
    auto set_identity = [](LinearLayer& L) {
        for (int r = 0; r < L.W.R; ++r)
            for (int c = 0; c < L.W.C; ++c)
                L.W(r, c) = (r == c) ? 1.0f : 0.0f;
        L.b.assign((size_t)L.W.C, 0.0f);
    };
    set_identity(attn.Wq);
    set_identity(attn.Wk);
    set_identity(attn.Wv);
    set_identity(attn.Wo);
    
    // x: 2 tokens × 4 dims (two heads: [0..1] and [2..3])
    Tensor2D x(2, d_model);
    // t=0
    x(0,0)=1; x(0,1)=0; x(0,2)=0; x(0,3)=1;
    // t=1
    x(1,0)=0; x(1,1)=1; x(1,2)=1; x(1,3)=0;
    
    Tensor2D y = attn.Forward(x, nullptr);
    
    // Expected:
    // t=0 -> single element softmax => y0 == x0
    float exp0[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    
    // t=1 -> per-head softmax over uE{0,1} with logits [0, 1/sqrt(2)]
    // Using the stabilized form used in your code: w0 = e^{-s} / (e^{-s}+1), w1 = 1/(e^{-s}+1)
    const float scale = 1.0f / std::sqrt( (float)(d_model / heads) ); // 1/sqrt(2)
    const float a = std::exp(-scale);
    const float w1 = 1.0f / (a + 1.0f);
    const float w0 = a / (a + 1.0f);
    // head0 output: [w0, w1], head1 output: [w1, w0]
    float exp1[4] = {w0, w1, w1, w0};
    
    const float eps = 1e-5f;
    for (int c=0;c<4;++c) if (std::fabs(y(0,c) - exp0[c]) > eps) return false;
    for (int c=0;c<4;++c) if (std::fabs(y(1,c) - exp1[c]) > eps) return false;
    return true;
}

bool TestPositionalEncoding() {
    const int T = 3, D = 4;
    PositionalEncoding pe(T, D);
    
    // Deterministic P
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < D; ++c)
            pe.P(t, c) = (float)(t * D + c + 1); // 1..12
    
    Tensor2D x(T, D);
    x.Zero();
    
    pe.AddInPlace(x); // x == P
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < D; ++c)
            if (x(t, c) != pe.P(t, c)) return false;
    
    pe.AddInPlace(x); // x == 2 * P
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < D; ++c)
            if (x(t, c) != 2.0f * pe.P(t, c)) return false;
    
    return true;
}

bool TestTransformerBlock() {
    const int D = 4, H = 2, F = 8;
    TransformerBlock blk(D, H, F);
    
    Tensor2D x(3, D);
    for (int t = 0; t < x.R; ++t)
        for (int c = 0; c < x.C; ++c)
            x(t, c) = 0.1f * (float)(t * x.C + c);
    
    // Block output
    Tensor2D y = blk.Forward(x, nullptr);
    
    // Manual composition: x + attn(ln1(x)) + ffn(ln2(x + attn(...)))
    Tensor2D n1 = blk.ln1.Forward(x);
    Tensor2D a  = blk.attn.Forward(n1, nullptr);
    Tensor2D x1 = x; AddInPlace(x1, a);
    Tensor2D n2 = blk.ln2.Forward(x1);
    Tensor2D f  = blk.ffn.Forward(n2);
    Tensor2D y_ref = x1; AddInPlace(y_ref, f);
    
    const float eps = 1e-6f;
    for (size_t i = 0; i < y.data.size(); ++i)
        if (std::fabs(y.data[i] - y_ref.data[i]) > eps) return false;
    
    return true;
}

