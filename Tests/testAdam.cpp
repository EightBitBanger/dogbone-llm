#include "test.h"
#include "../Transformer/AdamOptimization.h"

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

bool TestAdam() {
    const float lr  = 1e-3f;
    const float b1  = 0.9f;
    const float b2  = 0.999f;
    const float eps = 1e-8f;
    const float tol = 1e-9f;
    
    // ---------- Test 1: t = 1 "sign" step ----------
    std::vector<float> w = { 1.0f, -2.0f, 3.0f };
    std::vector<float> g = { 0.1f, -0.2f, 0.3f };
    AdamState s;
    
    // Expected for t=1 with bias correction: mhat=g, vhat=g^2
    std::vector<float> w_exp = w;
    for (size_t i = 0; i < w.size(); ++i) {
        float denom = std::sqrt(g[i]*g[i]) + eps;
        w_exp[i] -= lr * (g[i] / denom);
    }
    
    Adam::StepInPlace(w, g, s, lr, b1, b2, eps, /*t=*/1);
    if (!vec_close(w, w_exp, tol)) return false;
    if (s.m.size() != w.size() || s.v.size() != w.size()) return false;
    
    // ---------- Test 2: t = 2 with persistence ----------
    // Recompute expected using the closed-form bias corrections.
    // After step 1 (from zero state):
    //   m1 = (1-b1) * g
    //   v1 = (1-b2) * g^2
    // Step 2:
    //   m2 = b1*m1 + (1-b1)*g
    //   v2 = b2*v1 + (1-b2)*g^2
    //   mhat2 = m2 / (1 - b1^2)
    //   vhat2 = v2 / (1 - b2^2)
    std::vector<float> m1(g.size()), v1(g.size());
    for (size_t i = 0; i < g.size(); ++i) {
        m1[i] = (1.0f - b1) * g[i];
        v1[i] = (1.0f - b2) * (g[i] * g[i]);
    }
    
    std::vector<float> m2(g.size()), v2(g.size()), mhat2(g.size()), vhat2(g.size());
    for (size_t i = 0; i < g.size(); ++i) {
        m2[i] = b1 * m1[i] + (1.0f - b1) * g[i];
        v2[i] = b2 * v1[i] + (1.0f - b2) * (g[i] * g[i]);
        mhat2[i] = m2[i] / (1.0f - std::pow(b1, 2.0f));
        vhat2[i] = v2[i] / (1.0f - std::pow(b2, 2.0f));
    }
    
    std::vector<float> w_exp2 = w_exp; // start from post-step-1 expected
    for (size_t i = 0; i < g.size(); ++i) {
        float denom = std::sqrt(vhat2[i]) + eps;
        w_exp2[i] -= lr * (mhat2[i] / denom);
    }
    
    Adam::StepInPlace(w, g, s, lr, b1, b2, eps, /*t=*/2);
    if (!vec_close(w, w_exp2, 1e-8f)) return false;
    
    // ---------- Test 3: zero gradient no-op ----------
    std::vector<float> w3 = { -5.0f, 0.0f, 2.5f };
    std::vector<float> g3 = {  0.0f, 0.0f, 0.0f };
    AdamState s3;
    std::vector<float> w3_before = w3;
    Adam::StepInPlace(w3, g3, s3, lr, b1, b2, eps, /*t=*/1);
    if (!vec_close(w3, w3_before, 1e-12f)) return false;
    
    return true;
}
