#include "test.h"
#include "../Transformer/Activation.h"

#include <iostream>
#include <cmath>
#include <cstdint>

static inline float sum_all(const Tensor2D& t) {
    float s = 0.0f; for (size_t i = 0; i < t.data.size(); ++i) s += t.data[i]; return s;
}
static inline bool close_f(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

// Central-difference gradient check for one activation.
// If is_swiglu == true: y has half the columns of x (C/2), so shape dy accordingly.
static bool grad_check(const Tensor2D& x_in,
                       Tensor2D (*Fwd)(const Tensor2D&),
                       void     (*Bwd)(const Tensor2D&, const Tensor2D&, Tensor2D&),
                       bool is_swiglu = false,
                       float eps = 1e-4f,
                       float tol = 3e-3f) {
    // Clone input (we’ll perturb copies)
    Tensor2D x = x_in;

    // Analytic gradient via Backward with dy = 1’s
    Tensor2D y0 = Fwd(x);
    Tensor2D dy(y0.R, y0.C);
    for (size_t i = 0; i < dy.data.size(); ++i) dy.data[i] = 1.0f;

    Tensor2D dx_analytic;
    Bwd(x, dy, dx_analytic);
    if (dx_analytic.R != x.R || dx_analytic.C != x.C) return false;

    // Numeric gradient per element
    for (int r = 0; r < x.R; ++r) {
        for (int c = 0; c < x.C; ++c) {
            Tensor2D xp = x, xm = x;
            xp(r,c) += eps;
            xm(r,c) -= eps;

            float Lp = sum_all(Fwd(xp));  // L(x+eps) = sum(Forward(x+eps))
            float Lm = sum_all(Fwd(xm));  // L(x-eps)

            float g_num = (Lp - Lm) / (2.0f * eps);
            float g_ana = dx_analytic(r,c);

            if (!close_f(g_num, g_ana, tol)) return false;
        }
    }
    return true;
}

// Build a small tensor with mixed signs (avoid exact zeros for ReLU kinks)
static Tensor2D make_small_input(int R, int C) {
    Tensor2D x(R, C);
    float vals[] = {-1.3f, -0.6f, -0.2f, 0.2f, 0.7f, 1.1f, -0.9f, 0.5f};
    int idx = 0;
    for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) x(r,c) = vals[idx++ % 8];
    return x;
}

// ----- individual tests (bool) -----
extern ActivationFunctions Activation;

bool TestGELU_FwdBwd() {
    Tensor2D x = make_small_input(2, 5);
    return grad_check(x,
        &Activation.GELU_Forward,
        &Activation.GELU_Backward,
        false);
}

bool TestReLU_FwdBwd() {
    Tensor2D x = make_small_input(2, 5);
    // Slightly looser tol near 0 due to non-differentiability; grad_check uses values away from 0.
    return grad_check(x,
        &Activation.ReLU_Forward,
        &Activation.ReLU_Backward,
        false, 1e-4f, 5e-3f);
}

bool TestSiLU_FwdBwd() {
    Tensor2D x = make_small_input(2, 5);
    return grad_check(x,
        &Activation.SiLU_Forward,
        &Activation.SiLU_Backward,
        false);
}

bool TestMish_FwdBwd() {
    Tensor2D x = make_small_input(2, 5);
    return grad_check(x,
        &Activation.Mish_Forward,
        &Activation.Mish_Backward,
        false, 1e-4f, 4e-3f);
}

bool TestSwiGLU_FwdBwd() {
    // x_concat has C = 2*H
    const int R = 2, H = 3;
    Tensor2D x(R, 2*H);
    // Fill with varied values
    float vals[] = {-1.0f, -0.3f, 0.2f, 0.8f, 1.3f, -0.7f};
    int k = 0;
    for (int r = 0; r < R; ++r) for (int c = 0; c < 2*H; ++c) x(r,c) = vals[(k++) % 6];

    return grad_check(x,
        &Activation.SwiGLU_Forward,
        &Activation.SwiGLU_Backward,
        true, 1e-4f, 4e-3f);
}

// Quick sanity on the public selector + d_ff_mul behavior for SWIGLU
bool TestActivationSelectorRouting() {
    ActivationFunctions A;
    if (!A.SetActivationFunction(ActivationType::SWIGLU)) return false;
    if (A.d_ff_mul != 2) return false; // SwiGLU doubles feed-forward width

    // Make sure the routed Forward behaves like the direct SwiGLU_Forward
    Tensor2D x(2, 6);
    for (int r=0;r<2;++r) for (int c=0;c<6;++c) x(r,c) = 0.1f * (float)(1 + r*6 + c);

    Tensor2D yrouted = A.Forward(x);
    Tensor2D ydirect = ActivationFunctions::SwiGLU_Forward(x);

    if (yrouted.R != ydirect.R || yrouted.C != ydirect.C) return false;
    for (size_t i=0;i<yrouted.data.size();++i)
        if (!close_f(yrouted.data[i], ydirect.data[i], 1e-6f)) return false;

    return true;
}
