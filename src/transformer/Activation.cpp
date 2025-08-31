#include "Tensor2D.h"
#include "Activation.h"

#include <iostream>
#include <algorithm>
#include <cmath>

ActivationFunctions Activation;

bool ActivationFunctions::SetActivationFunction(ActivationType type) {
    d_ff_mul = 1;
    switch (type) {
        case ActivationType::GELU: Forward = &GELU_Forward; Backward = &GELU_Backward; return true;
        case ActivationType::RELU: Forward = &ReLU_Forward; Backward = &ReLU_Backward; return true;
        case ActivationType::SILU: Forward = &SiLU_Forward; Backward = &SiLU_Backward; return true;
        case ActivationType::MISH: Forward = &Mish_Forward; Backward = &Mish_Backward; return true;
        case ActivationType::SWIGLU: Forward = &SwiGLU_Forward; Backward = &SwiGLU_Backward; d_ff_mul = 2; return true;
    }
    return false;
}

ActivationFunctions::ActivationFunctions() :
    d_ff_mul(1),
    Forward(GELU_Forward),
    Backward(GELU_Backward)
{}

static inline float sigmoid_stable(float x){
    if (x >= 0) { float z = std::exp(-x); return 1.0f/(1.0f+z); }
    float z = std::exp(x); return z/(1.0f+z);
}

static inline float softplus_stable(float x) {
    // log(1 + exp(x)) with numeric stability
    if (x > 20.0f) return x;                // avoid overflow
    if (x < -20.0f) return std::exp(x);     // exp(x) ~ 0, softplus ~ exp(x)
    return std::log1p(std::exp(x));
}

// ---------------- GELU ----------------
Tensor2D ActivationFunctions::GELU_Forward(const Tensor2D& x) {
    Tensor2D y(x.R, x.C);
    const float k = std::sqrt(2.0f / 3.14159265358979323846f);
    for (size_t i = 0; i < x.data.size(); i++) {
        float v = x.data[i];
        float v3 = v * v * v;
        float t = std::tanh(k * (v + 0.044715f * v3));
        y.data[i] = 0.5f * v * (1.0f + t);
    }
    return y;
}
void ActivationFunctions::GELU_Backward(const Tensor2D& x_pre, const Tensor2D& dy, Tensor2D& dx) {
    const float k = std::sqrt(2.0f / 3.14159265358979323846f);
    dx = Tensor2D(x_pre.R, x_pre.C);
    for (int i = 0; i < x_pre.R * x_pre.C; i++) {
        float v = x_pre.data[(size_t)i];
        float v2 = v * v;
        float v3 = v2 * v;
        float u = k * (v + 0.044715f * v3);
        float t = std::tanh(u);
        float dt = 1.0f - t * t; // sech^2
        float du_dv = k * (1.0f + 0.134145f * v2); // 0.134145 = 3*0.044715
        float dgelu = 0.5f * (1.0f + t) + 0.5f * v * dt * du_dv;
        dx.data[(size_t)i] = dy.data[(size_t)i] * dgelu;
    }
}

// ---------------- ReLU ----------------
Tensor2D ActivationFunctions::ReLU_Forward(const Tensor2D& x) {
    Tensor2D y(x.R, x.C);
    for (size_t i = 0; i < x.data.size(); ++i) y.data[i] = x.data[i] > 0.0f ? x.data[i] : 0.0f;
    return y;
}
void ActivationFunctions::ReLU_Backward(const Tensor2D& x_pre, const Tensor2D& dy, Tensor2D& dx) {
    dx = Tensor2D(x_pre.R, x_pre.C);
    const int N = x_pre.R * x_pre.C;
    for (int i = 0; i < N; ++i) {
        const float v  = x_pre.data[(size_t)i];
        const float g  = dy.data[(size_t)i];
        dx.data[(size_t)i] = (v > 0.0f) ? g : 0.0f;
    }
}

// ---------------- SiLU ----------------
Tensor2D ActivationFunctions::SiLU_Forward(const Tensor2D& x) {
    Tensor2D y(x.R, x.C);
    for (size_t i = 0; i < x.data.size(); ++i) {
        float v = x.data[i];
        float s = sigmoid_stable(v);
        y.data[i] = v * s;
    }
    return y;
}
void ActivationFunctions::SiLU_Backward(const Tensor2D& x_pre, const Tensor2D& dy, Tensor2D& dx) {
    dx = Tensor2D(x_pre.R, x_pre.C);
    const int N = x_pre.R * x_pre.C;
    for (int i = 0; i < N; ++i) {
        float v = x_pre.data[(size_t)i];
        float g = dy.data[(size_t)i];
        float s = sigmoid_stable(v);
        float d_silu = s + v * s * (1.0f - s);
        dx.data[(size_t)i] = g * d_silu;
    }
}

// ---------------- Mish ----------------
Tensor2D ActivationFunctions::Mish_Forward(const Tensor2D& x) {
    Tensor2D y(x.R, x.C);
    for (size_t i = 0; i < x.data.size(); ++i) {
        float v = x.data[i];
        float sp = softplus_stable(v);
        float t  = std::tanh(sp);
        y.data[i] = v * t;
    }
    return y;
}
void ActivationFunctions::Mish_Backward(const Tensor2D& x_pre, const Tensor2D& dy, Tensor2D& dx) {
    dx = Tensor2D(x_pre.R, x_pre.C);
    const int N = x_pre.R * x_pre.C;
    for (int i = 0; i < N; ++i) {
        float v = x_pre.data[(size_t)i];
        float g = dy.data[(size_t)i];
        float sp = softplus_stable(v);
        float t  = std::tanh(sp);
        float s  = sigmoid_stable(v);
        float d_mish = t + v * s * (1.0f - t * t);
        dx.data[(size_t)i] = g * d_mish;
    }
}

// ---------------- SwiGLU ----------------
// x_concat = [A | B] along features (C = 2*H). Output is H features: A * SiLU(B).
Tensor2D ActivationFunctions::SwiGLU_Forward(const Tensor2D& x_concat) {
    const int R = x_concat.R;
    const int C = x_concat.C;
    const int H = C / 2;
    Tensor2D y_half(R, H);
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < H; ++c) {
            float a = x_concat(r, c);
            float b = x_concat(r, c + H);
            float s = sigmoid_stable(b);
            float silu = b * s;
            y_half(r, c) = a * silu;
        }
    }
    return y_half;
}
void ActivationFunctions::SwiGLU_Backward(const Tensor2D& x_concat, const Tensor2D& dy_half, Tensor2D& dx_concat) {
    const int R = x_concat.R, C = x_concat.C, H = C/2;
    dx_concat = Tensor2D(R, C);
    for (int r = 0; r < R; ++r){
        const float* xr = x_concat.Row(r);
        const float* dyr= dy_half.Row(r);
        float* dxr = dx_concat.Row(r);
        for (int c = 0; c < H; ++c){
            float a = xr[c], b = xr[c+H];
            float s = sigmoid_stable(b);
            float dSiLU = s + b * s * (1.0f - s);
            float dy = dyr[c];
            dxr[c]     = dy * (b * s);     // dA
            dxr[c+H]   = dy * a * dSiLU;   // dB
        }
    }
}
