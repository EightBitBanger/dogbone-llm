#include "Tensor2D.h"
#include "Activation.h"

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

// --- replace the old softplus + keep sigmoid_stable as-is ---
static inline float softplus_stable(float x) {
    // exact, smooth, numerically stable for all x:
    // softplus(x) = log1p(exp(-|x|)) + max(x, 0)
    float ax = std::fabs(x);
    return std::log1p(std::exp(-ax)) + std::max(x, 0.0f);
}

// derivative of softplus equals sigmoid
static inline float softplus_prime(float x) {
    return 1.0f / (1.0f + std::exp(-x)); // or sigmoid_stable(x)
}


// ---------------- GELU ----------------
Tensor2D ActivationFunctions::GELU_Forward(const Tensor2D& x) {
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    const float k = 0.044715f;
    Tensor2D y(x.R, x.C);
    for (int r = 0; r < x.R; ++r) {
        const float* xr = x.Row(r);
        float* yr = y.Row(r);
        for (int ccol = 0; ccol < x.C; ++ccol) {
            float v = xr[ccol];
            float u = c * (v + k * v * v * v);
            yr[ccol] = 0.5f * v * (1.0f + std::tanh(u));
        }
    }
    return y;
}

void ActivationFunctions::GELU_Backward(const Tensor2D& x_pre, const Tensor2D& dy, Tensor2D& dx) {
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    const float k = 0.044715f;
    dx = Tensor2D(x_pre.R, x_pre.C);
    for (int r = 0; r < x_pre.R; ++r) {
        const float* xr = x_pre.Row(r);
        const float* dyr = dy.Row(r);
        float* dxr = dx.Row(r);
        for (int ccol = 0; ccol < x_pre.C; ++ccol) {
            float v = xr[ccol];
            float u = c * (v + k * v * v * v);
            float t = std::tanh(u);
            float sech2 = 1.0f - t * t;          // sech^2(u)
            float du = c * (1.0f + 3.0f * k * v * v);
            float dgelu = 0.5f * (1.0f + t) + 0.5f * v * sech2 * du;
            dxr[ccol] = dyr[ccol] * dgelu;       // << multiply by upstream dy
        }
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
    for (int r = 0; r < x.R; ++r) {
        const float* xr = x.Row(r);
        float* yr = y.Row(r);
        for (int c = 0; c < x.C; ++c) {
            float v = xr[c];
            float s = sigmoid_stable(v);
            yr[c] = v * s;
        }
    }
    return y;
}

void ActivationFunctions::SiLU_Backward(const Tensor2D& x_pre, const Tensor2D& dy, Tensor2D& dx) {
    dx = Tensor2D(x_pre.R, x_pre.C);
    for (int r = 0; r < x_pre.R; ++r) {
        const float* xr = x_pre.Row(r);
        const float* dyr = dy.Row(r);
        float* dxr = dx.Row(r);
        for (int c = 0; c < x_pre.C; ++c) {
            float v = xr[c];
            float s = sigmoid_stable(v);
            float dsilu = s + v * s * (1.0f - s);
            dxr[c] = dyr[c] * dsilu; // << multiply by dy
        }
    }
}


// ---------------- Mish ----------------
Tensor2D ActivationFunctions::Mish_Forward(const Tensor2D& x) {
    Tensor2D y(x.R, x.C);
    for (int r = 0; r < x.R; ++r) {
        const float* xr = x.Row(r);
        float* yr = y.Row(r);
        for (int c = 0; c < x.C; ++c) {
            float v = xr[c];
            float p = softplus_stable(v);   // log1p(exp(v)) with a stable impl
            float t = std::tanh(p);
            yr[c] = v * t;
        }
    }
    return y;
}

void ActivationFunctions::Mish_Backward(const Tensor2D& x_pre, const Tensor2D& dy, Tensor2D& dx) {
    dx = Tensor2D(x_pre.R, x_pre.C);
    for (int r = 0; r < x_pre.R; ++r) {
        const float* xr  = x_pre.Row(r);
        const float* dyr = dy.Row(r);
        float* dxr = dx.Row(r);
        for (int c = 0; c < x_pre.C; ++c) {
            float v   = xr[c];
            float sp  = softplus_stable(v);
            float t   = std::tanh(sp);
            float spd = softplus_prime(v);          // <-- exact, consistent with forward
            float dm  = t + v * (1.0f - t*t) * spd; // chain rule
            dxr[c]    = dyr[c] * dm;
        }
    }
}


// ---------------- SwiGLU ----------------
// x_concat = [A | B] along features (C = 2*H). Output is H features: A * SiLU(B).
Tensor2D ActivationFunctions::SwiGLU_Forward(const Tensor2D& x_concat) {
    const int R = x_concat.R;
    const int C = x_concat.C;           // C must be even
    const int H = C / 2;
    Tensor2D y(R, H);
    for (int r = 0; r < R; ++r) {
        const float* xr = x_concat.Row(r);
        float* yr = y.Row(r);
        for (int c = 0; c < H; ++c) {
            float a = xr[c], b = xr[c + H];
            float s = sigmoid_stable(b);
            yr[c] = a * (b * s);        // a * silu(b)
        }
    }
    return y;
}

void ActivationFunctions::SwiGLU_Backward(const Tensor2D& x_concat, const Tensor2D& dy_half, Tensor2D& dx_concat) {
    const int R = x_concat.R;
    const int C = x_concat.C;
    const int H = C / 2;
    dx_concat = Tensor2D(R, C);
    for (int r = 0; r < R; ++r) {
        const float* xr  = x_concat.Row(r);
        const float* dyr = dy_half.Row(r);
        float* dxr = dx_concat.Row(r);
        for (int c = 0; c < H; ++c) {
            float a = xr[c], b = xr[c+H];
            float s = sigmoid_stable(b);
            float dSiLU = s + b * s * (1.0f - s);  // d/db (b * s)
            float g = dyr[c];
            dxr[c]     = g * (b * s);              // dL/dA
            dxr[c+H]   = g * a * dSiLU;            // dL/dB
        }
    }
}
