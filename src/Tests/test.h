#include "../Transformer/ShaderTensor.h"

#include <iostream>
#include <cmath>
#include <cstdint>
#include "Activation.h"
#include "../Transformer/Tensor2D.h"

// ---------- tiny helpers ----------
static inline bool almost_equal(float a, float b, float atol, float rtol) {
    float diff = std::fabs(a - b);
    float tol  = atol + rtol * std::max(std::fabs(a), std::fabs(b));
    return diff <= tol;
}

static inline float clamp(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline uint32_t lcg_step(uint32_t& s) { // simple deterministic rng
    s = 1664525u * s + 1013904223u;
    return s;
}

static Tensor2D make_random_tensor(int R, int C, uint32_t seed, float lo, float hi) {
    Tensor2D t(R, C);
    for (size_t i = 0; i < t.data.size(); ++i) {
        float u = (float)(lcg_step(seed) / 4294967295.0); // [0,1]
        t.data[i] = lo + (hi - lo) * u;
    }
    return t;
}

static float dot(const Tensor2D& a, const Tensor2D& b) {
    float s = 0.0f;
    for (size_t i = 0; i < a.data.size(); ++i) s += a.data[i] * b.data[i];
    return s;
}

static Tensor2D zeros_like(const Tensor2D& t) {
    Tensor2D z(t.R, t.C);
    for (size_t i = 0; i < z.data.size(); ++i) z.data[i] = 0.0f;
    return z;
}

static Tensor2D ones_like(const Tensor2D& t) {
    Tensor2D o(t.R, t.C);
    for (size_t i = 0; i < o.data.size(); ++i) o.data[i] = 1.0f;
    return o;
}

// Computes S(x) = sum( Forward(x) . dy ), a scalar.
// Its gradient wrt x must equal Backward(x, dy).
static float scalar_S(ActivationFunctions& A, const Tensor2D& x, const Tensor2D& dy) {
    Tensor2D y = A.Forward(x);
    return dot(y, dy);
}

// Finite-difference gradient check: compares Backward vs numeric gradient of S(x).
// Returns max absolute and relative error via out params.
static void gradcheck_activation(ActivationType type,
                                 int R, int C,
                                 float eps,
                                 float& out_max_abs_err,
                                 float& out_max_rel_err,
                                 uint32_t seed = 1u) {
    ActivationFunctions& A = Activation;
    A.SetActivationFunction(type); // sets Forward/Backward and d_ff_mul
    // For SwiGLU we need even C (features split into A|B), output has H=C/2
    if (type == ActivationType::SWIGLU) {
        if ((C % 2) != 0) C += 1; // nudge to even
    }

    // Build inputs
    Tensor2D x = make_random_tensor(R, C, seed, -3.0f, 3.0f);

    // dy must match Forward(x). For SwiGLU, Forward returns (R, C/2).
    Tensor2D y_shape = A.Forward(x); // one cheap call to get shape
    Tensor2D dy = make_random_tensor(y_shape.R, y_shape.C, seed + 1234u, -1.0f, 1.0f);

    // Analytical gradient
    Tensor2D dx_analytic(x.R, x.C);
    A.Backward(x, dy, dx_analytic);

    // Numeric gradient via central differences on S(x) = sum(Forward(x) * dy)
    Tensor2D dx_numeric(x.R, x.C);
    for (int r = 0; r < x.R; ++r) {
        for (int c = 0; c < x.C; ++c) {
            size_t idx = (size_t)r * (size_t)x.C + (size_t)c;
            float orig = x.data[idx];

            x.data[idx] = orig + eps;
            float Spos = scalar_S(A, x, dy);

            x.data[idx] = orig - eps;
            float Sneg = scalar_S(A, x, dy);

            x.data[idx] = orig; // restore
            float g = (Spos - Sneg) / (2.0f * eps);
            dx_numeric.data[idx] = g;
        }
    }

    // Compare
    out_max_abs_err = 0.0f;
    out_max_rel_err = 0.0f;
    for (size_t i = 0; i < dx_numeric.data.size(); ++i) {
        float a = dx_analytic.data[i];
        float n = dx_numeric.data[i];
        float abs_err = std::fabs(a - n);
        float rel_err = abs_err / (1e-8f + std::fabs(n));
        if (abs_err > out_max_abs_err) out_max_abs_err = abs_err;
        if (rel_err > out_max_rel_err) out_max_rel_err = rel_err;
    }
}

// ---------- specific sanity tests (simple, fast) ----------
static bool test_relu_known_values() {
    Activation.SetActivationFunction(ActivationType::RELU);
    Tensor2D x(1,3);
    x.data[0] = -1.0f; x.data[1] = 0.0f; x.data[2] = 2.0f;

    Tensor2D y = Activation.Forward(x);
    bool ok = true;
    ok = ok && (y.data[0] == 0.0f);
    ok = ok && (y.data[1] == 0.0f);
    ok = ok && (y.data[2] == 2.0f);

    Tensor2D dy = ones_like(y);
    Tensor2D dx(x.R, x.C);
    Activation.Backward(x, dy, dx);

    // derivative is 0 for v<=0 (your code uses v>0), 1 for v>0
    ok = ok && (dx.data[0] == 0.0f);
    ok = ok && (dx.data[1] == 0.0f);
    ok = ok && (dx.data[2] == 1.0f);
    return ok;
}

static bool test_silu_at_zero() {
    Activation.SetActivationFunction(ActivationType::SILU);
    Tensor2D x(1,1); x.data[0] = 0.0f;
    Tensor2D y = Activation.Forward(x);
    if (!almost_equal(y.data[0], 0.0f, 1e-7f, 0.0f)) return false;
    Tensor2D dy = ones_like(y);
    Tensor2D dx(1,1);
    Activation.Backward(x, dy, dx);
    // d/dx SiLU at 0 = 0.5
    return almost_equal(dx.data[0], 0.5f, 1e-6f, 0.0f);
}

static bool test_mish_at_zero() {
    Activation.SetActivationFunction(ActivationType::MISH);
    Tensor2D x(1,1); x.data[0] = 0.0f;
    Tensor2D y = Activation.Forward(x);
    if (!almost_equal(y.data[0], 0.0f, 1e-7f, 0.0f)) return false;
    Tensor2D dy = ones_like(y);
    Tensor2D dx(1,1);
    Activation.Backward(x, dy, dx);
    // d/dx Mish at 0 = tanh(softplus(0)) ~ tanh(log(2)) ~ 0.6
    return almost_equal(dx.data[0], 0.6f, 1e-3f, 0.0f);
}

static bool test_gelu_at_zero() {
    Activation.SetActivationFunction(ActivationType::GELU);
    Tensor2D x(1,1); x.data[0] = 0.0f;
    Tensor2D dy(1,1); dy.data[0] = 1.0f;
    Tensor2D dx(1,1);
    Activation.Backward(x, dy, dx);
    // with the tanh-approx gelu used here, derivative at 0 = 0.5
    return almost_equal(dx.data[0], 0.5f, 1e-4f, 0.0f);
}

static bool test_swglu_shape_and_simple_case() {
    Activation.SetActivationFunction(ActivationType::SWIGLU);
    const int R = 1, H = 2;
    Tensor2D x(1, 2*H);

    // A = [1, 1], B = [0, 0]  => y = A * SiLU(B) = 0
    x.data[0] = 1.0f; // A0
    x.data[1] = 1.0f; // A1
    x.data[2] = 0.0f; // B0
    x.data[3] = 0.0f; // B1

    Tensor2D y = Activation.Forward(x);
    if (!(y.R == R && y.C == H)) return false;
    if (!almost_equal(y.data[0], 0.0f, 1e-7f, 0.0f)) return false;
    if (!almost_equal(y.data[1], 0.0f, 1e-7f, 0.0f)) return false;

    // Backward: with dy = [1,1], dA = B*s = 0, dB = A * dSiLU(0) = A*0.5
    Tensor2D dyv = ones_like(y);
    Tensor2D dx(1, 2*H);
    Activation.Backward(x, dyv, dx);

    bool ok = true;
    ok = ok && almost_equal(dx.data[0], 0.0f, 1e-7f, 0.0f); // dA0
    ok = ok && almost_equal(dx.data[1], 0.0f, 1e-7f, 0.0f); // dA1
    ok = ok && almost_equal(dx.data[2], 0.5f, 1e-6f, 0.0f); // dB0
    ok = ok && almost_equal(dx.data[3], 0.5f, 1e-6f, 0.0f); // dB1
    return ok;
}

// ---------- gradient-check wrappers per activation ----------
static bool gradcheck_relu() {
    float a, r;
    gradcheck_activation(ActivationType::RELU, 3, 7, 1e-3f, a, r, 1337u);
    // relu is piecewise-linear; expect tiny error except near zero crossings
    return (a <= 5e-4f && r <= 5e-4f);
}

static bool gradcheck_silu() {
    float a, r;
    gradcheck_activation(ActivationType::SILU, 3, 7, 1e-4f, a, r, 42u);
    return (a <= 2e-4f && r <= 2e-3f);
}

static bool gradcheck_mish() {
    float a, r;
    gradcheck_activation(ActivationType::MISH, 3, 7, 2e-4f, a, r, 7u);
    return (a <= 5e-4f && r <= 3e-3f);
}

static bool gradcheck_gelu() {
    float a, r;
    // gelu uses tanh approximation; loosen tolerances slightly
    gradcheck_activation(ActivationType::GELU, 3, 7, 2e-4f, a, r, 2025u);
    return (a <= 1e-3f && r <= 6e-3f);
}

static bool gradcheck_swglu() {
    float a, r;
    // Need even C; the helper nudges it if odd
    gradcheck_activation(ActivationType::SWIGLU, 3, 8, 1e-4f, a, r, 99u);
    return (a <= 3e-4f && r <= 2e-3f);
}

// ---------- public entrypoint ----------
void RunAllUnitTests() {
    int pass = 0, total = 0;

    struct Item { const char* name; bool (*fn)(); };
    Item sanity[] = {
        { "relu known values",      &test_relu_known_values },
        { "silu at zero",           &test_silu_at_zero      },
        { "mish at zero",           &test_mish_at_zero      },
        { "gelu at zero (tanh approx)", &test_gelu_at_zero  },
        { "swiglu simple/shape",    &test_swglu_shape_and_simple_case }
    };
    for (size_t i = 0; i < sizeof(sanity)/sizeof(sanity[0]); ++i) {
        bool ok = sanity[i].fn();
        std::cout << "[sanity] " << sanity[i].name << ": " << (ok ? "OK" : "FAIL") << "\n";
        total++; if (ok) pass++;
    }

    Item grads[] = {
        { "gradcheck relu",   &gradcheck_relu  },
        { "gradcheck silu",   &gradcheck_silu  },
        { "gradcheck mish",   &gradcheck_mish  },
        { "gradcheck gelu",   &gradcheck_gelu  },
        { "gradcheck swiglu", &gradcheck_swglu }
    };
    for (size_t i = 0; i < sizeof(grads)/sizeof(grads[0]); ++i) {
        bool ok = grads[i].fn();
        std::cout << "[grad]   " << grads[i].name << ": " << (ok ? "OK" : "FAIL") << "\n";
        total++; if (ok) pass++;
    }

    std::cout << "activation tests: " << pass << "/" << total << " passed\n";
}



void TestTensorShader(ShaderTensor& shader) {
    // X[2x3] * W[3x2] + B[2]  ->  Y[2x2]
    float Xh[6] = { 1,2,3,  4,5,6 };
    float Wh[6] = { 1,0,  0,1,  1,1 };  // [[1,0],[0,1],[1,1]]
    float Bh[2] = { 0.5f, -1.0f };
    
    ShaderTensor& st = shader;
    st.createSSBO("X",    sizeof(Xh), 0); st.upload("X", Xh, sizeof(Xh));
    st.createSSBO("W",    sizeof(Wh), 1); st.upload("W", Wh, sizeof(Wh));
    st.createSSBO("B",    sizeof(Bh), 2); st.upload("B", Bh, sizeof(Bh));
    st.createSSBO("Y",    sizeof(float)*4, 3);
    int meta[3] = { 2,3,2 };  // T=2, IN=3, OUT=2
    st.createSSBO("Meta", sizeof(meta), 4); st.upload("Meta", meta, sizeof(meta));
    
    st.use();
    st.dispatch(1,1,1);
    st.sync();
    
    float Yh[4] = {};
    st.downloadSync("Y", Yh, sizeof(Yh));
    // Expect: row0 = [1*1+2*0+3*1+0.5, 1*0+2*1+3*1-1] = [4.5, 4.0]
    //         row1 = [4*1+5*0+6*1+0.5, 4*0+5*1+6*1-1] = [10.5,10.0]
    std::cout << "GPU shader test Y: [" << Yh[0] << "," << Yh[1] << "; " << Yh[2] << "," << Yh[3] << "]\n\n";
    
}

