#include "llm_math.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace math {

// ---------- helpers ----------
size_t idx2d(size_t r, size_t c, size_t cols) { return r * cols + c; }
float  rsqrt(float x) { return 1.0f / std::sqrt(x); }
float  clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

// ---------- vector ops ----------
float dot(const float* a, const float* b, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}
void add(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}
void add_inplace(float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] += b[i];
}
void sub(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
}
void mul(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
}
void scale_inplace(float* a, float s, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] *= s;
}
void copy(const float* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = src[i];
}
float l2_norm(const float* a, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i) s += a[i] * a[i];
    return std::sqrt(s);
}

// ---------- matrix ops (row-major) ----------
void matmul(const float* A, const float* B, float* C,
            size_t M, size_t K, size_t N) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float s = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                s += A[idx2d(m, k, K)] * B[idx2d(k, n, N)];
            }
            C[idx2d(m, n, N)] = s;
        }
    }
}

void matvec(const float* A, const float* x, float* y,
            size_t M, size_t N) {
    for (size_t m = 0; m < M; ++m) {
        float s = 0.0f;
        for (size_t n = 0; n < N; ++n) {
            s += A[idx2d(m, n, N)] * x[n];
        }
        y[m] = s;
    }
}

void matvec_bias(const float* A, const float* x, const float* b, float* y,
                 size_t M, size_t N) {
    for (size_t m = 0; m < M; ++m) {
        float s = 0.0f;
        for (size_t n = 0; n < N; ++n) s += A[idx2d(m, n, N)] * x[n];
        y[m] = s + b[m];
    }
}

void linear(const float* W_rowmajor_NxM, const float* x_N, const float* b_M,
            float* y_M, size_t N, size_t M) {
    for (size_t m = 0; m < M; ++m) {
        float s = 0.0f;
        for (size_t n = 0; n < N; ++n) {
            s += W_rowmajor_NxM[idx2d(n, m, M)] * x_N[n];
        }
        y_M[m] = s + (b_M ? b_M[m] : 0.0f);
    }
}

// ---------- activations ----------
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
void sigmoid_inplace(float* x, size_t n) {
    for (size_t i = 0; i < n; ++i) x[i] = sigmoid(x[i]);
}

float gelu(float x) {
    const float k0 = 0.044715f;
    const float rt2pi = 0.7978845608f; // sqrt(2/pi)
    float x3 = x * x * x;
    float t = rt2pi * (x + k0 * x3);
    float th = std::tanh(t);
    return 0.5f * x * (1.0f + th);
}
void gelu_inplace(float* x, size_t n) {
    for (size_t i = 0; i < n; ++i) x[i] = gelu(x[i]);
}

float silu(float x) { return x * sigmoid(x); }
void silu_inplace(float* x, size_t n) {
    for (size_t i = 0; i < n; ++i) x[i] = silu(x[i]);
}

// ---------- softmax family ----------
void softmax_inplace(float* x, size_t n) {
    if (n == 0) return;
    float xmax = x[0];
    for (size_t i = 1; i < n; ++i) if (x[i] > xmax) xmax = x[i];
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) { x[i] = std::exp(x[i] - xmax); sum += x[i]; }
    float inv = 1.0f / (sum > 0.0f ? sum : 1.0f);
    for (size_t i = 0; i < n; ++i) x[i] *= inv;
}

void masked_softmax_inplace(float* x, const uint8_t* mask_keep, size_t n, float neg_inf) {
    if (n == 0) return;
    float xmax = -1e30f;
    for (size_t i = 0; i < n; ++i) {
        float v = mask_keep && !mask_keep[i] ? neg_inf : x[i];
        if (v > xmax) xmax = v;
    }
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float v = mask_keep && !mask_keep[i] ? neg_inf : x[i];
        x[i] = std::exp(v - xmax);
        sum += x[i];
    }
    float inv = 1.0f / (sum > 0.0f ? sum : 1.0f);
    for (size_t i = 0; i < n; ++i) x[i] *= inv;
}

void log_softmax_inplace(float* x, size_t n) {
    if (n == 0) return;
    float xmax = x[0];
    for (size_t i = 1; i < n; ++i) if (x[i] > xmax) xmax = x[i];
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += std::exp(x[i] - xmax);
    float lsum = std::log(sum);
    for (size_t i = 0; i < n; ++i) x[i] = x[i] - xmax - lsum;
}

// ---------- normalization ----------
void layernorm_inplace(float* x, size_t n, const float* gamma, const float* beta, float eps) {
    if (n == 0) return;
    float mean = 0.0f;
    for (size_t i = 0; i < n; ++i) mean += x[i];
    mean /= (float)n;
    float var = 0.0f;
    for (size_t i = 0; i < n; ++i) { float d = x[i] - mean; var += d * d; }
    var /= (float)n;
    float inv = 1.0f / std::sqrt(var + eps);
    for (size_t i = 0; i < n; ++i) {
        float h = (x[i] - mean) * inv;
        h = (gamma ? h * gamma[i] : h);
        x[i] = h + (beta ? beta[i] : 0.0f);
    }
}

void rmsnorm_inplace(float* x, size_t n, const float* weight, float eps) {
    if (n == 0) return;
    float ms = 0.0f;
    for (size_t i = 0; i < n; ++i) ms += x[i] * x[i];
    ms /= (float)n;
    float inv = 1.0f / std::sqrt(ms + eps);
    for (size_t i = 0; i < n; ++i) x[i] = (x[i] * inv) * (weight ? weight[i] : 1.0f);
}

// ---------- losses / grads ----------
float cross_entropy_loss(const float* logits, size_t n, int target) {
    if (n == 0 || (size_t)target >= n) return 0.0f;
    float xmax = logits[0];
    for (size_t i = 1; i < n; ++i) if (logits[i] > xmax) xmax = logits[i];
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += std::exp(logits[i] - xmax);
    float logprob = logits[(size_t)target] - xmax - std::log(sum);
    return -logprob;
}

float softmax_xent_grad(const float* logits, size_t n, int target, float* dlogits_out) {
    if (n == 0) return 0.0f;
    std::vector<float> tmp(n);
    for (size_t i = 0; i < n; ++i) tmp[i] = logits[i];
    softmax_inplace(&tmp[0], n);
    for (size_t i = 0; i < n; ++i) dlogits_out[i] = tmp[i];
    if ((size_t)target < n) dlogits_out[(size_t)target] -= 1.0f;
    float loss = 0.0f;
    if ((size_t)target < n) {
        float p = clampf(tmp[(size_t)target], 1e-12f, 1.0f);
        loss = -std::log(p);
    }
    return loss;
}

// ---------- attention bits (single-head, naive CPU) ----------
void causal_mask_inplace(float* scores_rowmajor, size_t Tq, size_t Tk, float neg_inf) {
    size_t cols = Tk;
    for (size_t r = 0; r < Tq; ++r) {
        for (size_t c = 0; c < Tk; ++c) {
            if (c > r) scores_rowmajor[idx2d(r, c, cols)] = neg_inf;
        }
    }
}

void attention_single_head(const float* Q, const float* K, const float* V,
                           float* out,
                           size_t Tq, size_t Tk, size_t D,
                           bool causal) {
    const float scale = rsqrt((float)D);
    const float neg_inf = -1e9f;

    std::vector<float> scores(Tk);
    std::vector<float> weights(Tk);

    for (size_t rq = 0; rq < Tq; ++rq) {
        for (size_t ck = 0; ck < Tk; ++ck) {
            const float s = dot(&Q[idx2d(rq, 0, D)], &K[idx2d(ck, 0, D)], D) * scale;
            scores[ck] = causal && ck > rq ? neg_inf : s;
        }
        float xmax = scores[0];
        for (size_t i = 1; i < Tk; ++i) if (scores[i] > xmax) xmax = scores[i];
        float sum = 0.0f;
        for (size_t i = 0; i < Tk; ++i) { weights[i] = std::exp(scores[i] - xmax); sum += weights[i]; }
        float inv = 1.0f / (sum > 0.0f ? sum : 1.0f);
        for (size_t i = 0; i < Tk; ++i) weights[i] *= inv;

        float* out_row = &out[idx2d(rq, 0, D)];
        for (size_t d = 0; d < D; ++d) out_row[d] = 0.0f;
        for (size_t ck = 0; ck < Tk; ++ck) {
            const float w = weights[ck];
            const float* vrow = &V[idx2d(ck, 0, D)];
            for (size_t d = 0; d < D; ++d) out_row[d] += w * vrow[d];
        }
    }
}

// ---------- multi-head pack/unpack ----------
void split_heads(const float* x, float* out, size_t seq, size_t H, size_t Dh) {
    for (size_t s = 0; s < seq; ++s) {
        for (size_t h = 0; h < H; ++h) {
            const float* src = &x[idx2d(s, h * Dh, H * Dh)];
            float* dst = &out[((h * seq) + s) * Dh];
            for (size_t d = 0; d < Dh; ++d) dst[d] = src[d];
        }
    }
}

void merge_heads(const float* in, float* y, size_t seq, size_t H, size_t Dh) {
    for (size_t s = 0; s < seq; ++s) {
        for (size_t h = 0; h < H; ++h) {
            const float* src = &in[((h * seq) + s) * Dh];
            float* dst = &y[idx2d(s, h * Dh, H * Dh)];
            for (size_t d = 0; d < Dh; ++d) dst[d] = src[d];
        }
    }
}

// ---------- rotary positional embeddings (RoPE) ----------
void apply_rope_inplace(float* vec, size_t d, size_t pos, float base) {
    for (size_t i2 = 0; i2 < d; i2 += 2) {
        size_t i = i2 / 2;
        float inv = std::pow(base, -((float)i2) / (float)d);
        float angle = (float)pos * inv;
        float c = std::cos(angle);
        float s = std::sin(angle);
        float x0 = vec[i2];
        float x1 = vec[i2 + 1];
        vec[i2]     = x0 * c - x1 * s;
        vec[i2 + 1] = x0 * s + x1 * c;
    }
}

void apply_rope_batch_inplace(float* mat_rowmajor_seqXd, size_t seq, size_t d, size_t pos0, float base) {
    for (size_t t = 0; t < seq; ++t) {
        float* row = &mat_rowmajor_seqXd[idx2d(t, 0, d)];
        apply_rope_inplace(row, d, pos0 + t, base);
    }
}

} // namespace math
