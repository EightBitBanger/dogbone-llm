#pragma once
#include <cstddef>
#include <cstdint>

namespace math {

// ---------- helpers ----------
size_t idx2d(size_t r, size_t c, size_t cols);
float  rsqrt(float x);
float  clampf(float x, float lo, float hi);

// ---------- vector ops ----------
float dot(const float* a, const float* b, size_t n);
void  add(const float* a, const float* b, float* out, size_t n);
void  add_inplace(float* a, const float* b, size_t n);
void  sub(const float* a, const float* b, float* out, size_t n);
void  mul(const float* a, const float* b, float* out, size_t n);
void  scale_inplace(float* a, float s, size_t n);
void  copy(const float* src, float* dst, size_t n);
float l2_norm(const float* a, size_t n);

// ---------- matrix ops (row-major) ----------
void matmul(const float* A, const float* B, float* C,
            size_t M, size_t K, size_t N);
void matvec(const float* A, const float* x, float* y,
            size_t M, size_t N);
void matvec_bias(const float* A, const float* x, const float* b, float* y,
                 size_t M, size_t N);
void linear(const float* W_rowmajor_NxM, const float* x_N, const float* b_M,
            float* y_M, size_t N, size_t M);

// ---------- activations ----------
float sigmoid(float x);
void  sigmoid_inplace(float* x, size_t n);
float gelu(float x);
void  gelu_inplace(float* x, size_t n);
float silu(float x);
void  silu_inplace(float* x, size_t n);

// ---------- softmax family ----------
void softmax_inplace(float* x, size_t n);
void masked_softmax_inplace(float* x, const uint8_t* mask_keep, size_t n, float neg_inf);
void log_softmax_inplace(float* x, size_t n);

// ---------- normalization ----------
void layernorm_inplace(float* x, size_t n, const float* gamma, const float* beta, float eps);
void rmsnorm_inplace(float* x, size_t n, const float* weight, float eps);

// ---------- losses / grads ----------
float cross_entropy_loss(const float* logits, size_t n, int target);
float softmax_xent_grad(const float* logits, size_t n, int target, float* dlogits_out);

// ---------- attention bits (single-head, naive CPU) ----------
void causal_mask_inplace(float* scores_rowmajor, size_t Tq, size_t Tk, float neg_inf);
void attention_single_head(const float* Q, const float* K, const float* V,
                           float* out,
                           size_t Tq, size_t Tk, size_t D,
                           bool causal);

// ---------- multi-head pack/unpack ----------
void split_heads(const float* x, float* out, size_t seq, size_t H, size_t Dh);
void merge_heads(const float* in, float* y, size_t seq, size_t H, size_t Dh);

// ---------- rotary positional embeddings (RoPE) ----------
void apply_rope_inplace(float* vec, size_t d, size_t pos, float base);
void apply_rope_batch_inplace(float* mat_rowmajor_seqXd, size_t seq, size_t d, size_t pos0, float base);

} // namespace math
