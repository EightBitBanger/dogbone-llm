#include "LayerNormalization.h"

#include <cmath>

LayerNorm::LayerNorm() : eps(1e-5f) {}
LayerNorm::LayerNorm(int d_model) : gamma(d_model, 1.0f), beta(d_model, 0.0f), eps(1e-5f) {}

Tensor2D LayerNorm::Forward(const Tensor2D& x) const {
    Tensor2D y(x.R, x.C);
    for (int t = 0; t < x.R; t++) {
        const float* xr = x.Row(t);
        float* yr = y.Row(t);
        
        float mean = 0.0f;
        for (int c = 0; c < x.C; c++) mean += xr[c];
        mean /= (float)x.C;
        
        float var = 0.0f;
        for (int c = 0; c < x.C; c++) {
            float d = xr[c] - mean;
            var += d * d;
        }
        var /= (float)x.C;
        float inv = 1.0f / std::sqrt(var + eps);
        
        for (int c = 0; c < x.C; c++) {
            float z = (xr[c] - mean) * inv;
            yr[c] = z * gamma[(size_t)c] + beta[(size_t)c];
        }
    }
    return y;
}
