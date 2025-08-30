#include "Tensor2D.h"
#include "Activation.h"

#include <algorithm>
#include <cmath>

void GELU_InPlace(Tensor2D& x) {
    const float k = std::sqrt(2.0f / 3.14159265358979323846f);
    for (size_t i = 0; i < x.data.size(); i++) {
        float v = x.data[i];
        float v3 = v * v * v;
        float t = std::tanh(k * (v + 0.044715f * v3));
        x.data[i] = 0.5f * v * (1.0f + t);
    }
}

void GELU_Backward(const Tensor2D& x, const Tensor2D& dy, Tensor2D& dx) {
    const float k = std::sqrt(2.0f / 3.14159265358979323846f);
    dx = Tensor2D(x.R, x.C);
    for (int i = 0; i < x.R * x.C; i++) {
        float v = x.data[(size_t)i];
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
