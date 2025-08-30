#include "Tensor2D.h"
#include "LinearLayer.h"

#include <algorithm>
#include <cmath>

LinearLayer::LinearLayer() {}
LinearLayer::LinearLayer(int d_in, int d_out) : W(d_in, d_out), b((size_t)d_out, 0.0f) {
    float scale = 1.0f / std::sqrt((float)d_in);
    for (size_t i = 0; i < W.data.size(); i++) {
        W.data[i] = scale * ((float)std::rand() / (float)RAND_MAX - 0.5f);
    }
}

Tensor2D LinearLayer::Forward(const Tensor2D& x) const {
    Tensor2D y = MatMul(x, W);
    AddBiasRowInPlace(y, b);
    return y;
}
