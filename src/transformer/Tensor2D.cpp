#include "Tensor2D.h"

Tensor2D::Tensor2D() : R(0), C(0) {}
Tensor2D::Tensor2D(int r, int c) : R(r), C(c), data((size_t)r * (size_t)c) {}

float* Tensor2D::Row(int r) { return &data[(size_t)r * (size_t)C]; }
const float* Tensor2D::Row(int r) const {return &data[(size_t)r * (size_t)C]; }

void Tensor2D::Zero() {std::fill(data.begin(), data.end(), 0.0f);}

Tensor2D MatMul(const Tensor2D& a, const Tensor2D& b) {
    Tensor2D y(a.R, b.C);
    for (int m = 0; m < a.R; m++) {
        for (int n = 0; n < b.C; n++) {
            float sum = 0.0f;
            for (int k = 0; k < a.C; k++) {
                sum += a.data[(size_t)m * a.C + k] * b.data[(size_t)k * b.C + n];
            }
            y.data[(size_t)m * b.C + n] = sum;
        }
    }
    return y;
}

void AddInPlace(Tensor2D& y, const Tensor2D& x) {
    for (size_t i = 0; i < y.data.size(); i++) y.data[i] += x.data[i];
}

void AddBiasRowInPlace(Tensor2D& y, const std::vector<float>& b) {
    for (int r = 0; r < y.R; r++) {
        for (int c = 0; c < y.C; c++) {
            y.data[(size_t)r * y.C + c] += b[(size_t)c];
        }
    }
}
