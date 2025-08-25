#ifndef TENSOR_H
#define TENSOR_H

struct Tensor2D {
    // Shape: [R, C], row-major
    int R;
    int C;
    std::vector<float> data;

    Tensor2D() : R(0), C(0) {}
    Tensor2D(int r, int c) : R(r), C(c), data((size_t)r * (size_t)c) {}

    float* Row(int r) { return &data[(size_t)r * (size_t)C]; }
    const float* Row(int r) const { return &data[(size_t)r * (size_t)C]; }

    void Zero() {
        std::fill(data.begin(), data.end(), 0.0f);
    }
};

// y = a @ b   (a:[M,K], b:[K,N]) -> y:[M,N]
static Tensor2D MatMul(const Tensor2D& a, const Tensor2D& b) {
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

static void AddInPlace(Tensor2D& y, const Tensor2D& x) {
    for (size_t i = 0; i < y.data.size(); i++) y.data[i] += x.data[i];
}

static void AddBiasRowInPlace(Tensor2D& y, const std::vector<float>& b) {
    for (int r = 0; r < y.R; r++) {
        for (int c = 0; c < y.C; c++) {
            y.data[(size_t)r * y.C + c] += b[(size_t)c];
        }
    }
}

#endif
