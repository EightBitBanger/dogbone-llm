#include "Tensor2D.h"
#include "PositionalEncoding.h"

#include <algorithm>

PositionalEncoding::PositionalEncoding() {}
PositionalEncoding::PositionalEncoding(int max_T, int d_model) : P(max_T, d_model) {
    float scale = 0.02f;
    for (size_t i = 0; i < P.data.size(); i++) {
        P.data[i] = scale * ((float)std::rand() / (float)RAND_MAX - 0.5f);
    }
}

void PositionalEncoding::AddInPlace(Tensor2D& x) const {
    int limit = (x.R <= P.R) ? x.R : P.R;
    for (int t = 0; t < limit; t++) {
        const float* pe = P.Row(t);
        float* row = x.Row(t);
        for (int c = 0; c < x.C; c++) row[c] += pe[c];
    }
}
