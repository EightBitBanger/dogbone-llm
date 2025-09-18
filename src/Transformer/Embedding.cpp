#include "Embedding.h"
#include <cmath>
#include <cstring>
#include <cstdlib>

Embedding::Embedding() {}

Embedding::Embedding(int vocab_size, int d_model) : W(vocab_size, d_model) {
    float scale = 0.02f; // Initialize to a small uniform
    for (size_t i = 0; i < W.data.size(); i++) {
        W.data[i] = scale * ((float)std::rand() / (float)RAND_MAX - 0.5f);
    }
}

Tensor2D Embedding::Forward(const std::vector<int>& ids) const {
    int T = (int)ids.size();
    Tensor2D out(T, W.C);
    for (int t = 0; t < T; t++) {
        int id = ids[(size_t)t];
        if (id < 0 || id >= W.R) id = 0;
        const float* row = W.Row(id);
        std::memcpy(out.Row(t), row, sizeof(float) * (size_t)W.C);
        // Scale token embeddings by sqrt(d_model) to balance against positional encodings
        float* out_row = out.Row(t);
        float s = std::sqrt((float)W.C);
        for (int c = 0; c < W.C; ++c) out_row[c] *= s;
    }
    return out;
}
