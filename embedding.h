#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "tensor.h"

struct Embedding {
    // weights: [vocab_size, d_model]
    Tensor2D W;
    Embedding() {}
    Embedding(int vocab_size, int d_model) : W(vocab_size, d_model) {
        // init: small uniform
        float scale = 0.02f;
        for (size_t i = 0; i < W.data.size(); i++) {
            W.data[i] = scale * ((float)std::rand() / (float)RAND_MAX - 0.5f);
        }
    }
    // input ids: [T], output: [T, d_model]
    Tensor2D Forward(const std::vector<int>& ids) const {
        int T = (int)ids.size();
        Tensor2D out(T, W.C);
        for (int t = 0; t < T; t++) {
            int id = ids[(size_t)t];
            if (id < 0 || id >= W.R) id = 0;
            const float* row = W.Row(id);
            std::memcpy(out.Row(t), row, sizeof(float) * (size_t)W.C);
        }
        return out;
    }
};

#endif
