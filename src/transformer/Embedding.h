#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <cstdlib>
#include <cstring>

#include "Tensor2D.h"

struct Embedding {
    Tensor2D W;  // weights: [vocab_size, d_model]
    
    Embedding();
    Embedding(int vocab_size, int d_model);
    
    // Input ids: [T], output: [T, d_model]
    Tensor2D Forward(const std::vector<int>& ids) const;
};

#endif
