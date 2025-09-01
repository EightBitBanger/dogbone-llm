#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include "LayerNormalization.h"
#include "FeedForward.h"
#include "MultiHeadSelfAttention.h"

class TransformerBlock {
public:
    
    LayerNorm ln1;
    LayerNorm ln2;
    MultiHeadSelfAttention attn;
    FeedForward ffn;
    
    TransformerBlock();
    TransformerBlock(int d_model, int n_heads, int d_ff);
    
    Tensor2D Forward(const Tensor2D& x_in, float* scratch) const;
};

#endif
