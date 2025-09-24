#ifndef MULTI_HEAD_SELF_ATTENTION_H
#define MULTI_HEAD_SELF_ATTENTION_H

#include "LinearLayer.h"

class MultiHeadSelfAttention {
public:
    
    int d_model;
    int n_heads;
    int d_head;
    
    LinearLayer Wq;
    LinearLayer Wk;
    LinearLayer Wv;
    LinearLayer Wo;
    
    MultiHeadSelfAttention();
    MultiHeadSelfAttention(int dmodel, int heads);
    
    Tensor2D Forward(const Tensor2D& x, float* scratch) const;
};

#endif
