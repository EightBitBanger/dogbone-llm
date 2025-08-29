#include "TransformerBlock.h"

TransformerBlock::TransformerBlock() {}

TransformerBlock::TransformerBlock(int d_model, int n_heads, int d_ff)
    : ln1(d_model), ln2(d_model), attn(d_model, n_heads), ffn(d_model, d_ff) {}

Tensor2D TransformerBlock::Forward(const Tensor2D& x_in, float* scratch) const {
    Tensor2D x = x_in;
    Tensor2D n1 = ln1.Forward(x);
    Tensor2D a = attn.Forward(n1, scratch);
    AddInPlace(x, a); // residual
    Tensor2D n2 = ln2.Forward(x);
    Tensor2D f = ffn.Forward(n2);
    AddInPlace(x, f); // residual
    return x;
}
