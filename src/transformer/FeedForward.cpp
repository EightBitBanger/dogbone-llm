#include "FeedForward.h"

FeedForward::FeedForward() {}
FeedForward::FeedForward(int d_model, int d_ff) : fc1(d_model, d_ff), fc2(d_ff, d_model) {}

Tensor2D FeedForward::Forward(const Tensor2D& x) const {
    Tensor2D h = fc1.Forward(x);
    GELU_InPlace(h);
    Tensor2D y = fc2.Forward(h);
    return y;
}
