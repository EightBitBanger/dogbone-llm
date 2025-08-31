#include "FeedForward.h"

FeedForward::FeedForward() {}
FeedForward::FeedForward(int d_model, int d_ff) : fc1(d_model, Activation.d_ff_mul * d_ff), fc2(d_ff, d_model) {}

Tensor2D FeedForward::Forward(const Tensor2D& x) const {
    Tensor2D x_concat = fc1.Forward(x);         // [T, d_ff_mul * d_ff]
    Tensor2D h = Activation.Forward(x_concat);  // [T, d_ff]
    Tensor2D y = fc2.Forward(h);                // [T, d_model]
    return y;
}
