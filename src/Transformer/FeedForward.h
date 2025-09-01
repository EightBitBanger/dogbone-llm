#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H

#include "LinearLayer.h"
#include "Activation.h"

class FeedForward {
public:
    
    LinearLayer fc1; // d_model -> d_ff
    LinearLayer fc2; // d_ff -> d_model
    
    FeedForward();
    FeedForward(int d_model, int d_ff);
    
    Tensor2D Forward(const Tensor2D& x) const;
};

#endif
