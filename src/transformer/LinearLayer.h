#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include <vector>
#include "Tensor2D.h"

class LinearLayer {
public:
    
    Tensor2D W;             // [d_in, d_out]
    std::vector<float> b;   // [d_out]
    
    LinearLayer();
    LinearLayer(int d_in, int d_out);
    
    Tensor2D Forward(const Tensor2D& x) const;
};

#endif
