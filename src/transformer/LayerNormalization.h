#ifndef LAYER_NORMALIZATION_H
#define LAYER_NORMALIZATION_H

#include "Tensor2D.h"

class LayerNorm {
public:
    
    std::vector<float> gamma;
    std::vector<float> beta;
    float eps;
    
    LayerNorm();
    LayerNorm(int d_model);
    
    Tensor2D Forward(const Tensor2D& x) const;
    
};

#endif
