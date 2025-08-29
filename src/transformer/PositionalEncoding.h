#ifndef POSITIONAL_ENCODING_H
#define POSITIONAL_ENCODING_H

#include "Tensor2D.h"

class PositionalEncoding {
public:
    
    Tensor2D P;
    
    PositionalEncoding();
    PositionalEncoding(int max_T, int d_model);
    
    void AddInPlace(Tensor2D& x) const;
};

#endif
