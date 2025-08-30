#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

class Tensor2D {
public:
    
    int R; // Shape: [R, C], row-major
    int C;
    std::vector<float> data;
    
    Tensor2D();
    Tensor2D(int r, int c);
    
    float* Row(int r);
    const float* Row(int r) const;
    
    void Zero();
};

Tensor2D MatMul(const Tensor2D& a, const Tensor2D& b);

void AddInPlace(Tensor2D& y, const Tensor2D& x);

void AddBiasRowInPlace(Tensor2D& y, const std::vector<float>& b);

#endif
