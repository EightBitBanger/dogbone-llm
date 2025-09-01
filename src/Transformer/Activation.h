#pragma once
#include "Tensor2D.h"
#include <vector>

enum class ActivationType {GELU, RELU, SILU, MISH, SWIGLU};


class ActivationFunctions {
public:
    
    // Model width multiplier. SwiGLU = 2.
    int d_ff_mul;
    
    // Set the activation function by type
    bool SetActivationFunction(ActivationType type);
    
    Tensor2D (*Forward)(const Tensor2D&);
    void     (*Backward)(const Tensor2D&, const Tensor2D&, Tensor2D&);
    
    ActivationFunctions();
    
private:
    
    // GELU
    static Tensor2D GELU_Forward(const Tensor2D& x);
    static void GELU_Backward(const Tensor2D& x_pre, const Tensor2D& dy, Tensor2D& dx);
    
    // ReLU
    static Tensor2D ReLU_Forward(const Tensor2D& x);
    static void ReLU_Backward(const Tensor2D& x_pre, const Tensor2D& dy, Tensor2D& dx);
    
    // SiLU (Swish)
    static Tensor2D SiLU_Forward(const Tensor2D& x);
    static void SiLU_Backward(const Tensor2D& x_pre, const Tensor2D& dy, Tensor2D& dx);
    
    // Mish
    static Tensor2D Mish_Forward(const Tensor2D& x);
    static void Mish_Backward(const Tensor2D& x_pre, const Tensor2D& dy, Tensor2D& dx);
    
    // SwiGLU (concat of a and b; returns a * SiLU(b))
    static Tensor2D SwiGLU_Forward(const Tensor2D& x_concat);
    static void SwiGLU_Backward(const Tensor2D& x_concat, const Tensor2D& dy_half, Tensor2D& dx_concat);
};

extern ActivationFunctions Activation;
