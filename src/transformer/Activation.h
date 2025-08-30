#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

// GELU approx and backward
void GELU_InPlace(Tensor2D& x);
void GELU_Backward(const Tensor2D& x, const Tensor2D& dy, Tensor2D& dx);

#endif
