#ifndef TEST_SUITE_H
#define TEST_SUITE_H

#include <iostream>
#include <cmath>
#include <cstdint>

bool TestTensor2D();
bool TestLinearLayer();
bool TestLayerNorm();
bool TestMultiHeadSelfAttention();
bool TestPositionalEncoding();
bool TestTransformerBlock();
bool TestGradientAccumulator();
bool TestAdam();

bool TestGELU_FwdBwd();
bool TestReLU_FwdBwd();
bool TestSiLU_FwdBwd();
bool TestMish_FwdBwd();
bool TestSwiGLU_FwdBwd();
bool TestActivationSelectorRouting();

bool TestCrossEntropyLoss();

static void RunTests() {
    std::cout << "Tensor2D                  ";
    if (TestTensor2D()) {std::cout << "passed\n";} else {std::cout << "failed\n";}
    std::cout << "LayerNorm                 ";
    if (TestLayerNorm()) {std::cout << "passed\n";} else {std::cout << "failed\n";}
    std::cout << "LinearLayer               ";
    if (TestLinearLayer()) {std::cout << "passed\n";} else {std::cout << "failed\n";}
    std::cout << "MultiHeadSelfAttention    ";
    if (TestMultiHeadSelfAttention()) {std::cout << "passed\n";} else {std::cout << "failed\n";}
    std::cout << "TransformerBlock          ";
    if (TestTransformerBlock()) {std::cout << "passed\n";} else {std::cout << "failed\n";}
    std::cout << "PositionalEncoding        ";
    if (TestPositionalEncoding()) {std::cout << "passed\n";} else {std::cout << "failed\n";}
    std::cout << "GradientAccumulator       ";
    if (TestGradientAccumulator()) {std::cout << "passed\n";} else {std::cout << "failed\n";}
    std::cout << "AdamOptimization          ";
    if (TestAdam()) {std::cout << "passed\n";} else {std::cout << "failed\n";}
    
    std::cout << "Activation ReLU           ";
    if (TestReLU_FwdBwd()) {std::cout << "passed\n";} else {std::cout << "failed\n";}
    std::cout << "Activation GELU           ";
    if (TestGELU_FwdBwd()) {std::cout << "passed\n";} else {std::cout << "failed\n";}
    std::cout << "Activation Mish           ";
    if (TestMish_FwdBwd()) {std::cout << "passed\n";} else {std::cout << "failed\n";}
    std::cout << "Activation SiLU           ";
    if (TestSiLU_FwdBwd()) {std::cout << "passed\n";} else {std::cout << "failed\n";}
    std::cout << "Activation SwiGLU         ";
    if (TestSwiGLU_FwdBwd()) {std::cout << "passed\n";} else {std::cout << "failed\n";}
    std::cout << "Activation selection      ";
    if (TestActivationSelectorRouting()) {std::cout << "passed\n";} else {std::cout << "failed\n";}
    
    std::cout << "Cross entropy loss        ";
    if (TestCrossEntropyLoss()) {std::cout << "passed\n";} else {std::cout << "failed\n";}
}

#endif
