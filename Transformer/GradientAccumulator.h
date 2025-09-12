#ifndef GRADIENT_ACCUMULATOR_H
#define GRADIENT_ACCUMULATOR_H

#include "Tensor2D.h"
#include "LanguageModel.h"

class GradientAccumulator {
public:
    
    Tensor2D d_tokW;
    Tensor2D d_posP;
    Tensor2D d_lmW;
    std::vector<float> d_lmb;
    
    struct BlockGrads {
        // LayerNorm grads (as vectors)
        std::vector<float> d_ln1g, d_ln1b;
        std::vector<float> d_ln2g, d_ln2b;
        // Attention linear layers
        Tensor2D dWq; std::vector<float> dbq;
        Tensor2D dWk; std::vector<float> dbk;
        Tensor2D dWv; std::vector<float> dbv;
        Tensor2D dWo; std::vector<float> dbo;
        // Feedforward layers
        Tensor2D d_fc1W; std::vector<float> d_fc1b;
        Tensor2D d_fc2W; std::vector<float> d_fc2b;
    };
    std::vector<BlockGrads> layers;
    
    void InitLike(const LanguageModel& model);
    
    void Add(const GradientAccumulator& other);
    
    void Scale(float s);
};

#endif
