#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>

#include "Tensor2D.h"
#include "LinearLayer.h"
#include "LayerNormalization.h"
#include "TransformerLauguageModel.h"
#include "AdamOptimization.h"
#include "GradientAccumulator.h"

class NeuralNetwork {
public:
    
    Adam opt;
    
    // Adam states for top/bottom parameters
    AdamState tokW_s, posP_s, lmW_s, lmb_s;
    
    // Per-layer states
    struct BlockStates {
        AdamState ln1g_s, ln1b_s;
        AdamState ln2g_s, ln2b_s;
        AdamState WqW_s, Wqb_s;
        AdamState WkW_s, Wkb_s;
        AdamState WvW_s, Wvb_s;
        AdamState WoW_s, Wob_s;
        AdamState fc1W_s, fc1b_s;
        AdamState fc2W_s, fc2b_s;
    };
    std::vector<BlockStates> layer_states;
    
    // Attention scratch buffers
    std::vector<float> attnScratch_p;
    std::vector<float> attnScratch_gs;
    
    void InitScratch(int max_T);
    
    NeuralNetwork(float lr);
    
    void LinearBackward(const Tensor2D& x, const LinearLayer& lin, const Tensor2D& dy, Tensor2D& dW, std::vector<float>& db, Tensor2D& dx);
    
    void LayerNormBackward(const Tensor2D& x, const LayerNorm& ln, const Tensor2D& dy, std::vector<float>& dgamma, std::vector<float>& dbeta, Tensor2D& dx);
    
    // One step on a single (inputs, targets)
    float Step(TransformerLauguageModel& model, const std::vector<int>& inputs, const std::vector<int>& targets, 
               int pad_id, GradientAccumulator* acc = NULL, bool apply_updates = true);
    
    // Apply a pre-accumulated gradient in one Adam step (with optional scaling)
    void ApplyGradients(TransformerLauguageModel& model, const GradientAccumulator& G, float scale = 1.0f);
};

#endif
