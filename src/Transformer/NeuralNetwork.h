#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <unordered_map>

#include "Tensor2D.h"
#include "LinearLayer.h"
#include "LayerNormalization.h"
#include "LauguageModel.h"
#include "AdamOptimization.h"
#include "GradientAccumulator.h"
#include "ShaderTensor.h"

class NeuralNetwork {
public:
    
    Adam opt;
    
    // Adam states for top/bottom parameters
    AdamState tokW_s, posP_s, lmW_s, lmb_s;
    
    bool g_matmul_built = false;
    
    bool g_attn_built = false;
    bool g_attn_apply_built = false;
    
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
    
    // Pre-Allocate Attention Scratch Buffers Up To max_T Tokens.
    void InitScratch(int max_T);
    
    // Construct Trainer/Optimizer Wrapper With A Base Learning Rate.
    NeuralNetwork(float lr);
    
    // Compute Fully-Connected Backward: Gradients For Weights, Bias, And Inputs.
    void LinearBackward(const Tensor2D& x, const LinearLayer& lin, const Tensor2D& dy, Tensor2D& dW, std::vector<float>& db, Tensor2D& dx);
    
    // Compute LayerNorm Backward Per Row With Affine Scale (Gamma) And Bias (Beta).
    void LayerNormBackward(const Tensor2D& x, const LayerNorm& ln, const Tensor2D& dy, std::vector<float>& dgamma, std::vector<float>& dbeta, Tensor2D& dx);
    
    // Run Forward + Backward For One Sequence And Optionally Apply Adam Updates.
    float Step(LauguageModel& model, const std::vector<int>& inputs, const std::vector<int>& targets, 
               int pad_id, GradientAccumulator* acc = NULL, bool apply_updates = true);
    
    // Runs one training step using GPU-accelerated forwards where available.
    // Falls back to CPU internally if the GPU path is unavailable.
    float StepGPU(LauguageModel& model, const std::vector<int>& inputs, const std::vector<int>& targets, int pad_id, GradientAccumulator* acc, bool apply_updates);
    float StepGPU_Batched(LauguageModel& model,
                          const std::vector<std::vector<int>>& inputs_list,
                          const std::vector<std::vector<int>>& targets_list,
                          int pad_id,
                          GradientAccumulator* acc,
                          bool apply_updates);
    
    // Apply Accumulated Gradients In A Single Adam Step (Optional Scaling).
    void ApplyGradients(LauguageModel& model, const GradientAccumulator& G, float scale = 1.0f);
    
    // Build required shaders.
    void BuildShaders();
    
    // Attach a ShaderTensor for optional GPU acceleration (pass nullptr to disable).
    void UseGPU(ShaderTensor* st);

    // Upload all model weights once into GPU SSBOs (resident mode).
    void UploadWeightsToGPU(LauguageModel& model);
    // Refresh resident SSBOs from current CPU weights (call after optimizer step).
    void RefreshGPUWeightsFromModel(const LauguageModel& model);
    // Release all resident buffers.
    void ReleaseGPUWeights();
    // Toggle resident-weight path.
    void EnableResidentWeights(bool enable);

    struct GPUResident {
        bool enabled = false;
        struct WeightBuf { unsigned w=0, b=0; std::ptrdiff_t wBytes=0, bBytes=0; int IN=0, OUT=0; };
        std::unordered_map<const LinearLayer*, WeightBuf> map;
    } mGPUResident;
    
    
    
    // Batched GPU linear forward: concatenate many [Ti x IN] into one, run once on GPU, then split back.
    // Returns true if GPU path was used; falls back to CPU per-slice and still returns outputs.
    bool LinearForwardGPU_Batched(const LinearLayer& L,
                                  const std::vector<Tensor2D>& Xs,
                                  std::vector<Tensor2D>& Ys);
    
private:

    ShaderTensor* mGpu = nullptr;
};

#endif
