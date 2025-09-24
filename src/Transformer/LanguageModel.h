#ifndef TRANSFORMER_LANGUAGE_MODEL_H
#define TRANSFORMER_LANGUAGE_MODEL_H

#include <fstream>
#include <vector>
#include <string>

#include "Embedding.h"
#include "PositionalEncoding.h"
#include "TransformerBlock.h"
#include "LinearLayer.h"
#include "Tensor2D.h"

class LanguageModel {
public:
    
    int vocab_size;
    int n_ctx;
    int d_model;
    int n_heads;
    int d_ff;
    int n_layers;
    
    bool tie_weights;
    
    Embedding tok;
    PositionalEncoding pos;
    std::vector<TransformerBlock> layers;
    LinearLayer lm_head;
    
    LanguageModel();
    LanguageModel(int vocab, int dmodel, int heads, int ff, int layers_count, int ctx);
    
    Tensor2D Forward(const std::vector<int>& ids) const;
    Tensor2D Forward(const std::vector<int>& ids, float* scratch) const;
    
    bool SaveToStream(std::ostream& out) const;
    bool LoadFromStream(std::istream& in);
    
    bool Save(const std::string& path) const;
    bool Load(const std::string& path);
};

#endif
