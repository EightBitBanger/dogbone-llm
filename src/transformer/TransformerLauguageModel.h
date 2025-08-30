#ifndef TRANSFORMER_LAUGUAGE_MODEL_H
#define TRANSFORMER_LAUGUAGE_MODEL_H

#include <fstream>
#include <vector>
#include <string>

#include "Embedding.h"
#include "PositionalEncoding.h"
#include "TransformerBlock.h"
#include "LinearLayer.h"
#include "Tensor2D.h"

class TransformerLauguageModel {
public:
    
    int vocab_size;
    int d_model;
    int n_heads;
    int d_ff;
    int n_layers;
    int max_T;
    
    Embedding tok;
    PositionalEncoding pos;
    std::vector<TransformerBlock> layers;
    LinearLayer lm_head; // d_model -> vocab_size
    
    TransformerLauguageModel();
    TransformerLauguageModel(int vocab, int dmodel, int heads, int ff, int layers_count, int maxT);
    
    Tensor2D Forward(const std::vector<int>& ids) const;
    Tensor2D Forward(const std::vector<int>& ids, float* scratch) const;
    
    bool SaveToStream(std::ostream& out) const;
    bool LoadFromStream(std::istream& in);
    
    bool Save(const std::string& path) const;
    bool Load(const std::string& path);
};

#endif
