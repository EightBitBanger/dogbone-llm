#ifndef LANGUAGE_MODEL_CONTEXT_STREAM_H
#define LANGUAGE_MODEL_CONTEXT_STREAM_H

#include "Transformer/NeuralNetwork.h"
#include "Transformer/LanguageModel.h"
#include "ContextWindow.h"
#include "Tokenizer.h"
#include "Sampler.h"
#include "SemanticCoherence.h"
#include "Modelio.h"

#include "main.h"

#include <string>
#include <vector>

// A thin convenience wrapper that owns a (model, vocab, context) trio
// and exposes a few helpers for loading, saving, feeding text, and sampling.
class LanguageModelContextStream {
public:
    
    ContextWindow context;
    
    LanguageModelContextStream(int context_size);
    
    Tokenizer&      Vocab();
    LanguageModel&  Model();
    ContextWindow&  Context();
    
    int GetMaxContext() const;
    void SetMaxContext(int new_size);
    
    /// Reset just the rolling context buffer.
    void ClearContext();
    
    /// Unload model and vocabulary and reset context.
    void Unload();
    
    /// Load / Save using existing package helpers.
    /// Returns true on success.
    bool LoadPackage(const std::string& filename, NeuralNetwork& trainer, uint64_t& epoch, float& learnRate, float& avgLoss);
    bool SavePackage(const std::string& filename, NeuralNetwork& trainer, uint64_t epoch, float learnRate, float avgLoss);
    
    /// Add raw text into the rolling context using this instance's vocabulary.
    /// Returns number of tokens added.
    size_t FeedString(const std::string& text);
    
    /// Generate a stream of tokens within the bound specified by the sentence structure.
    /// Returns the decoded text that was appended to the rolling context.
    std::string GenerateStream(TokenSampler& Sampler, SamplingParams& sampler, ContextWindow& contextWindow, SentenceStructure& structure, int tokenCountMax);
    
    bool IsReady() const;
    
private:
    int mContextSize;
    
    Tokenizer vocabulary;
    LanguageModel model;
    SemanticCoherence semantic;
};

#endif
