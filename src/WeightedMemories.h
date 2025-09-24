#ifndef WEIGHTED_REINFORCEMENT_MEMORY_H
#define WEIGHTED_REINFORCEMENT_MEMORY_H

#include <unordered_map>
#include <vector>
#include <string>
#include <limits>
#include <stdint.h>

#include "Sampler.h"
#include "ContextWindow.h"


struct WRM_Suggestion {
    int next_token;
    float score;
    std::vector<int> matched_sequence; // includes the prefix + next_token
};

class WeightedReinforcementMemory {
public:
    // maxNGram: highest n-gram length to remember (>=2 recommended)
    // capacity: soft cap on unique sequences to keep (we prune when exceeded)
    // decay   : multiplicative weight decay applied on Observe() calls (0..1)
    WeightedReinforcementMemory(unsigned int maxNGram = 5,
                                unsigned int capacity = 20000,
                                float decay = 0.9975f);
    
    // Learn from the tail of a context window (uses up to maxNGram tokens).
    // Notes: relies on ContextWindow::GetContext() and ContextWindow::Size().
    void Observe(ContextWindow& context);
    
    // Direct reinforcement (useful for external nudges or rewards).
    void Reinforce(const std::vector<int>& seq, float amount);
    
    // Given a tail (prefix) of length L (1..maxNGram-1), return up to topK
    // likely next tokens based on learned n-grams that start with that prefix.
    std::vector<WRM_Suggestion> SuggestNext(const std::vector<int>& prefix,
                                            unsigned int topK = 5) const;
    
    // Clear all memory.
    void Clear();
    
    // Optional: nudge global time forward without observing (applies decay).
    void Tick(unsigned int steps = 1);
    
    // Housekeeping knobs
    void SetCapacity(unsigned int cap);
    void SetDecay(float d);
    unsigned int GetCapacity() const;
    float GetDecay() const;
    unsigned int GetMaxNGram() const;
    
    // Dump top-N memories to std::cout (sorted by weight, desc)
    void DumpToConsole(unsigned int maxItems = 50) const;
    
    // Keying n-grams as comma-joined strings keeps things simple.
    static std::string KeyFrom(const std::vector<int>& seq);
    static std::string KeyFromSpan(const std::vector<int>& tokens,
                                size_t start, size_t len);
    
    void DecayAll_();
    void PruneIfNeeded_();
    
private:
    std::unordered_map<std::string, float> mWeights;     // sequence -> weight
    std::unordered_map<std::string, uint32_t> mLastSeen; // sequence -> step
    unsigned int mMaxNGram;
    unsigned int mCapacity;
    float mDecay;
    uint32_t mStepCounter;
};

#endif
