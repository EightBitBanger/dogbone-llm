#ifndef SAMPLER_H
#define SAMPLER_H

#include <vector>
#include "Transformer/Transformer.h"

struct SamplingParams {
    float temperature;        // >0, 1.0 = neutral, <1 = sharper, >1 = more random
    int   top_k;              // 0 or negative = disabled; otherwise keep best k
    float top_p;              // 0<top_p<=1; 1.0 = disabled; keep smallest set with a cumulative probability >= top_p
    float presence_penalty;   // >=0; add negative bias to any token that has appeared at least once
    float frequency_penalty;  // >=0; subtract count*penalty from logits
    unsigned int seed;        // used with std::srand
    
    SamplingParams()
    : temperature(1.0f), top_k(0), top_p(1.0f),
      presence_penalty(0.0f), frequency_penalty(0.0f),
      seed(42u) {}
};

struct TokenCandidate {
    int   id;              // Token word index
    float prob;            // normalized probability (optionally re-normalized over returned set) usually between 10 - -20 ish
    float logit;           // pre-softmax score after penalties/temperature/top-k
    float cumulative_prob; // cumulative probability in the returned, sorted list
};

class TokenSampler {
public:
    
    // Returns the index of the largest value in `row[0..V-1]`.
    int  ArgMax(const float* row, int V);
    
    // Computes a numerically stable softmax of `logits` into `probs` (resizes `probs`).
    void SoftmaxStable(const std::vector<float>& logits, std::vector<float>& probs);
    
    // Samples an index from a categorical distribution described by `probs` (sum ~ 1).
    int  SampleFromProbs(const std::vector<float>& probs);
    
    // Applies repetition penalties to `logits` based on `context_ids`:
    //  - presence_penalty: subtract if a token appeared at least once
    //  - frequency_penalty: subtract count * penalty per token
    void ApplyRepetitionPenalties(std::vector<float>& logits, const std::vector<int>& context_ids, float presence_penalty, float frequency_penalty);
    
    // Keeps only the top `k` logits; all others are set to a very negative value (suppressed).
    void ApplyTopK(std::vector<float>& logits, int k);
    
    // Truncates to the smallest set of tokens whose cumulative probability >= `top_p`, then renormalizes.
    void ApplyTopP(std::vector<float>& probs, float top_p);
    
    // Generates the next token id given `model` and `context_ids`, using `P`:
    //  - temperature <= 0: greedy ArgMax
    //  - otherwise: applies penalties, temperature, optional top-k/top-p, then samples.
    int  GetNextToken(const LauguageModel& model, const std::vector<int>& context_ids, const SamplingParams& P);
    
    // Generates a list of the next probable tokens.
    std::vector<TokenCandidate> GetProbableTokens(const LauguageModel& model, const std::vector<int>& context_ids, 
                                                  const SamplingParams& P, int max_candidates, float min_prob, bool renormalize);
};

extern TokenSampler Sampler;
#endif
