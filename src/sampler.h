#ifndef SAMPLER_H
#define SAMPLER_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "Transformer/Transformer.h"

static int ArgMax(const float* row, int V) {
    int best = 0;
    float bv = row[0];
    for (int v = 1; v < V; v++) {
        if (row[v] > bv) { bv = row[v]; best = v; }
    }
    return best;
}

// Stable softmax into `probs` from `logits` (length V).
static void SoftmaxStable(const std::vector<float>& logits, std::vector<float>& probs) {
    const int V = (int)logits.size();
    probs.resize((size_t)V);
    float maxv = logits[0];
    for (int i = 1; i < V; i++) if (logits[i] > maxv) maxv = logits[i];

    float denom = 0.0f;
    for (int i = 0; i < V; i++) {
        float e = std::exp(logits[i] - maxv);
        probs[(size_t)i] = e;
        denom += e;
    }
    if (denom <= 0.0f) {
        float inv = 1.0f / (float)V;
        for (int i = 0; i < V; i++) probs[(size_t)i] = inv;
    } else {
        float inv = 1.0f / denom;
        for (int i = 0; i < V; i++) probs[(size_t)i] *= inv;
    }
}

// Sample an index from probs (assumed normalized).
static int SampleFromProbs(const std::vector<float>& probs) {
    float r = (float)std::rand() / (float)RAND_MAX;
    float c = 0.0f;
    const int V = (int)probs.size();
    for (int i = 0; i < V; i++) {
        c += probs[(size_t)i];
        if (r <= c) return i;
    }
    return V - 1;
}

struct SamplingParams {
    float temperature;        // >0, 1.0 = neutral, <1 = sharper, >1 = more random
    int   top_k;              // 0 or negative = disabled; otherwise keep best k
    float top_p;              // 0<top_p<=1; 1.0 = disabled; keep smallest set with cumprob >= top_p
    float presence_penalty;   // >=0; add negative bias to any token that has appeared at least once
    float frequency_penalty;  // >=0; subtract count*penalty from logits
    unsigned int seed;        // used with std::srand

    SamplingParams()
    : temperature(1.0f), top_k(0), top_p(1.0f),
      presence_penalty(0.0f), frequency_penalty(0.0f),
      seed(42u) {}
};

// Apply penalties to logits based on the context.
// presence: if token appeared at least once, subtract presence_penalty
// frequency: subtract count * frequency_penalty
static void ApplyRepetitionPenalties(std::vector<float>& logits,
                                     const std::vector<int>& context_ids,
                                     float presence_penalty,
                                     float frequency_penalty) {
    if (presence_penalty <= 0.0f && frequency_penalty <= 0.0f) return;
    // Count occurrences
    // (For speed you could cap to last N tokens; here we use full context.)
    std::unordered_map<int, int> counts;
    for (size_t i = 0; i < context_ids.size(); i++) {
        int id = context_ids[i];
        if (id >= 0) counts[id] += 1;
    }
    if (counts.empty()) return;

    for (std::unordered_map<int,int>::const_iterator it = counts.begin(); it != counts.end(); ++it) {
        int tok = it->first;
        int cnt = it->second;
        if (tok < 0 || tok >= (int)logits.size()) continue;
        float delta = 0.0f;
        if (presence_penalty > 0.0f && cnt > 0) delta += presence_penalty;
        if (frequency_penalty > 0.0f)           delta += frequency_penalty * (float)cnt;
        logits[(size_t)tok] -= delta;
    }
}

// Keep only the largest-k logits; others set to -inf (approx: very negative).
static void ApplyTopK(std::vector<float>& logits, int k) {
    const int V = (int)logits.size();
    if (k <= 0 || k >= V) return;
    // Find kth largest threshold
    std::vector<float> buf = logits;
    std::nth_element(buf.begin(), buf.begin() + (size_t)(V - k), buf.end());
    float thresh = buf[(size_t)(V - k)];
    const float NEG_INF = -1e30f;
    for (int i = 0; i < V; i++) {
        if (logits[(size_t)i] < thresh) logits[(size_t)i] = NEG_INF;
    }
}

// After softmax, zero out tail beyond cumulative probability top_p and renormalize.
static void ApplyTopP(std::vector<float>& probs, float top_p) {
    if (top_p >= 1.0f) return;
    if (top_p <= 0.0f) top_p = 1e-6f;

    const int V = (int)probs.size();
    // Sort indices by prob desc
    std::vector<int> idx((size_t)V);
    for (int i = 0; i < V; i++) idx[(size_t)i] = i;
    std::sort(idx.begin(), idx.end(), [&probs](int a, int b) {
        return probs[(size_t)a] > probs[(size_t)b];
    });

    float cum = 0.0f;
    int cut = V;
    for (int i = 0; i < V; i++) {
        cum += probs[(size_t)idx[(size_t)i]];
        if (cum >= top_p) { cut = i + 1; break; }
    }

    float renorm = 0.0f;
    for (int i = cut; i < V; i++) probs[(size_t)idx[(size_t)i]] = 0.0f;
    for (int i = 0; i < cut; i++) renorm += probs[(size_t)idx[(size_t)i]];
    if (renorm > 0.0f) {
        float inv = 1.0f / renorm;
        for (int i = 0; i < cut; i++) probs[(size_t)idx[(size_t)i]] *= inv;
    } else {
        // Degenerate: uniform
        float inv = 1.0f / (float)V;
        for (int i = 0; i < V; i++) probs[(size_t)i] = inv;
    }
}

// Return next token ID using configured strategy.
// If temperature <= 0 -> falls back to greedy.
static int SampleNextToken(const TransformerLauguageModel& model,
                           const std::vector<int>& context_ids,
                           const SamplingParams& P) {
    if (P.seed != 0u) std::srand(P.seed + (unsigned int)context_ids.size());

    // Forward to get logits for the current context
    Tensor2D logits_t = model.Forward(context_ids);
    const int V = logits_t.C;
    const float* last = logits_t.Row(logits_t.R - 1);

    // Copy logits to a working buffer
    std::vector<float> logits((size_t)V);
    for (int i = 0; i < V; i++) logits[(size_t)i] = last[i];

    // Repetition penalties
    ApplyRepetitionPenalties(logits, context_ids, P.presence_penalty, P.frequency_penalty);

    // Temperature / Greedy
    if (P.temperature <= 0.0f) {
        // Greedy pick
        return ArgMax(&logits[0], V);
    } else {
        // Scale logits by 1/temperature
        float invT = 1.0f / P.temperature;
        for (int i = 0; i < V; i++) logits[(size_t)i] *= invT;

        // Top-k
        if (P.top_k > 0) ApplyTopK(logits, P.top_k);

        // Softmax
        std::vector<float> probs;
        SoftmaxStable(logits, probs);

        // Top-p
        if (P.top_p < 1.0f) ApplyTopP(probs, P.top_p);

        // Sample
        return SampleFromProbs(probs);
    }
}

#endif
