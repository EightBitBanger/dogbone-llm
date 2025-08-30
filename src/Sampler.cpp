#include "Sampler.h"

#include <unordered_map>

TokenSampler Sampler;

int TokenSampler::ArgMax(const float* row, int V) {
    int best = 0;
    float bv = row[0];
    for (int v = 1; v < V; v++) {
        if (row[v] > bv) { bv = row[v]; best = v; }
    }
    return best;
}

void TokenSampler::SoftmaxStable(const std::vector<float>& logits, std::vector<float>& probs) {
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

int TokenSampler::SampleFromProbs(const std::vector<float>& probs) {
    float r = (float)std::rand() / (float)RAND_MAX;
    float c = 0.0f;
    const int V = (int)probs.size();
    for (int i = 0; i < V; i++) {
        c += probs[(size_t)i];
        if (r <= c) return i;
    }
    return V - 1;
}

// Optional: comparator if you don't want a lambda
bool TokenCandidateGreater(const TokenCandidate& a, const TokenCandidate& b) {
    return a.prob > b.prob;
}

// Apply penalties to logits based on the context.
void TokenSampler::ApplyRepetitionPenalties(std::vector<float>& logits,
                                            const std::vector<int>& context_ids,
                                            float presence_penalty,
                                            float frequency_penalty) {
    if (presence_penalty <= 0.0f && frequency_penalty <= 0.0f) return;
    
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

void TokenSampler::ApplyTopK(std::vector<float>& logits, int k) {
    const int V = (int)logits.size();
    if (k <= 0 || k >= V) return;
    
    std::vector<float> buf = logits;
    std::nth_element(buf.begin(), buf.begin() + (size_t)(V - k), buf.end());
    float thresh = buf[(size_t)(V - k)];
    const float NEG_INF = -1e30f;
    for (int i = 0; i < V; i++) {
        if (logits[(size_t)i] < thresh) logits[(size_t)i] = NEG_INF;
    }
}

void TokenSampler::ApplyTopP(std::vector<float>& probs, float top_p) {
    if (top_p >= 1.0f) return;
    if (top_p <= 0.0f) top_p = 1e-6f;
    
    const int V = (int)probs.size();
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
        float inv = 1.0f / (float)V;
        for (int i = 0; i < V; i++) probs[(size_t)i] = inv;
    }
}

int TokenSampler::GetNextToken(const TransformerLauguageModel& model,
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
    TokenSampler::ApplyRepetitionPenalties(logits, context_ids, P.presence_penalty, P.frequency_penalty);
    
    // Temperature / Greedy
    if (P.temperature <= 0.0f) {
        // Greedy pick
        return TokenSampler::ArgMax(&logits[0], V);
    } else {
    // Scale logits by 1/temperature
    float invT = 1.0f / P.temperature;
    for (int i = 0; i < V; i++) logits[(size_t)i] *= invT;
    
    // Top-k
    if (P.top_k > 0) TokenSampler::ApplyTopK(logits, P.top_k);
    
    // Softmax
    std::vector<float> probs;
    TokenSampler::SoftmaxStable(logits, probs);
    
    // Top-p
    if (P.top_p < 1.0f) TokenSampler::ApplyTopP(probs, P.top_p);
    
    return TokenSampler::SampleFromProbs(probs);
    // Sample
    }
}

// Returns a ranked list of candidate tokens with probabilities.
// - max_candidates: cap the list length (e.g., 5, 20). If <= 0, returns all passing min_prob.
// - min_prob: filter out tiny tails (e.g., 1e-6f). Set to 0 to keep everything allowed by top-k/top-p.
// - renormalize: if true, re-normalizes probs across the returned set so they sum to 1.0.
std::vector<TokenCandidate> TokenSampler::GetProbableTokens(const TransformerLauguageModel& model, const std::vector<int>& context_ids, const SamplingParams& P,
                                                            int max_candidates, float min_prob, bool renormalize) {
    if (P.seed != 0u) std::srand(P.seed + (unsigned int)context_ids.size());
    
    // 1) Forward + grab last-step logits
    Tensor2D logits_t = model.Forward(context_ids);
    const int V = logits_t.C;
    const float* last = logits_t.Row(logits_t.R - 1);
    
    // 2) Copy logits into a mutable buffer
    std::vector<float> logits((size_t)V);
    for (int i = 0; i < V; ++i) logits[(size_t)i] = last[i];
    
    // 3) Repetition penalties (presence/frequency)
    Sampler.ApplyRepetitionPenalties(logits, context_ids, P.presence_penalty, P.frequency_penalty);
    
    // 4) Temperature / Greedy edge case
    if (P.temperature <= 0.0f) {
        const int idx = Sampler.ArgMax(&logits[0], V);
    
        std::vector<TokenCandidate> out(1);
        out[0].id = idx;
        out[0].logit = logits[(size_t)idx];
        out[0].prob = 1.0f;
        out[0].cumulative_prob = 1.0f;
        return out;
    }
    
    // 5) Scale logits by 1/T
    const float invT = 1.0f / P.temperature;
    for (int i = 0; i < V; ++i) logits[(size_t)i] *= invT;
    
    // 6) Top-k pruning in logit space (zero/neg-inf out disallowed)
    if (P.top_k > 0) Sampler.ApplyTopK(logits, P.top_k);
    
    // 7) Softmax -> probabilities
    std::vector<float> probs;
    Sampler.SoftmaxStable(logits, probs); // probs.size() == V
    
    // 8) Top-p / nucleus (this typically zeros low-prob tail and renormalizes)
    if (P.top_p < 1.0f) Sampler.ApplyTopP(probs, P.top_p);
    
    // 9) Collect candidates above min_prob (and not masked)
    std::vector<TokenCandidate> cands;
    cands.reserve((size_t)V);
    for (int i = 0; i < V; ++i) {
        const float p = probs[(size_t)i];
        if (p > 0.0f && p >= min_prob) {
            TokenCandidate c;
            c.id = i;
            c.prob = p;
            c.logit = logits[(size_t)i];
            c.cumulative_prob = 0.0f; // fill after sorting
            cands.push_back(c);
        }
    }
    
    if (cands.empty()) {
        // Fallback: return argmax even if it didn't pass thresholds
        const int idx = Sampler.ArgMax(&logits[0], V);
        std::vector<TokenCandidate> out(1);
        out[0].id = idx;
        out[0].logit = logits[(size_t)idx];
        out[0].prob = 1.0f;
        out[0].cumulative_prob = 1.0f;
        return out;
    }
    
    // 10) Sort by probability descending
    std::sort(cands.begin(), cands.end(), TokenCandidateGreater);
    
    // 11) Truncate to max_candidates if requested
    if (max_candidates > 0 && (int)cands.size() > max_candidates) {
        cands.resize((size_t)max_candidates);
    }
    
    // 12) Optional re-normalization over the returned set
    if (renormalize) {
        float sum = 0.0f;
        for (size_t i = 0; i < cands.size(); ++i) sum += cands[i].prob;
        if (sum > 0.0f) {
            for (size_t i = 0; i < cands.size(); ++i) cands[i].prob /= sum;
        }
    }
    
    // 13) Fill cumulative probabilities in sorted order
    float running = 0.0f;
    for (size_t i = 0; i < cands.size(); ++i) {
        running += cands[i].prob;
        cands[i].cumulative_prob = running;
    }
    
    return cands;
}

