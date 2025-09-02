#ifndef WEIGHTED_REINFORCEMENT_MEMORY_H
#define WEIGHTED_REINFORCEMENT_MEMORY_H

#include <vector>
#include <cstddef>
#include <cstdint>
#include <string>

#include "Sampler.h"

// Manages memory as sequences of token IDs with a weight per sequence.
// Lifetime is per sequence (0 = permanent). Expired sequences are removed on cleanup.
class WeightedReinforcementMemory {
public:
    explicit WeightedReinforcementMemory(float defaultBoost = 0.2f);

    // Add or refresh a memory sequence. Merges if an identical sequence already exists (compares IDs only).
    void addMemorySequence(const std::vector<TokenCandidate>& sequence, float weight, unsigned int lifetime = 0);

    // Age non-permanent sequences and drop expired ones.
    void decay();

    // Probability reinforcement: boosts prob for candidates whose IDs appear in any active sequence.
    // 'boost' scales how strongly each sequence's weight contributes (if boost <= 0, uses the default).
    void reinforceProbability(std::vector<TokenCandidate>& candidates, float boost);

    // Lifetime reinforcement: extend lifetime of sequences that contain any of 'tokenIds' by 'bonus' ticks.
    void reinforceLifetime(const std::vector<int>& tokenIds, unsigned int bonus);

    // Persistence
    bool save(const std::string& filename = "memories.context") const;
    bool load(const std::string& filename = "memories.context");

    // Housekeeping
    void clear();
    std::size_t size() const { return mSeqs.size(); }

private:
    // Sequences and metadata
    std::vector< std::vector<TokenCandidate> > mSeqs;
    std::vector<float>         mWeights;    // per-sequence weight
    std::vector<unsigned int>  mLifetime;   // per-sequence lifetime (0 = permanent)
    std::vector<bool>          mExpired;    // true if expired
    std::vector<bool>          mPermanent;  // true if lifetime==0 at insert

    float                      mDefaultBoost;

    // Utils
    void cleanup();
    static void normalize(std::vector<TokenCandidate>& candidates);
    static bool sequencesEqualById(const std::vector<TokenCandidate>& a, const std::vector<TokenCandidate>& b);
};

#endif
