#include "WeightedMemories.h"
#include <iostream>
#include <iomanip>
#include <fstream>

#include <algorithm>
#include <cctype>

#include "WeightedMemories.h"
#include "ContextWindow.h"
#include <stdlib.h>
#include <algorithm>
#include <sstream>

WeightedReinforcementMemory::WeightedReinforcementMemory(unsigned int maxNGram,
                                                         unsigned int capacity,
                                                         float decay) : 
    mMaxNGram(maxNGram < 2 ? 2 : maxNGram),
    mCapacity(capacity > 0 ? capacity : 20000),
    mDecay(decay < 0.0f ? 0.0f : (decay > 1.0f ? 1.0f : decay)),
    mStepCounter(0u) {}

void WeightedReinforcementMemory::SetCapacity(unsigned int cap) {
    mCapacity = cap > 0 ? cap : 1;
    PruneIfNeeded_();
}

void WeightedReinforcementMemory::SetDecay(float d) {
    if (d < 0.0f) d = 0.0f;
    if (d > 1.0f) d = 1.0f;
    mDecay = d;
}

unsigned int WeightedReinforcementMemory::GetCapacity() const { return mCapacity; }
float        WeightedReinforcementMemory::GetDecay()     const { return mDecay; }
unsigned int WeightedReinforcementMemory::GetMaxNGram()  const { return mMaxNGram; }

void WeightedReinforcementMemory::Clear() {
    mWeights.clear();
    mLastSeen.clear();
    mStepCounter = 0u;
}

void WeightedReinforcementMemory::Tick(unsigned int steps) {
    if (steps == 0u) return;
    for (unsigned int i = 0u; i < steps; ++i) {
        DecayAll_();
        if (mStepCounter < std::numeric_limits<uint32_t>::max()) {
            mStepCounter += 1u;
        }
    }
}

// Observe the current tail of the stream and reinforce every n-gram
// ending at the last token, for n = 2..mMaxNGram.
// Uses ContextWindow::GetContext() / Size() from your code. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}
void WeightedReinforcementMemory::Observe(ContextWindow& context) {
    // Lightweight global decay each time we learn; keeps old patterns from dominating.
    DecayAll_();

    const std::vector<int>& toks = context.GetContext();
    const unsigned int T = (unsigned int)toks.size();
    if (T < 2u) {
        if (mStepCounter < std::numeric_limits<uint32_t>::max()) mStepCounter += 1u;
        return;
    }

    const unsigned int maxN = std::min(mMaxNGram, T);
    // Reinforce all suffix n-grams that end at the last token.
    for (unsigned int n = 2u; n <= maxN; ++n) {
        const size_t start = (size_t)(T - n);
        const std::string key = KeyFromSpan(toks, start, (size_t)n);

        // Increment weight with small constant; recent steps add slightly more.
        const float bump = 1.0f;
        std::unordered_map<std::string, float>::iterator it = mWeights.find(key);
        if (it == mWeights.end()) {
            mWeights[key] = bump;
        } else {
            it->second += bump;
        }
        mLastSeen[key] = mStepCounter;
    }

    if (mStepCounter < std::numeric_limits<uint32_t>::max()) mStepCounter += 1u;
    PruneIfNeeded_();
}

// External nudge: reinforce any sequence directly.
void WeightedReinforcementMemory::Reinforce(const std::vector<int>& seq, float amount) {
    if (seq.size() < 2 || amount <= 0.0f) return;
    const std::string key = KeyFrom(seq);
    std::unordered_map<std::string, float>::iterator it = mWeights.find(key);
    if (it == mWeights.end()) {
        mWeights[key] = amount;
    } else {
        it->second += amount;
    }
    mLastSeen[key] = mStepCounter;
    PruneIfNeeded_();
}

// Suggest likely next tokens given a prefix.
// Strategy: scan stored n-grams; for each that begins with the prefix,
// add its weight to the candidate "next token" (the token right after prefix).
std::vector<WRM_Suggestion>
WeightedReinforcementMemory::SuggestNext(const std::vector<int>& prefix,
                                         unsigned int topK) const {
    std::vector<WRM_Suggestion> out;
    if (prefix.size() == 0u) return out;

    // Aggregate scores per next_token
    std::unordered_map<int, float> scores;
    std::unordered_map<int, std::vector<int> > reps;

    // Build string form of prefix for fast starts-with checks.
    const std::string preKey = KeyFrom(prefix);
    const size_t preLen = preKey.size();

    // Iterate all learned sequences (simple, but OK at this scale).
    for (std::unordered_map<std::string, float>::const_iterator it = mWeights.begin();
         it != mWeights.end(); ++it) {
        const std::string& key = it->first;
        const float w = it->second;
        if (w <= 0.0f) continue;

        // Check if this n-gram starts with the given prefix
        // We require either exact match (no suggestion) or prefix + ','.
        if (key.size() <= preLen) continue;
        if (key.compare(0u, preLen, preKey) != 0) continue;
        if (key[preLen] != ',') continue;

        // Parse just the next token after the prefix (first number after preLen+1)
        // key = "<p0>,<p1>,...,<pL-1>,<next>[,rest...]"
        size_t pos = preLen + 1u;
        bool neg = false;
        long val = 0;
        // simple parse: read until ',' or end
        size_t i = pos;
        if (i < key.size() && key[i] == '-') { neg = true; ++i; }
        while (i < key.size() && key[i] >= '0' && key[i] <= '9') {
            val = val * 10 + (long)(key[i] - '0');
            ++i;
        }
        if (neg) val = -val;

        int nextTok = (int)val;
        std::unordered_map<int, float>::iterator sit = scores.find(nextTok);
        if (sit == scores.end()) {
            scores[nextTok] = w;
        } else {
            sit->second += w;
        }

        // Build one representative matched sequence for UX (prefix + next token)
        std::unordered_map<int, std::vector<int> >::iterator r = reps.find(nextTok);
        if (r == reps.end()) {
            std::vector<int> seq(prefix);
            seq.push_back(nextTok);
            reps[nextTok] = seq;
        }
    }

    // Collect and partial sort
    std::vector<std::pair<int, float> > pairs;
    pairs.reserve(scores.size());
    for (std::unordered_map<int, float>::const_iterator it = scores.begin();
         it != scores.end(); ++it) {
        pairs.push_back(std::make_pair(it->first, it->second));
    }
    // Simple selection of topK
    if (pairs.size() > 1u) {
        std::partial_sort(pairs.begin(),
                          pairs.begin() + (std::min((size_t)topK, pairs.size())),
                          pairs.end(),
                          // descending by score
                          [](const std::pair<int,float>& a, const std::pair<int,float>& b){
                              return a.second > b.second;
                          });
    }

    const size_t limit = std::min((size_t)topK, pairs.size());
    out.reserve(limit);
    for (size_t idx = 0; idx < limit; ++idx) {
        WRM_Suggestion s;
        s.next_token = pairs[idx].first;
        s.score = pairs[idx].second;
        std::unordered_map<int, std::vector<int> >::const_iterator r = reps.find(s.next_token);
        if (r != reps.end()) s.matched_sequence = r->second;
        out.push_back(s);
    }
    return out;
}

void WeightedReinforcementMemory::DecayAll_() {
    if (mDecay >= 1.0f) {
        if (mStepCounter < std::numeric_limits<uint32_t>::max()) mStepCounter += 1u;
        return;
    }
    for (std::unordered_map<std::string, float>::iterator it = mWeights.begin();
         it != mWeights.end(); ++it) {
        it->second *= mDecay;
    }
    if (mStepCounter < std::numeric_limits<uint32_t>::max()) mStepCounter += 1u;
}

void WeightedReinforcementMemory::PruneIfNeeded_() {
    // Soft prune: if we exceed capacity by a margin, drop lowest-weight items.
    const size_t sz = mWeights.size();
    if (sz <= (size_t)mCapacity) return;

    // Collect into a small vector and sort ascending by weight.
    std::vector<std::pair<std::string, float> > items;
    items.reserve(sz);
    for (std::unordered_map<std::string, float>::const_iterator it = mWeights.begin();
         it != mWeights.end(); ++it) {
        items.push_back(std::make_pair(it->first, it->second));
    }
    std::sort(items.begin(), items.end(),
              [](const std::pair<std::string,float>& a,
                 const std::pair<std::string,float>& b){
                  return a.second < b.second;
              });

    // Remove the weakest extra entries
    size_t over = sz - (size_t)mCapacity;
    for (size_t i = 0; i < over; ++i) {
        const std::string& k = items[i].first;
        mWeights.erase(k);
        mLastSeen.erase(k);
    }
}

std::string WeightedReinforcementMemory::KeyFrom(const std::vector<int>& seq) {
    std::ostringstream oss;
    for (size_t i = 0; i < seq.size(); ++i) {
        if (i) oss << ',';
        oss << seq[i];
    }
    return oss.str();
}

std::string WeightedReinforcementMemory::KeyFromSpan(const std::vector<int>& tokens,
                                                     size_t start, size_t len) {
    std::ostringstream oss;
    const size_t end = start + len;
    for (size_t i = start; i < end; ++i) {
        if (i != start) oss << ',';
        oss << tokens[i];
    }
    return oss.str();
}

void WeightedReinforcementMemory::DumpToConsole(unsigned int maxItems) const {
    // Collect items into a sortable vector
    std::vector< std::pair<std::string, float> > items;
    items.reserve(mWeights.size());
    for (std::unordered_map<std::string, float>::const_iterator it = mWeights.begin();
         it != mWeights.end(); ++it) {
        items.push_back(std::make_pair(it->first, it->second));
    }

    // Sort by weight descending
    std::sort(items.begin(), items.end(),
              [](const std::pair<std::string,float>& a,
                 const std::pair<std::string,float>& b){
                  return a.second > b.second;
              });

    // Header
    std::cout << "=== WeightedReinforcementMemory Dump ===\n";
    std::cout << "maxNGram=" << mMaxNGram
              << "  capacity=" << mCapacity
              << "  size=" << (unsigned int)mWeights.size()
              << "  decay=" << std::fixed << std::setprecision(6) << mDecay
              << "  step=" << mStepCounter
              << "\n";

    // Rows
    const unsigned int limit = (unsigned int)std::min((size_t)maxItems, items.size());
    for (unsigned int i = 0u; i < limit; ++i) {
        const std::string& key = items[i].first;
        const float weight = items[i].second;

        // lookup last-seen step (0 if missing)
        uint32_t last = 0u;
        std::unordered_map<std::string, uint32_t>::const_iterator ls = mLastSeen.find(key);
        if (ls != mLastSeen.end()) last = ls->second;

        // key is "t0,t1,...,tn" (token ids). Print as-is to keep it fast/simple.
        std::cout << std::setw(5) << (i + 1) << "  "
                  << "w=" << std::setw(10) << std::setprecision(6) << std::fixed << weight
                  << "  last=" << std::setw(10) << last
                  << "  seq=[" << key << "]\n";
    }

    if (items.size() > limit) {
        std::cout << "... (" << (items.size() - limit) << " more)\n";
    }
    std::cout.flush();
}
