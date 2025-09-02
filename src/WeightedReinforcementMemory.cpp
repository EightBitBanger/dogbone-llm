#include "WeightedReinforcementMemory.h"
#include <fstream>
#include <sstream>

WeightedReinforcementMemory::WeightedReinforcementMemory(float defaultBoost)
    : mDefaultBoost(defaultBoost) {}

void WeightedReinforcementMemory::addMemorySequence(const std::vector<TokenCandidate>& sequence, float weight, unsigned int lifetime) {
    // Merge by exact ID sequence if already present
    for (std::size_t i = 0; i < mSeqs.size(); ++i) {
        if (!mExpired[i] && sequencesEqualById(mSeqs[i], sequence)) {
            if (!mPermanent[i]) {
                mLifetime[i] = lifetime;
                mPermanent[i] = (lifetime == 0u);
            }
            if (weight > mWeights[i]) mWeights[i] = weight;
            return;
        }
    }
    mSeqs.push_back(sequence);
    mWeights.push_back(weight);
    mLifetime.push_back(lifetime);
    mPermanent.push_back(lifetime == 0u);
    mExpired.push_back(false);
}

void WeightedReinforcementMemory::decay() {
    for (std::size_t i = 0; i < mSeqs.size(); ++i) {
        if (mExpired[i]) continue;
        if (mPermanent[i]) continue;
        if (mLifetime[i] > 0u) {
            --mLifetime[i];
            if (mLifetime[i] == 0u) {
                mExpired[i] = true;
            }
        }
    }
    cleanup();
}

void WeightedReinforcementMemory::reinforceProbability(std::vector<TokenCandidate>& candidates, float boost) {
    const float b = (boost > 0.0f) ? boost : mDefaultBoost;
    if (candidates.empty() || mSeqs.empty()) return;

    for (std::size_t c = 0; c < candidates.size(); ++c) {
        float seqWeightSum = 0.0f;
        const int id = candidates[c].id;

        // Sum weights of all active sequences containing this id (once per sequence)
        for (std::size_t i = 0; i < mSeqs.size(); ++i) {
            if (mExpired[i]) continue;
            const std::vector<TokenCandidate>& seq = mSeqs[i];
            bool found = false;
            for (std::size_t j = 0; j < seq.size(); ++j) {
                if (seq[j].id == id) { found = true; break; }
            }
            if (found) seqWeightSum += mWeights[i];
        }

        if (seqWeightSum > 0.0f) {
            candidates[c].prob += b * seqWeightSum;
        }
    }

    normalize(candidates);
}

void WeightedReinforcementMemory::reinforceLifetime(const std::vector<int>& tokenIds, unsigned int bonus) {
    if (tokenIds.empty() || bonus == 0u) return;

    for (std::size_t i = 0; i < mSeqs.size(); ++i) {
        if (mExpired[i]) continue;
        if (mPermanent[i]) continue;

        const std::vector<TokenCandidate>& seq = mSeqs[i];
        bool hit = false;
        for (std::size_t t = 0; t < tokenIds.size() && !hit; ++t) {
            const int id = tokenIds[t];
            for (std::size_t j = 0; j < seq.size(); ++j) {
                if (seq[j].id == id) { hit = true; break; }
            }
        }
        if (hit) {
            mLifetime[i] += bonus;
        }
    }
}

bool WeightedReinforcementMemory::save(const std::string& filename) const {
    std::ofstream out(filename.c_str());
    if (!out.is_open()) return false;

    // Header and count
    out << "WRMSEQ1" << "\n";
    // Only count active sequences
    std::size_t activeCount = 0;
    for (std::size_t i = 0; i < mSeqs.size(); ++i) if (!mExpired[i]) ++activeCount;
    out << activeCount << "\n";

    for (std::size_t i = 0; i < mSeqs.size(); ++i) {
        if (mExpired[i]) continue;
        // meta line: weight lifetime length
        out << mWeights[i] << " " << mLifetime[i] << " " << mSeqs[i].size() << "\n";
        // ids line
        for (std::size_t j = 0; j < mSeqs[i].size(); ++j) {
            out << mSeqs[i][j].id;
            if (j + 1 < mSeqs[i].size()) out << " ";
        }
        out << "\n";
    }
    return true;
}

bool WeightedReinforcementMemory::load(const std::string& filename) {
    std::ifstream in(filename.c_str());
    if (!in.is_open()) return false;

    std::string header;
    if (!std::getline(in, header)) return false;
    if (header != "WRMSEQ1") return false;

    std::string line;
    if (!std::getline(in, line)) return false;
    std::istringstream issCount(line);
    std::size_t count = 0;
    if (!(issCount >> count)) return false;

    clear();
    mSeqs.reserve(count);
    mWeights.reserve(count);
    mLifetime.reserve(count);
    mPermanent.reserve(count);
    mExpired.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        if (!std::getline(in, line)) return false;
        std::istringstream issMeta(line);
        float weight = 0.0f;
        unsigned int life = 0u;
        std::size_t length = 0u;
        if (!(issMeta >> weight >> life >> length)) return false;

        if (!std::getline(in, line)) return false;
        std::istringstream issIds(line);
        std::vector<TokenCandidate> seq;
        seq.resize(length);
        for (std::size_t j = 0; j < length; ++j) {
            int id = 0;
            if (!(issIds >> id)) return false;
            seq[j].id = id;
            seq[j].prob = 0.0f; // not used here
        }

        mSeqs.push_back(seq);
        mWeights.push_back(weight);
        mLifetime.push_back(life);
        mPermanent.push_back(life == 0u);
        mExpired.push_back(false);
    }

    return true;
}

void WeightedReinforcementMemory::clear() {
    mSeqs.clear();
    mWeights.clear();
    mLifetime.clear();
    mExpired.clear();
    mPermanent.clear();
}

void WeightedReinforcementMemory::cleanup() {
    std::size_t w = 0;
    for (std::size_t r = 0; r < mSeqs.size(); ++r) {
        if (!mExpired[r]) {
            if (w != r) {
                mSeqs[w]      = mSeqs[r];
                mWeights[w]   = mWeights[r];
                mLifetime[w]  = mLifetime[r];
                mExpired[w]   = mExpired[r];
                mPermanent[w] = mPermanent[r];
            }
            ++w;
        }
    }
    mSeqs.resize(w);
    mWeights.resize(w);
    mLifetime.resize(w);
    mExpired.resize(w);
    mPermanent.resize(w);
}

void WeightedReinforcementMemory::normalize(std::vector<TokenCandidate>& candidates) {
    float total = 0.0f;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        total += candidates[i].prob;
    }
    if (total <= 0.0f) return;
    const float inv = 1.0f / total;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        candidates[i].prob *= inv;
    }
}

bool WeightedReinforcementMemory::sequencesEqualById(const std::vector<TokenCandidate>& a, const std::vector<TokenCandidate>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].id != b[i].id) return false;
    }
    return true;
}
