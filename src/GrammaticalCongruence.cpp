#include "GrammaticalCongruence.h"
#include <cctype>
#include <cmath>
#include <limits>

static const std::size_t kContextLookback = 16;

GrammaticalCongruence::GrammaticalCongruence(const std::vector<std::string>& id_to_word)
: mIdToWord(id_to_word)
{
    // Build membership sets
    for (std::size_t i = 0; i < mIdToWord.size(); ++i) {
        const std::string& w = mIdToWord[i];
        mVocabExact.insert(w);
        mVocabLower.insert(LowerASCII(w));
    }

    // Function words / auxiliaries / preps / specials
    mLex["a"] = G_DET; mLex["an"] = G_DET; mLex["the"] = G_DET;

    mLex["i"] = G_PRON_SUBJ;
    mLex["you"] = G_PRON_SUBJ | G_PRON_OBJ;
    mLex["he"] = G_PRON_SUBJ; mLex["she"] = G_PRON_SUBJ; mLex["it"] = G_PRON_SUBJ | G_PRON_OBJ;
    mLex["we"] = G_PRON_SUBJ; mLex["they"] = G_PRON_SUBJ;
    mLex["me"] = G_PRON_OBJ; mLex["him"] = G_PRON_OBJ; mLex["her"] = G_PRON_OBJ;
    mLex["us"] = G_PRON_OBJ; mLex["them"] = G_PRON_OBJ;

    mLex["can"] = G_MODAL; mLex["will"] = G_MODAL; mLex["should"] = G_MODAL;
    mLex["would"] = G_MODAL; mLex["could"] = G_MODAL;
    mLex["may"] = G_MODAL; mLex["might"] = G_MODAL;

    const char* preps[] = {
        "of","in","on","for","with","at","from","by","about","as",
        "into","like","through","after","over","between","without",
        "before","under","around"
    };
    for (std::size_t i = 0; i < sizeof(preps)/sizeof(preps[0]); ++i) {
        mLex[preps[i]] = G_PREP;
    }

    mLex["to"] = G_TO;

    mLex["am"] = G_VERB_BASE; mLex["are"] = G_VERB_BASE; mLex["is"] = G_VERB_3SG;
    mLex["do"] = G_VERB_BASE; mLex["does"] = G_VERB_3SG; mLex["did"] = G_VERB_PAST;
    mLex["have"] = G_VERB_BASE; mLex["has"] = G_VERB_3SG; mLex["had"] = G_VERB_PAST;
    mLex["was"] = G_VERB_PAST; mLex["were"] = G_VERB_PAST;
}

// --- Public API -------------------------------------------------------------

void GrammaticalCongruence::Apply(std::vector<TokenCandidate>& candidates,
                            const std::vector<int>& context_ids,
                            float lambda,
                            bool hard_select) const
{
    // Build a small lowercased context window (lower only if that form exists)
    std::vector<std::string> ctxLower;
    const std::size_t N = context_ids.size();
    const std::size_t start = (N > kContextLookback) ? (N - kContextLookback) : 0;
    for (std::size_t i = start; i < N; ++i) {
        int id = context_ids[i];
        if (id >= 0 && static_cast<std::size_t>(id) < mIdToWord.size()) {
            const std::string& w = mIdToWord[static_cast<std::size_t>(id)];
            ctxLower.push_back(ToSafeLowerIfInVocab(w));
        }
    }

    // Score each candidate and find best
    float bestTotal = -std::numeric_limits<float>::infinity();
    std::size_t bestIdx = 0;

    for (std::size_t i = 0; i < candidates.size(); ++i) {
        int id = candidates[i].id;
        std::string w = std::string();
        if (id >= 0 && static_cast<std::size_t>(id) < mIdToWord.size()) {
            w = mIdToWord[static_cast<std::size_t>(id)];
        }
        float g = ScoreGrammar(ctxLower, w);
        candidates[i].logit += lambda * g;

        if (candidates[i].logit > bestTotal) {
            bestTotal = candidates[i].logit;
            bestIdx = i;
        }
    }

    if (hard_select && !candidates.empty()) {
        const float BIG = 50.0f;
        const float PEN = 5.0f;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (i == bestIdx) {
                candidates[i].logit += BIG;
                candidates[i].prob = 1.0f;
                candidates[i].cumulative_prob = 1.0f;
            } else {
                candidates[i].logit -= PEN;
                candidates[i].prob = 0.0f;
                candidates[i].cumulative_prob = 0.0f;
            }
        }
        return;
    }

    // Soft re-rank -> recompute probs over this set
    SoftmaxOverLogits(candidates);
}

// --- Private helpers --------------------------------------------------------

std::string GrammaticalCongruence::LowerASCII(const std::string& s) {
    std::string out = s;
    for (std::size_t i = 0; i < out.size(); ++i) {
        unsigned char ch = static_cast<unsigned char>(out[i]);
        out[i] = static_cast<char>(std::tolower(ch));
    }
    return out;
}

std::string GrammaticalCongruence::ToSafeLowerIfInVocab(const std::string& w) const {
    // Only use lower-cased form if it exists in the vocabulary.
    std::string low = LowerASCII(w);
    if (mVocabLower.find(low) != mVocabLower.end()) return low;
    if (mVocabExact.find(w) != mVocabExact.end())   return w;
    return w; // Fallback: unknown word stays as-is
}

unsigned GrammaticalCongruence::GuessTag(const std::string& token) const {
    if (token.empty()) return G_UNKNOWN;

    std::string w = LowerASCII(token);
    std::unordered_map<std::string, unsigned>::const_iterator it = mLex.find(w);
    if (it != mLex.end()) return it->second;

    if (w.size() == 1 && std::ispunct(static_cast<unsigned char>(w[0])) != 0)
        return G_PUNCT;

    std::size_t n = w.size();
    if (n >= 3 && w.substr(n-3) == "ing") return G_VERB_GER | G_NOUN_SG;
    if (n >= 2 && w.substr(n-2) == "ed")  return G_VERB_PAST;
    if (n >= 1 && w[n-1] == 's')          return G_NOUN_PL | G_VERB_3SG;

    if (n >= 2) {
        std::string tail2 = w.substr(n-2);
        if (tail2 == "al" || tail2 == "ic" || tail2 == "ry") return G_ADJ;
    }
    return G_NOUN_SG; // noun-ish fallback
}

bool GrammaticalCongruence::StartsWithVowel(const std::string& w) {
    if (w.empty()) return false;
    char c = static_cast<char>(std::tolower(static_cast<unsigned char>(w[0])));
    return (c=='a'||c=='e'||c=='i'||c=='o'||c=='u');
}

bool GrammaticalCongruence::IsPastTenseHint(const std::vector<std::string>& ctxLower) {
    // Very light: scan last few tokens for common time hints
    int limit = static_cast<int>(ctxLower.size()) - 1;
    int min_i = (limit - 5 > -1) ? (limit - 5) : 0;
    for (int i = limit; i >= min_i; --i) {
        const std::string& w = ctxLower[static_cast<std::size_t>(i)];
        if (w == "yesterday" || w == "last" || w == "ago") return true;
    }
    return false;
}

int GrammaticalCongruence::RecentSubject(const std::vector<std::string>& ctxLower) {
    int limit = static_cast<int>(ctxLower.size()) - 1;
    int min_i = (limit - 7 > -1) ? (limit - 7) : 0;
    for (int i = limit; i >= min_i; --i) {
        const std::string& w = ctxLower[static_cast<std::size_t>(i)];
        if (w == "i") return 1;
        if (w == "you") return 2;
        if (w == "he" || w == "she" || w == "it") return 3;
        if (w == "we") return 4;
        if (w == "they") return 5;
    }
    return 0;
}

float GrammaticalCongruence::ScoreGrammar(const std::vector<std::string>& ctxLower,
                                    const std::string& candidate) const
{
    std::string candSafe = ToSafeLowerIfInVocab(candidate);
    unsigned ct = GuessTag(candSafe);
    float s = 0.0f;

    if (!ctxLower.empty()) {
        const std::string& prev = ctxLower.back();
        unsigned pt = GuessTag(prev);

        if (prev == "a"  && StartsWithVowel(candSafe))  s -= 2.0f;
        if (prev == "an" && !StartsWithVowel(candSafe)) s -= 2.0f;

        if ((pt & G_DET) != 0) {
            if ((ct & (G_ADJ | G_NOUN_SG | G_NOUN_PL)) == 0) s -= 1.5f;
        }
        if ((pt & G_PREP) != 0) {
            if ((ct & (G_DET | G_PRON_OBJ | G_ADJ | G_NOUN_SG | G_NOUN_PL)) == 0) s -= 1.0f;
        }
        if ((pt & G_MODAL) != 0) {
            if ((ct & G_VERB_BASE) == 0) s -= 2.0f;
        }
        if ((pt & G_PUNCT) != 0 && (GuessTag(candSafe) & G_PUNCT) != 0) {
            s -= 5.0f;
        }

        if (prev == "to") {
            bool infinitiveLikely = false;
            if (ctxLower.size() >= 2) {
                unsigned p2 = GuessTag(ctxLower[ctxLower.size()-2]);
                if ((p2 & G_MODAL) != 0 ||
                    (p2 & (G_VERB_BASE | G_VERB_3SG | G_VERB_PAST | G_VERB_GER)) != 0) {
                    infinitiveLikely = true;
                }
            }
            if (infinitiveLikely) {
                if ((ct & G_VERB_BASE) == 0) s -= 1.5f;
            } else {
                if ((ct & (G_DET | G_PRON_OBJ | G_ADJ | G_NOUN_SG | G_NOUN_PL)) == 0) s -= 1.0f;
            }
        }
    }

    int subj = RecentSubject(ctxLower);
    if (subj == 3) {
        if (candSafe == "are" || candSafe == "have" || candSafe == "do") s -= 1.5f;
        if ((ct & G_VERB_3SG) != 0) s += 0.5f;
    } else if (subj == 1 || subj == 2 || subj == 4 || subj == 5) {
        if (candSafe == "is" || candSafe == "has" || candSafe == "does") s -= 1.0f;
        if ((ct & G_VERB_BASE) != 0) s += 0.3f;
    }

    if (IsPastTenseHint(ctxLower)) {
        if ((ct & G_VERB_PAST) != 0) s += 0.4f;
        if ((ct & G_VERB_3SG)  != 0) s -= 0.2f;
    }

    return s;
}

void GrammaticalCongruence::SoftmaxOverLogits(std::vector<TokenCandidate>& cands) {
    if (cands.empty()) return;

    float maxLogit = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < cands.size(); ++i) {
        if (cands[i].logit > maxLogit) maxLogit = cands[i].logit;
    }

    double sumExp = 0.0;
    std::vector<double> expVals(cands.size(), 0.0);
    for (std::size_t i = 0; i < cands.size(); ++i) {
        double e = std::exp(static_cast<double>(cands[i].logit - maxLogit));
        expVals[i] = e;
        sumExp += e;
    }
    if (sumExp <= 0.0) sumExp = 1.0;

    double cum = 0.0;
    for (std::size_t i = 0; i < cands.size(); ++i) {
        double p = expVals[i] / sumExp;
        cands[i].prob = static_cast<float>(p);
        cum += p;
        cands[i].cumulative_prob = static_cast<float>(cum);
    }
}
