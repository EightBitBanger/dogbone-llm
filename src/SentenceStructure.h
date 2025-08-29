#ifndef SENTENCE_SHAPER_H
#define SENTENCE_SHAPER_H

#include <string>
#include <vector>
#include <unordered_set>
#include <cctype>
#include <algorithm>
#include <sstream>

class SentenceShaper {
public:
    struct Options {
        bool fix_punctuation_and_case;
        bool fix_articles;
        bool cleanup_determiners;
        bool remove_immediate_duplicates;
        bool subject_verb_agreement;
        bool gentle_adverb_placement;

        Options()
        : fix_punctuation_and_case(true)
        , fix_articles(true)
        , cleanup_determiners(true)
        , remove_immediate_duplicates(true)
        , subject_verb_agreement(true)
        , gentle_adverb_placement(true) {}
    };

    explicit SentenceShaper(const Options& options = Options())
    : opt_(options) {}

    void SetOptions(const Options& options) { opt_ = options; }
    const Options& GetOptions() const { return opt_; }

    std::string Shape(const std::string& input) const {
        // 1) tokenize (words + punctuation)
        std::vector<std::string> t = TokenizeWordsAndPunct(input);

        // 2) local cleanup
        if (opt_.fix_articles) FixArticles(t);
        if (opt_.cleanup_determiners) CleanupDeterminers(t);
        CollapseSingularPluralPairs(t);          // NEW: "cat cats" -> "cats"
        if (opt_.remove_immediate_duplicates) RemoveImmediateDuplicates(t);

        // 3) tiny grammatical nudges
        RemoveRedundantPronounBeforeBe(t);       // NEW: "... NP it is ..." -> "... is ..."
        if (opt_.subject_verb_agreement) SubjectVerbAgreementAll(t); // NEW: per-clause, all auxiliaries
        if (opt_.remove_immediate_duplicates) RemoveImmediateDuplicates(t); // catch "are are" after edits
        if (opt_.gentle_adverb_placement) GentleAdverbPlacement(t);

        // 4) punctuation + casing
        if (opt_.fix_punctuation_and_case) {
            NormalizePunctuation(t);
            SentenceCase(t);
        }

        return JoinTokens(t);
    }

private:
    Options opt_;

    // ---------- Utilities ----------
    static inline bool IsAlphaNum(char c) { return (std::isalnum((unsigned char)c) != 0); }
    static inline bool IsSpace(char c)    { return (std::isspace((unsigned char)c) != 0); }

    static inline bool IsLetters(const std::string& s) {
        for (size_t i = 0; i < s.size(); ++i) {
            if (std::isalpha((unsigned char)s[i]) == 0) return false;
        }
        return !s.empty();
    }

    static inline std::string ToLower(const std::string& s) {
        std::string r = s;
        for (size_t i = 0; i < r.size(); ++i)
            r[i] = (char)std::tolower((unsigned char)r[i]);
        return r;
    }

    static inline bool IEquals(const std::string& a, const std::string& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i)
            if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i])) return false;
        return true;
    }

    static inline bool EndsWith(const std::string& s, const std::string& suf) {
        if (s.size() < suf.size()) return false;
        size_t i = s.size(), j = suf.size();
        while (j > 0) {
            if (std::tolower((unsigned char)s[i-1]) != std::tolower((unsigned char)suf[j-1])) return false;
            --i; --j;
        }
        return true;
    }

    static inline bool IsPunctToken(const std::string& tok) {
        return (tok.size()==1 && (tok[0]=='.'||tok[0]==','||tok[0]=='!'||tok[0]=='?'||tok[0]==';'||tok[0]==':'||tok[0]=='('||tok[0]==')'));
    }

    static inline bool IsBoundaryWord(const std::string& lw) {
        // Treat coordinators as soft clause boundaries for subject detection
        return (lw=="and" || lw=="but" || lw=="or" || lw=="yet");
    }

    static inline bool StartsWithVowelSound(const std::string& lw) {
        static const std::unordered_set<std::string> exceptions_a = {
            "university","unicorn","user","euro","one","once","ouija","ubiquity"
        };
        if (lw.empty()) return false;
        if (exceptions_a.find(lw) != exceptions_a.end()) return false;
        char c = lw[0];
        return (c=='a'||c=='e'||c=='i'||c=='o'||c=='u');
    }

    // ---------- Tokenization ----------
    static std::vector<std::string> TokenizeWordsAndPunct(const std::string& s) {
        std::vector<std::string> out;
        std::string cur;
        for (size_t i = 0; i < s.size(); ++i) {
            char ch = s[i];
            bool wordChar = IsAlphaNum(ch) || ch=='\'' || ch=='-';
            if (wordChar) {
                cur.push_back(ch);
            } else {
                if (!cur.empty()) { out.push_back(cur); cur.clear(); }
                if (!IsSpace(ch)) out.push_back(std::string(1, ch));
            }
        }
        if (!cur.empty()) out.push_back(cur);
        return out;
    }

    static std::string JoinTokens(const std::vector<std::string>& toks) {
        std::ostringstream oss;
        for (size_t i = 0; i < toks.size(); ++i) {
            const std::string& t = toks[i];
            bool isPunct = (t.size()==1 && (t[0]=='.'||t[0]==','||t[0]=='!'||t[0]=='?'||t[0]==';'||t[0]==':'||t[0]==')'));
            bool isOpenParen = (t == "(");
            if (i > 0) {
                const std::string& prev = toks[i-1];
                bool prevIsOpen = (prev == "(");
                bool prevIsPunct = (prev.size()==1 && (prev[0]=='.'||prev[0]==','||prev[0]=='!'||prev[0]=='?'||prev[0]==';'||prev[0]==':'||prev[0]=='('));
                if (!isPunct && !isOpenParen && !prevIsOpen) oss << ' ';
                if (isOpenParen && !prevIsPunct) oss << ' ';
            }
            oss << t;
        }
        return oss.str();
    }

    // ---------- Coarse lexicon ----------
    static inline bool IsDeterminer(const std::string& lw) {
        return (lw=="a"||lw=="an"||lw=="the"||lw=="this"||lw=="that"||lw=="these"||lw=="those");
    }
    static inline bool IsPronounSingular(const std::string& lw) {
        return (lw=="i"||lw=="he"||lw=="she"||lw=="it"||lw=="this"||lw=="that");
    }
    static inline bool IsPronounPlural(const std::string& lw) {
        return (lw=="we"||lw=="you"||lw=="they"||lw=="these"||lw=="those");
    }
    static inline bool LooksPluralNoun(const std::string& lw) {
        static const std::unordered_set<std::string> irregular = {
            "men","women","children","people","mice","geese","teeth","feet","oxen","data","media"
        };
        if (irregular.find(lw) != irregular.end()) return true;
        if (EndsWith(lw, "ies")) return true;
        if (EndsWith(lw, "ses") || EndsWith(lw, "xes") || EndsWith(lw, "zes") || EndsWith(lw, "ches") || EndsWith(lw, "shes")) return true;
        if (EndsWith(lw, "s") && !EndsWith(lw, "ss")) return true;
        return false;
    }
    static inline bool LooksVerbBe(const std::string& lw) {
        return (lw=="am"||lw=="is"||lw=="are"||lw=="was"||lw=="were"||lw=="be"||lw=="been"||lw=="being");
    }
    static inline bool IsAuxHasHave(const std::string& lw) { return (lw=="has"||lw=="have"); }
    static inline bool IsAuxDo(const std::string& lw)      { return (lw=="do"||lw=="does"); }
    static inline bool IsAdverbLY(const std::string& lw)   { return lw.size()>2 && EndsWith(lw,"ly"); }

    // ---------- Pass A: punctuation + casing ----------
    static void NormalizePunctuation(std::vector<std::string>& t) {
        std::vector<std::string> r;
        for (size_t i = 0; i < t.size(); ++i) {
            const std::string& tok = t[i];
            bool isP = IsPunctToken(tok);
            if (isP && !r.empty()) {
                if (r.back() == tok) continue;          // collapse .. ,, !!
                if (tok == "," && r.back() == "(") continue;
            }
            r.push_back(tok);
        }
        t.swap(r);
        if (!t.empty()) {
            const std::string& last = t.back();
            if (!(last=="."||last=="!"||last=="?")) t.push_back(".");
        }
    }

    static void SentenceCase(std::vector<std::string>& t) {
        bool capNext = true;
        for (size_t i = 0; i < t.size(); ++i) {
            std::string& tok = t[i];
            if (tok=="."||tok=="!"||tok=="?") { capNext = true; continue; }
            if (capNext) {
                if (!tok.empty() && std::isalpha((unsigned char)tok[0]) != 0) {
                    tok[0] = (char)std::toupper((unsigned char)tok[0]);
                    capNext = false;
                }
            }
            if (IEquals(tok, "i")) tok = "I";
        }
    }

    // ---------- Pass B: local word fixes ----------
    static void FixArticles(std::vector<std::string>& t) {
        for (size_t i = 0; i + 1 < t.size(); ++i) {
            std::string lw = ToLower(t[i]);
            if (lw=="a" || lw=="an") {
                std::string nextLower = ToLower(t[i+1]);
                bool vowel = StartsWithVowelSound(nextLower);
                if (lw=="a" && vowel) t[i] = "an";
                if (lw=="an" && !vowel) t[i] = "a";
            }
        }
    }

    static void CleanupDeterminers(std::vector<std::string>& t) {
        std::vector<std::string> out; out.reserve(t.size());
        for (size_t i = 0; i < t.size(); ++i) {
            std::string lw = ToLower(t[i]);
            if (!out.empty() && IsDeterminer(lw) && IsDeterminer(ToLower(out.back()))) {
                out.back() = t[i]; // keep latest
            } else {
                out.push_back(t[i]);
            }
        }
        t.swap(out);

        std::vector<std::string> out2; out2.reserve(t.size());
        for (size_t i = 0; i < t.size(); ++i) {
            if (!out2.empty() && IEquals(t[i], out2.back()) && (IEquals(t[i],"of")||IEquals(t[i],"to"))) continue;
            out2.push_back(t[i]);
        }
        t.swap(out2);
    }

    static void RemoveImmediateDuplicates(std::vector<std::string>& t) {
        std::vector<std::string> out; out.reserve(t.size());
        for (size_t i = 0; i < t.size(); ++i) {
            if (!out.empty()) {
                bool isP = IsPunctToken(t[i]);
                if (!isP && IEquals(t[i], out.back())) continue;
            }
            out.push_back(t[i]);
        }
        t.swap(out);
    }

    // NEW: "cat cats" -> "cats" (basic s/es/ies)
    static void CollapseSingularPluralPairs(std::vector<std::string>& t) {
        if (t.size() < 2) return;
        std::vector<std::string> out; out.reserve(t.size());
        out.push_back(t[0]);
        for (size_t i = 1; i < t.size(); ++i) {
            std::string prev = ToLower(out.back());
            std::string curr = ToLower(t[i]);
            bool bothWords = IsLetters(prev) && IsLetters(curr);
            bool collapse = false;

            if (bothWords) {
                if (curr == prev + "s") collapse = true;
                else if (curr == prev + "es") collapse = true;
                else if (EndsWith(prev, "y")) {
                    std::string root = prev.substr(0, prev.size()-1);
                    if (curr == root + "ies") collapse = true;
                }
            }
            if (collapse) {
                out.back() = t[i]; // keep the plural form
            } else {
                out.push_back(t[i]);
            }
        }
        t.swap(out);
    }

    // NEW: Remove NP + pronoun + BE redundancy (e.g., "An apple ... it is ...")
    static void RemoveRedundantPronounBeforeBe(std::vector<std::string>& t) {
        if (t.size() < 3) return;
        for (size_t i = 1; i + 1 < t.size(); ++i) {
            std::string lw = ToLower(t[i]);
            if (!(lw=="it" || lw=="they")) continue;
            std::string next = ToLower(t[i+1]);
            if (!LooksVerbBe(next)) continue;

            // Do not remove if pronoun starts the sentence/clause.
            // Look back up to 6 tokens or until boundary/punct.
            bool sawNP = false;
            bool boundaryHit = false;
            size_t j = i;
            while (j > 0) {
                --j;
                if (IsPunctToken(t[j]) || IsBoundaryWord(ToLower(t[j]))) { boundaryHit = true; break; }
                std::string lj = ToLower(t[j]);
                if (!IsDeterminer(lj) && IsLetters(lj)) { sawNP = true; break; } // noun-ish/adj-ish word
            }
            if (sawNP && !boundaryHit) {
                // remove the redundant pronoun
                t.erase(t.begin() + (long)i);
                --i; // stay stable
            }
        }
    }

    // ---------- Pass C: grammatical nudges ----------
    // Determine plurality of subject by scanning left from pos until a boundary.
    static bool GuessSubjectPluralLeft(const std::vector<std::string>& t, size_t pos) {
        size_t i = (pos == 0 ? 0 : pos - 1);
        while (true) {
            std::string lw = ToLower(t[i]);
            if (IsPunctToken(t[i]) || IsBoundaryWord(lw)) break;
            if (IsPronounPlural(lw)) return true;
            if (IsPronounSingular(lw)) return false;
            if (LooksPluralNoun(lw)) return true;
            if (i == 0) break;
            --i;
        }
        // default: singular (safer)
        return false;
    }

    // Adjust each auxiliary independently within its clause.
    static void SubjectVerbAgreementAll(std::vector<std::string>& t) {
        if (t.empty()) return;
        for (size_t i = 0; i < t.size(); ++i) {
            std::string v = ToLower(t[i]);
            if (!(LooksVerbBe(v) || IsAuxHasHave(v) || IsAuxDo(v))) continue;

            bool subjPlural = GuessSubjectPluralLeft(t, i);
            // be (present)
            if (v=="is" && subjPlural) { t[i] = "are"; continue; }
            if (v=="are" && !subjPlural) { t[i] = "is"; continue; }

            // have
            if (v=="has" && subjPlural) { t[i] = "have"; continue; }
            if (v=="have" && !subjPlural) {
                bool firstPerson = false;
                for (size_t k = 0; k < i; ++k) if (IEquals(t[k], "I")) { firstPerson = true; break; }
                t[i] = firstPerson ? "have" : "has";
                continue;
            }

            // do
            if (v=="does" && subjPlural) { t[i] = "do"; continue; }
            if (v=="do" && !subjPlural) { t[i] = "does"; continue; }
        }
    }

    static void GentleAdverbPlacement(std::vector<std::string>& t) {
        for (size_t i = 0; i + 1 < t.size(); ++i) {
            std::string lw = ToLower(t[i]);
            if (!IsAdverbLY(lw)) continue;
            if (IsPunctToken(t[i+1])) continue;

            size_t jmax = t.size() - 1;
            if (i + 3 < jmax) jmax = i + 3;

            bool moved = false;
            for (size_t j = i + 1; j <= jmax; ++j) {
                std::string wj = ToLower(t[j]);
                if (LooksVerbBe(wj) || IsAuxHasHave(wj) || IsAuxDo(wj)) {
                    std::string adv = t[i];
                    t.erase(t.begin() + (long)i);
                    if (j > i) j--;
                    t.insert(t.begin() + (long)(j + 1), adv);
                    moved = true;
                    break;
                }
            }
            if (moved) i = 0;
        }
    }
};

#endif
