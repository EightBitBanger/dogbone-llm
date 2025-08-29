#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>

#include <vector>
#include <string>
#include <unordered_map>

#include <iostream>
#include <sstream>

#include <algorithm>
#include <limits>
#include <iomanip>

#include "tokenizer.h"
#include "Transformer/Transformer.h"
#include "sampler.h"

#include <thread>
#include <atomic>
#include <chrono>

#include <ctime>
#include <cfloat>


#include <windows.h>

template<typename T>
static inline T clampv(T v, T lo, T hi) { return (v < lo ? lo : (v > hi ? hi : v)); }


static void TrainModel(std::string& trainingFilename, std::string& modelFilename,
                       TransformerLauguageModel& model, Vocabulary& vocab,
                       int layerWidth, int headCount, int feedWidth, int layerDepth,
                       int contextSize, float learningRate, float learningRateMin, 
                       float learningRateDecay, float& avgLoss, float lossDropout);

NeuralNetwork trainer(0.001f);



/*
static std::string GetDate(void) {
    std::time_t t = std::time(NULL);
    std::tm local_tm;
#if defined(_WIN32)
    localtime_s(&local_tm, &t);
#else
    localtime_r(&t, &local_tm);
#endif
    char buf[32];
    // YYYYMMDD-HHMM (zero-padded)
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M", &local_tm);
    return std::string(buf);
}
*/

bool FileTextLoad(std::string& path, std::string& out) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    
    std::ostringstream buffer;
    buffer << in.rdbuf();
    const std::string bufferString = buffer.str();
    
    out.clear();
    out.reserve(bufferString.size());
    
    for (size_t i = 0; i < bufferString.size(); ++i) {
        char c = bufferString[i];
        if (c != '\n' && c != '\r' && c != ',' && c != '`') {
            out.push_back(c);
        }
    }
    return true;
}

bool FileExists(std::string filename) {
    std::ofstream fStream(filename, std::fstream::in | std::fstream::binary);
    if (!fStream.is_open()) {
        fStream.close();
        return false;
    }
    fStream.close();
    return true;
}

bool FileDelete(std::string filename) {
    if (remove( filename.c_str() ) == 0) 
        return true;
    return false;
}

bool is_sentence_end(const std::string& s) {
    if (s.empty()) return false;
    char c = s.back();
    return (c == '.' || c == '!' || c == '?');
}

bool is_special(const std::string& s) {
    return s.size() >= 2 && s.front() == '<' && s.back() == '>';
}

bool is_wordish(const std::string& s) {
    // Count as a "word" if it has at least one alnum
    for (unsigned char ch : s) {
        if (std::isalnum(ch)) return true;
    }
    return false;
}

struct TokenOutputPipeline {
    bool first = false;
    bool capitalize_next = true;
    bool saw_word = false;

    void reset() {
        first = false;
        capitalize_next = true;
        saw_word = false;   // NEW
    }

    static inline bool EndsSentence(const std::string& s) {
        if (s.empty()) return false;
        char c = s[(size_t)s.size() - 1];
        return (c == '.' || c == '!' || c == '?');
    }

    static inline void CapitalizeFirstLetter(std::string& s) {
        for (size_t i = 0; i < s.size(); ++i) {
            // skip leading quotes/paren etc. e.g., "(", "\"", ""
            if (std::isalpha(static_cast<unsigned char>(s[i]))) {
                s[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[i])));
                break;
            }
        }
    }

    void emit(const std::string& tok_in) {
        std::string tok = tok_in;
        
        auto is_tight_punct = [](const std::string& s)->bool {
            if (s.size() != 1) return false;
            switch (s[0]) {
                case '.': case '!': case '?':
                case ',': case ';': case ':':
                case ')': case ']':
                    return true;
                default: return false;
            }
        };
        auto is_wordish = [](const std::string& s)->bool {
            for (unsigned char ch : s) if (std::isalnum(ch)) return true;
            return false;
        };
        
        // If we haven't printed any word yet, silently drop tight punctuation
        if (!saw_word && is_tight_punct(tok)) {
            return; // don't flip capitalize_next, don't print anything
        }
        
        // Apply capitalization if needed
        if (capitalize_next && !tok.empty()) {
            CapitalizeFirstLetter(tok);
            capitalize_next = false;
        }
        
        if (!first) {
            std::cout << tok;
            first = true;
        } else {
            if (is_tight_punct(tok)) {
                std::cout << tok;
            } else {
                std::cout << " " << tok;
            }
        }
        std::cout.flush();
        
        if (is_wordish(tok)) saw_word = true;
        
        // After sentence enders, capitalize next word
        if (tok == "." || tok == "!" || tok == "?") {
            capitalize_next = true;
        }
    }
};

// Build (inputs, targets) pairs for next-token training from raw text lines.

// Build fixed-length (inputs, targets) pairs for next-token training from raw text lines.
// Each line is tokenized with optional BOS/EOS and then split into blocks of length <= block_len.
// Overlap is controlled by `stride` (e.g., stride=32 with block_len=128).
static void BuildNextTokenDataset(const Vocabulary& vocab,
                                  const std::vector<std::string>& corpus,
                                  std::vector<std::vector<int>>& inputs,
                                  std::vector<std::vector<int>>& targets,
                                  int block_len,
                                  int stride = 32) {
    inputs.clear();
    targets.clear();
    if (block_len < 2) return;

    for (const std::string& line : corpus) {
        // Tokenize line with BOS/EOS so the model learns sentence boundaries
        std::vector<int> ids = Encode(vocab, line, /*add_bos=*/true, /*add_eos=*/true);
        if (ids.size() < 2) continue;

        // Slide a window of `block_len` over ids with the given stride
        // Note: We create x = ids[t : t+block_len-1], y = ids[t+1 : t+block_len]
        // and right-pad with pad_id if needed (handled by Trainer/CE via pad_id).
        // Here we simply drop last incomplete windows smaller than 2 tokens;
        // exact padding to block_len is done by Trainer when it sees pad_id.
        const int n = (int)ids.size();
        for (int t = 0; t < n - 1; t += stride) {
            int end = t + block_len;
            if (end > n) end = n;
            int xlen = end - t - 1; // because y is shifted by +1
            if (xlen < 1) break;

            std::vector<int> x;
            std::vector<int> y;
            x.reserve((size_t)block_len);
            y.reserve((size_t)block_len);
            for (int i = 0; i < xlen; ++i) {
                x.push_back(ids[(size_t)(t + i)]);
                y.push_back(ids[(size_t)(t + i + 1)]);
            }
            inputs.push_back(std::move(x));
            targets.push_back(std::move(y));

            if (end == n) break;
        }
    }
}


// Return true to print (possibly modified) token, false to skip.
// You can mutate `out_token` to whatever you want printed.
static bool PreprocessToken(const Vocabulary& vocab, int token_id, std::string& out_token) {
    if (token_id == vocab.bos_id) return false; // don't print BOS
    if (token_id == vocab.eos_id) return false; // stop handled by caller

    if (token_id < 0 || token_id >= (int)vocab.id_to_word.size()) {
        out_token = "<UNK>";
        return true;
    }
    out_token = vocab.id_to_word[(size_t)token_id];

    // Example place to do filtering/transforms before printing:
    // if (out_token == "badword") { out_token = "[redacted]"; }
    return true;
}




// ================================
// Training state
// ================================

struct TrainState {
    uint32_t version;
    uint64_t epoch;
    float lr;
    TrainState() : version(1u), epoch(0ull), lr(0.0f) {}
};

static bool SaveTrainState(const std::string& modelPath, const TrainState& s) {
    std::string path = modelPath + ".state";
    std::ofstream out(path.c_str(), std::ios::binary);
    if (!out.is_open()) return false;
    out.write((const char*)&s.version, sizeof(uint32_t));
    out.write((const char*)&s.epoch,   sizeof(uint64_t));
    out.write((const char*)&s.lr,      sizeof(float));
    return out.good();
}

static bool LoadTrainState(const std::string& modelPath, TrainState& s) {
    std::string path = modelPath + ".state";
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in.is_open()) return false;
    in.read((char*)&s.version, sizeof(uint32_t));
    in.read((char*)&s.epoch,   sizeof(uint64_t));
    in.read((char*)&s.lr,      sizeof(float));
    return in.good();
}







//
// Key input capturing
#include <conio.h>

static bool KeyPressedNonBlocking() {
    return _kbhit() != 0;
}

static int ReadKeyNonBlocking() {
    return _getch();
}


