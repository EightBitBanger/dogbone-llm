#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <thread>

#include <vector>
#include <string>
#include <unordered_map>

#include <sstream>

#include <algorithm>
#include <limits>
#include <iomanip>

#include <thread>
#include <atomic>
#include <chrono>

#include <ctime>
#include <cfloat>

#include "Utils/Timer.h"
#include "Utils/Print.h"
#include "GLContext.h"

#include "Tokenizer.h"
#include "Modelio.h"
#include "Sampler.h"

#include "SemanticCoherence.h"
#include "ContextWindow.h"
#include "Transformer/Transformer.h"
#include "WeightedReinforcementMemory.h"

#include "Tests/test.h"

#include <windows.h>

static bool KeyPressedNonBlocking();
static int ReadKeyNonBlocking();



// ---- Two-model helpers ----
static std::string TokensToString(const Tokenizer& vocab,
                                  const std::vector<Token>& tokens,
                                  size_t startIndex) {
    std::string out;
    for (size_t i = startIndex; i < tokens.size(); ++i) {
        const std::string w = vocab[(size_t)tokens[i]];
        if (out.empty()) out += w;
        else {
            bool punct = (w.size() == 1) && (w[0] == '.' || w[0] == ',' || w[0] == '!' || w[0] == '?' || w[0] == ';' || w[0] == ':');
            if (!punct) out += " ";
            out += w;
        }
    }
    return out;
}

static std::string GenerateOnce(LanguageModel& model, Tokenizer& vocab, SamplingParams& sampler, ContextWindow& context, SentenceStructure& sentenceStruct,
                                int tokenCountMax, int context_size) {
    // Sample until a word token is found to kick off the stream.
    // Prevents starting with a period and immediately ending the response stream.
    bool found=false;
    while (!found) {
        std::vector<TokenCandidate> candidate = Sampler.GetProbableTokens(model, context.GetContext(), sampler, vocab.Size(), 0.0001f, true);
        // Check candidates list for an appropriate token
        for (unsigned int c=0; c < candidate.size(); c++) {
            Token token = candidate[c].id;
            std::string word = vocab[token];
            if (semantic.is_wordish(word)) {
                found = true;
                break;
            }
        }
        if (found) break;
        context.Add( vocab.word_to_id["."] ); // Prime the context
    }
    
    ContextWindow userCW(context_size);
    sentenceStruct.sentenceCounter = 0;
    sentenceStruct.wordsCounter    = 0;
    userCW.Clear();
    
    std::vector<Token>& ctx = context.GetContext();
    size_t startSize = ctx.size();
    
    for (int t = 0; t < tokenCountMax; ++t) {
        if (!semantic.ProcessTokenStream(model, vocab, sampler, context, userCW, sentenceStruct)) break;
        if (KeyPressedNonBlocking()) break;
    }
    std::string out = TokensToString(vocab, ctx, startSize);
    return out;
}





struct SizePx { int width, height; };

template<typename T>
static inline T clampv(T v, T lo, T hi) { return (v < lo ? lo : (v > hi ? hi : v)); }


static void TrainModelCPU(std::string& trainingFilename, std::string& modelFilename, LanguageModel& model, Tokenizer& vocab, Timer& time, 
                          int layerWidth, int headCount, int feedWidth, int layerDepth, int contextSize, 
                          float& learningRate, float learningRateMin, float learningDecayBegin, float learningRateDecay, 
                          float& avgLoss, float lossDropout);

static void TrainModelGPU(std::string& trainingFilename, std::string& modelFilename, LanguageModel& model, Tokenizer& vocab, Timer& time, 
                          int layerWidth, int headCount, int feedWidth, int layerDepth, int contextSize, 
                          float& learningRate, float learningRateMin, float learningDecayBegin, float learningRateDecay, 
                          float& avgLoss, float lossDropout, ShaderTensor& gpu);

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

void WindowResizePx(int width, int height) {
    HWND hwnd = GetConsoleWindow();
    if (!hwnd) return;
    RECT r;
    GetWindowRect(hwnd, &r);
    // keep current position, just change size
    MoveWindow(hwnd, r.left, r.top, width, height, TRUE);
}

SizePx DisplayGetSize() {
    return { GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN) };
}


std::uint64_t CalculateModelParameterCount(int d_model, int n_layers, int feed, int vocab, int n_ctx, bool learned_pos, bool tie_embed) {
    std::uint64_t E  = (std::uint64_t)vocab * (std::uint64_t)d_model;
    std::uint64_t P  = learned_pos ? (std::uint64_t)n_ctx * (std::uint64_t)d_model : 0ULL;
    std::uint64_t attn = 4ULL * d_model * d_model + 4ULL * d_model;          // Q,K,V,O + biases
    std::uint64_t mlp  = 2ULL * d_model * feed + (std::uint64_t)feed + d_model;
    std::uint64_t ln   = 4ULL * d_model;                                      // two LayerNorms (γ,β)
    std::uint64_t per_layer = attn + mlp + ln;
    std::uint64_t core = E + P + (std::uint64_t)n_layers * per_layer;
    std::uint64_t head = tie_embed ? (std::uint64_t)vocab
                                   : (std::uint64_t)vocab * d_model + (std::uint64_t)vocab;
    return core + head;
}


std::vector<std::string> StringExplode(const std::string& value, const char character) {
	std::vector<std::string> result;
    std::istringstream iss(value);
    
    for (std::string token; std::getline(iss, token, character); ) {
        
        if (std::move(token) == "") 
            continue;
        
        result.push_back(std::move(token));
    }
    return result;
}


// Build (inputs, targets) pairs for next-token training from raw text lines.

// Build fixed-length (inputs, targets) pairs for next-token training from raw text lines.
// Each line is tokenized with optional BOS/EOS and then split into blocks of length <= block_len.
// Overlap is controlled by `stride` (e.g., stride=32 with block_len=128).
static void BuildNextTokenDataset(const Tokenizer& vocab,
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
        std::vector<int> ids = Encode(vocab, line);
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


