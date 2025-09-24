#ifndef MAIN_H
#define MAIN_H

#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>

#include <vector>
#include <string>
#include <unordered_map>

#include <sstream>

#include <algorithm>
#include <limits>
#include <iomanip>
#include <chrono>
#include <iostream>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "Platform/Platform.h"
#include "GLContext.h"

#include "Utils/Strings.h"
#include "Utils/Timer.h"
#include "Utils/Print.h"

#include "BuildDataSet.h"
#include "Tokenizer.h"
#include "Modelio.h"
#include "Sampler.h"

#include "SemanticCoherence.h"
#include "ContextWindow.h"
#include "Transformer/Transformer.h"
#include "WeightedMemories.h"
#include "LanguageModelContextStream.h"

#include "Tests/test.h"

void TrainModelCPU(std::string& trainingFilename, std::string& modelFilename, LanguageModel& model, Tokenizer& vocab, Timer& time, 
                          int layerWidth, int headCount, int feedWidth, int layerDepth, int contextSize, 
                          float& learningRate, float learningRateMin, float learningDecayBegin, float learningRateDecay, 
                          float& avgLoss, float lossDropout, int datasetStride);

void TrainModelGPU(std::string& trainingFilename, std::string& modelFilename, LanguageModel& model, Tokenizer& vocab, Timer& time, 
                          int layerWidth, int headCount, int feedWidth, int layerDepth, int contextSize, 
                          float& learningRate, float learningRateMin, float learningDecayBegin, float learningRateDecay, 
                          float& avgLoss, float lossDropout, int datasetStride, ShaderTensor& gpu);


std::uint64_t CalculateModelParameterCount(int d_model, int n_layers, int feed, int vocab, int n_ctx, bool learned_pos, bool tie_embed);




// Build (inputs, targets) pairs for next-token training from raw text lines.

// Build fixed-length (inputs, targets) pairs for next-token training from raw text lines.
// Each line is tokenized with optional BOS/EOS and then split into blocks of length <= block_len.
// Overlap is controlled by `stride` (e.g., stride=32 with block_len=128).
void BuildNextTokenDataset(const Tokenizer& vocab, const std::vector<std::string>& corpus, std::vector<std::vector<int>>& inputs, std::vector<std::vector<int>>& targets,
                                  int block_len, int stride = 32);

#endif
