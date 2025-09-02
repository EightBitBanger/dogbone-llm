#include "main.h"
#include <iostream>

//#define RUN_UNIT_TESTS

int main() {
    std::string trainingFilename  = "training.txt";
    std::string modelFilename     = "dataset.model";
    
    // Setup window size
    SizePx displaySz = DisplayGetSize();
    WindowResizePx(displaySz.width * 0.7f, displaySz.height * 0.65f);
    
    // model hyperparameters (canonical names)
    const int   n_ctx           = 128;                 // max sequence length
    const int   d_model         = 128;                 // Model node width
    const int   n_layers        = 8;                   // Model depth
    const int   d_ff            = 4 * d_model;         // Should be 4 * d_model
    
    // optimizer   
    const float lr              = 0.001f;              // Initial learning rate
    const float lr_min          = 0.00001f;            // Minimum learning rate
    const float lr_decay        = 0.997f;              // Rate decay multiplier
    const float lr_decay_begin  = 2.0f;                // Target to begin decaying
    const float target_loss     = 0.3f;                // End training at the target loss
    
    // sampling
    const float temperature     = 0.5f;               
    const int   top_k           = 40;                  // Keep only the k most likely tokens and drop the rest
    const float top_p           = 1.0f;                // Keep the smallest set of tokens whose probabilities add up to p
    const int   context_size    = 128;                 // Number of tokens to 'remember'
    
    // attention
    const int   n_heads         = 16;
    const int   d_head          = d_model / n_heads;   // d_model must be divisible by n_heads
    
    // Sentence structuring
    const int tokenCountMax           = 1024;          // Absolute max number of tokens to return as the response
    const int wordsPerSentence        = 24;            // Max number of words per sentence
    const int sentenceCountMax        = 3;             // Max number of sentences or strings of tokens broken by a period
    
    const bool usingGraphicsAcceleration = true;      // Use the graphics card as a math accelerator/co-processor
    
    // Rough checks
    if (n_ctx < 1)                 {std::cerr << "n_ctx must be >= 1\n"; return 1;}
    if (n_heads < 1)               {std::cerr << "n_heads must be >= 1\n"; return 1;}
    if ((d_model % n_heads) != 0)  {std::cerr << "d_model must be divisible by n_heads\n"; return 1;}
    
    Timer timer;
    uint64_t epoch;
    float avgLoss;
    float learnRate = lr;
    float lossDrop = target_loss;
    
    std::srand(42);
    
    Tokenizer vocab;
    LauguageModel model;
    
    GLContext gl;
    
    // Set the GPU shader
    ShaderTensor shader;
    if (usingGraphicsAcceleration) {
        // OpenGL
        gl.init();
        
        // Initiate shaders
        trainer.UseGPU(&shader);
        trainer.EnableResidentWeights(true);
        trainer.BuildShaders();
    } else {
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        std::cout << "Logical processors " << (int)sysinfo.dwNumberOfProcessors << " CPU ready...\n";
    }
    
#ifdef RUN_UNIT_TESTS
    std::cout << "\n-------- Unit tests --------\n";
    
    RunAllUnitTests();
    
    //TestTensorShader(shader);
    std::cout << "\n\n";
#endif
    
    // Set the activation function
    Activation.SetActivationFunction( ActivationType::SWIGLU );
    
    /*
    // Check for the model file to load raw training text
    std::string trainingText;
    if (!LoadModelPackage(modelFilename, model, vocab, trainer, epoch, learnRate, avgLoss)) {
        if (!FileTextLoad(trainingFilename, trainingText) || trainingText.empty()) {
            std::cout << "Training data file not found or empty 'training.txt'..." << "\n";
        }
        std::cout << "Model package not found..." << "\n";
        
        if (usingGraphicsAcceleration) {
            TrainModelGPU(trainingFilename, modelFilename, model, vocab, timer, 
                        d_model, n_heads, d_ff, n_layers, n_ctx, 
                        learnRate, lr_min, lr_decay_begin, lr_decay, avgLoss, lossDrop, shader);
        } else {
            TrainModelCPU(trainingFilename, modelFilename, model, vocab, timer, 
                        d_model, n_heads, d_ff, n_layers, n_ctx, 
                        learnRate, lr_min, lr_decay_begin, lr_decay, avgLoss, lossDrop);
        }
        
    } else {
        
        std::cout << "Model package loaded" << "\n";
    }
    */
    
    // Setup sentence structuring
    SentenceStructure sentenceStruct;
    sentenceStruct.sentenceCountMax = sentenceCountMax;
    sentenceStruct.wordsPerSentence = wordsPerSentence;
    
    // Fire up the sampler
    SamplingParams sampler;
    sampler.temperature       = temperature;
    sampler.top_k             = std::min((int)vocab.Size(), top_k);
    sampler.top_p             = top_p;
    sampler.presence_penalty  = 0.7f;   // 0.5 - 1.0  if you see loops/repeats
    sampler.frequency_penalty = 0.4f;   // 0.5 - 1.0  for stronger anti-repetition
    sampler.seed              = 42;     // Seed for random runs
    
    ContextWindow context(context_size);
    
    std::cout << "\n";
    while (true) {
        std::cout << "> ";
        std::string keyboard_string;
        
        if (!std::getline(std::cin, keyboard_string)) break;
        
        if (keyboard_string.empty()) 
            continue;
        
        // Check system function calls
        if (keyboard_string[0] == '/') {
            keyboard_string.erase(keyboard_string.begin());
            
            // ================
            // Purge the model
            if (keyboard_string == "clear") {
                std::cout << "Reloading the model and context...\n";
                vocab = Tokenizer();
                model = LauguageModel();
                model = LauguageModel();
                continue;
            }
            
            // ================
            // Kick off a learning cycle
            if (keyboard_string == "train" || keyboard_string == "learn") {
                if (usingGraphicsAcceleration) {
                    TrainModelGPU(trainingFilename, modelFilename, model, vocab, timer, 
                                d_model, n_heads, d_ff, n_layers, n_ctx, 
                                learnRate, lr_min, lr_decay_begin, lr_decay, avgLoss, lossDrop, shader);
                } else {
                    TrainModelCPU(trainingFilename, modelFilename, model, vocab, timer, 
                                d_model, n_heads, d_ff, n_layers, n_ctx, 
                                learnRate, lr_min, lr_decay_begin, lr_decay, avgLoss, lossDrop);
                }
                
                continue;
            }
            
            // ================
            // Set/get the learning rate
            if (keyboard_string.rfind("rate", 0) == 0) {
                std::istringstream iss(keyboard_string);
                std::string cmd;
                iss >> cmd;
                if (iss >> learnRate) {
                    std::cout << "Learning rate  " << std::setprecision(4) << learnRate;
                    uint64_t ep = 0; float oldLR = 0.0f;
                    if (FileExists(modelFilename)) 
                        UpdateModelLROnDisk(modelFilename, learnRate, &ep, &oldLR);
                } else {
                    std::cout << "Learning rate  " << std::setprecision(4) << learnRate;
                }
                
                std::cout << "\n\n";
                continue;
            }
            
            // ================
            // Set/get loss drop out
            if (keyboard_string.rfind("loss", 0) == 0) {
                std::istringstream iss(keyboard_string);
                std::string cmd;
                iss >> cmd;
                iss >> lossDrop;
                std::cout << "Target loss " << std::setprecision(4) << lossDrop;
                std::cout << "\n\n";
                continue;
            }
            
            // ================
            // Get model details
            if (keyboard_string.rfind("info", 0) == 0) {
                std::cout << "---- Model details ----\n\n";
                std::cout << "Context size       " << n_ctx << "\n";
                std::cout << "Layer width        " << d_model << "\n";
                std::cout << "Number of layers   " << n_layers << "\n\n";
                
                std::cout << "Feed size          " << d_ff << "\n";
                std::cout << "attention heads    " << n_heads << "\n\n";
                
                std::cout << "Context window     " << context_size << "\n";
                std::cout << "Vocabulary         " << vocab.Size() << "\n";
                
                std::cout << "\n";
                
                continue;
            }
        }
        
        // Check if the model is zeroed
        if (model.d_model == 0) {
            std::cout << "Model not loaded...\n\n";
            continue;
        }
        
        // Encode user prompt
        ContextWindow userContextWindow(context_size);
        std::vector<Token>& userContext = userContextWindow.GetContext();
        userContext = Encode(vocab, keyboard_string);
        for (unsigned int id=0; id < userContext.size(); id++) 
            context.Add( userContext[id] );
        
        // Reset the context and counters
        userContextWindow.Clear();
        
        sentenceStruct.sentenceCounter = 0;
        sentenceStruct.wordsCounter = 0;
        
        // Sample until a word token is found to kick off the stream
        bool found=false;
        for (unsigned int retry = 0; retry < 4; retry++) {
            // Prime the context with a beginning of sentence token
            context.Add(vocab.token.bos_id);
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
        }
        
        // Check beginning token
        if (!found) {
            std::cout << "Error: stream did not produce tokens\n\n";
        }
        
        // Kick off a response stream
        for (unsigned int tokenCount=0; tokenCount < tokenCountMax; tokenCount++) {
            
            // Process the token stream
            if (!semantic.ProcessTokenStream(model, vocab, sampler, context, userContextWindow, sentenceStruct)) 
                break;
            
            // Terminate the token steam early
            if (KeyPressedNonBlocking()) 
                break;
        }
        
        printLn();
        std::cout << "\n\n";
    }
    return 0;
}










static void TrainModelGPU(std::string& trainingFilename, std::string& modelFilename,
                          LauguageModel& model, Tokenizer& vocab, Timer& time,
                          int layerWidth, int headCount, int feedWidth, int layerDepth, int contextSize, 
                          float& learningRate, float learningRateMin, float learningDecayBegin, float learningRateDecay, 
                          float& avgLoss, float lossDropout,
                          ShaderTensor& gpu) {
    // Load training text
    std::string corpusText;
    if (!FileTextLoad(trainingFilename, corpusText) || corpusText.empty()) {
        std::cerr << "Training text not found or empty: '" << trainingFilename << "'\n";
        return;
    }
    std::vector<std::string> corpus(1, corpusText);
    
    // Try to resume: model + vocab + optimizer
    uint64_t savedEpoch           = 0;
    float    restoredLearningRate = learningRate;
    
    int   currentEpoch            = 0;            // set below
    float currentLearningRate     = learningRate; // set below
    
    // Dataset stride (token step)
    const int datasetStride = 32;
    
    // Optimizer state / trainer
    trainer = NeuralNetwork(learningRate);
    trainer.UseGPU(&gpu);
    trainer.EnableResidentWeights(true);
    
    bool resumed = LoadModelPackage(modelFilename, model, vocab, trainer,
                                    savedEpoch, restoredLearningRate, avgLoss);
    
    if (resumed) {
        currentEpoch        = (int)savedEpoch;
        currentLearningRate = restoredLearningRate;
        std::cout << "Resuming training @ epoch " << (unsigned long long)currentEpoch << " (GPU)\n";
    } else {
        // Fresh start: build vocab and model
        vocab.FitToCorpus(vocab, corpus);
        model   = LauguageModel((int)vocab.Size(), layerWidth, headCount, feedWidth, layerDepth, contextSize);
        trainer = NeuralNetwork(learningRate);
        trainer.UseGPU(&gpu);
        trainer.EnableResidentWeights(true);
        currentEpoch        = 0;
        currentLearningRate = learningRate;
    }
    
    // Build dataset using existing vocab
    std::vector<std::vector<int>> inputSequences, targetTokens;
    BuildNextTokenDataset(vocab, corpus, inputSequences, targetTokens, contextSize, datasetStride);
    
    std::cout << "Training model on GPU...  (press any key to stop)\n\n";
    
    // Print last epoch if resumed
    if (resumed) {
        std::cout << "\r"
                << "Epoch  " << std::right << std::setw(1) << savedEpoch << "  "
                << "   Rate  " << std::setprecision(8) << std::setw(1) << currentLearningRate << "  "
                << "   Loss  " << std::fixed << std::setprecision(8) << std::setw(1) << avgLoss << "\n";
    }
    
    // Single-threaded loop
    // We still do micro-batch gradient accumulation to stabilize updates.
    bool keepTraining = true;
    while (keepTraining) {
        float runningLossSum     = 0.0f;
        float runningAvgLoss     = 0.0f;
        int   runningSampleCount = 0;
        
        trainer.opt.learning_rate = currentLearningRate;
        
        const size_t microBatchSize = 64; // tune: larger accumulation lowers update variance
        size_t       batchCount     = 0;
        
        GradientAccumulator acc;
        acc.InitLike(model);  // start a fresh accumulator
        
        for (size_t i = 0; i < inputSequences.size(); ++i) {
            // Training pass (forward may run on GPU; backward & grads on CPU)
            float sampleLoss = trainer.StepGPU(model, inputSequences[i], targetTokens[i],
                                            vocab.token.pad_id, &acc, false);
            
            runningLossSum     += sampleLoss;
            runningSampleCount += 1;
            runningAvgLoss      = runningLossSum / std::max(1, runningSampleCount);
            batchCount++;
            
            // Show progress
            float percentTowardDropout = 100.0f * (lossDropout / std::max(runningAvgLoss, 1e-12f));
            if (percentTowardDropout > 100.0f) percentTowardDropout = 100.0f;
            if (percentTowardDropout < 0.0f)   percentTowardDropout = 0.0f;
            
            float epochProgress = 100.0f * (float)runningSampleCount / (float)inputSequences.size();
            
            std::cout << "\r"
                    << "Epoch  " << std::right << std::setw(1) << currentEpoch << "  "
                    << "   Rate  " << std::setprecision(8) << std::setw(1) << currentLearningRate << "  "
                    << "   Loss  " << std::fixed << std::setprecision(8) << std::setw(1) << runningAvgLoss << "   "
                    << "   " << std::setprecision(3) << percentTowardDropout << " %       ";
            if (epochProgress < 100.0f) {
                std::cout << "   " << std::setprecision(3) << epochProgress << " %       ";
            } else {
                std::cout << "                   ";
            }
            std::cout << std::flush;
            
            // Apply once per micro-batch (average grads over samples)
            if (batchCount == microBatchSize || i + 1 == inputSequences.size()) {
                if (batchCount > 0) acc.Scale(1.0f / (float)batchCount);
                
                // Make sure all prior GPU writes are visible (conservative)
                gpu.sync();
                
                trainer.ApplyGradients(model, acc, 1.0f);
                
                // Reset accumulator for next micro-batch
                acc = GradientAccumulator();
                acc.InitLike(model);
                batchCount = 0;
            }
            
            //SaveModelPackage(modelFilename, model, vocab, trainer,
            //                (uint64_t)currentEpoch, currentLearningRate, runningAvgLoss);
            
            // Allow user to abort
            if (KeyPressedNonBlocking()) {
                keepTraining = false;
                std::cout << "\n\n";
                break;
            }
            
            // Early stop when reaching target loss
            if (runningAvgLoss < lossDropout) {
                keepTraining = false;
                break;
            }
        }
        
        currentEpoch++;
        std::cout << "\n";
        
        // LR decay (same logic as CPU path)
        float epochAverage = (runningSampleCount > 0) ? (runningLossSum / (float)runningSampleCount) : 0.0f;
        if (epochAverage < learningDecayBegin) {
            float next = learningRateDecay * currentLearningRate;
            if (next < learningRateMin) next = learningRateMin;
            currentLearningRate = next;
        }
    }
}




static void TrainModelCPU(std::string& trainingFilename, std::string& modelFilename,
                          LauguageModel& model, Tokenizer& vocab, Timer& time,
                          int layerWidth, int headCount, int feedWidth, int layerDepth, int contextSize, 
                          float& learningRate, float learningRateMin, float learningDecayBegin, float learningRateDecay, 
                          float& avgLoss, float lossDropout) {
    // Load training text
    std::string corpusText;
    if (!FileTextLoad(trainingFilename, corpusText) || corpusText.empty()) {
        std::cerr << "Training text not found or empty: '" << trainingFilename << "'\n";
        return;
    }
    std::vector<std::string> corpus(1, corpusText);

    // Try to resume: model + vocab + optimizer
    uint64_t savedEpoch              = 0;
    float    restoredLearningRate    = learningRate;

    int   currentEpoch               = 0;            // set below
    float currentLearningRate        = learningRate; // set below

    // Dataset stride (token step)
    const int datasetStride = 32;

    // Optimizer state
    trainer = NeuralNetwork(learningRate);
    
    bool resumed = LoadModelPackage(modelFilename, model, vocab, trainer,
                                    savedEpoch, restoredLearningRate, avgLoss);

    if (resumed) {
        currentEpoch        = (int)savedEpoch;
        currentLearningRate = restoredLearningRate;
        std::cout << "Resuming training @ epoch " << (unsigned long long)currentEpoch << "\n";
    } else {
        // Fresh start: build vocab and model
        vocab.FitToCorpus(vocab, corpus);
        model   = LauguageModel((int)vocab.Size(), layerWidth, headCount, feedWidth, layerDepth, contextSize);
        trainer = NeuralNetwork(learningRate);
        currentEpoch        = 0;
        currentLearningRate = learningRate;
    }

    // Build dataset using existing vocab
    std::vector<std::vector<int> > inputSequences, targetTokens;
    BuildNextTokenDataset(vocab, corpus, inputSequences, targetTokens, contextSize, datasetStride);

    std::cout << "Training model...  (press any key to stop)\n\n";

    // Print last epoch if resumed
    if (resumed) {
        std::cout << "\r"
                  << "Epoch  " << std::right << std::setw(1) << savedEpoch << "  "
                  << "   Rate  " << std::setprecision(8) << std::setw(1) << currentLearningRate << "  "
                  << "   Loss  " << std::fixed << std::setprecision(8) << std::setw(1) << avgLoss << "\n";
    }
    
    // Determine logical processors
    int logicalProcs = 0;
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    logicalProcs = (int)sysinfo.dwNumberOfProcessors;
    if (logicalProcs <= 0) logicalProcs = 1;
    
    // Attempt to spread the threads across many cores
    auto pin_this_thread_to_cpu = [](unsigned int cpuIndex) {
        const unsigned int kBits = (unsigned int)(sizeof(DWORD_PTR) * 8u);
        DWORD_PTR mask = (cpuIndex >= kBits) ? 0 : ( (DWORD_PTR)1u << (cpuIndex % kBits) );
        if (mask != 0) {
            SetThreadAffinityMask(GetCurrentThread(), mask);
        }
    };
    
    bool keepTraining = true;
    while (keepTraining) {
        float runningLossSum     = 0.0f;
        float runningAvgLoss     = 0.0f;
        int   runningSampleCount = 0;
        
        trainer.opt.learning_rate = currentLearningRate;
        
        const size_t microBatchSize = logicalProcs;    // Multi-threaded micro-batch training
        const int threadsUsed = logicalProcs;          // Always spawn one worker per logical processor
        
        for (size_t startIdx = 0; startIdx < inputSequences.size(); startIdx += microBatchSize) {
            size_t endIdx     = std::min(startIdx + microBatchSize, inputSequences.size());
            size_t batchSize  = endIdx - startIdx;
            
            std::vector<GradientAccumulator> threadGrads((size_t)threadsUsed);
            std::vector<float> threadLoss((size_t)threadsUsed, 0.0f);
            std::vector<size_t> threadSampleCount((size_t)threadsUsed, 0);
            
            std::vector<std::thread> threadPool;
            threadPool.reserve((size_t)threadsUsed);
            
            // Launch workers threads per logical processor
            for (int t = 0; t < threadsUsed; ++t) {
                threadPool.emplace_back([&, t]() {
                    pin_this_thread_to_cpu((unsigned int)t);
                    
                    threadGrads[(size_t)t].InitLike(model);
                    NeuralNetwork localTrainer = trainer;
                    localTrainer.opt.learning_rate = trainer.opt.learning_rate;
                    
                    // Stride work assignment across the micro-batch
                    for (size_t j = startIdx + (size_t)t; j < endIdx; j += (size_t)threadsUsed) {
                        // Training pass
                        float sampleLoss = localTrainer.Step(model, inputSequences[j], targetTokens[j],
                                                        vocab.token.pad_id, &threadGrads[(size_t)t], false);
                        
                        threadLoss[(size_t)t]        += sampleLoss;
                        threadSampleCount[(size_t)t] += 1;
                    }
                });
            }
            
            // Join workers
            for (size_t i = 0; i < threadPool.size(); ++i) {
                threadPool[i].join();
            }
            
            // Merge gradients/loss
            GradientAccumulator accumulatedGradients; 
            accumulatedGradients.InitLike(model);
            
            float  batchLossSum      = 0.0f;
            size_t batchSampleCount  = 0;
            
            for (int t = 0; t < threadsUsed; ++t) {
                accumulatedGradients.Add(threadGrads[(size_t)t]);
                batchLossSum     += threadLoss[(size_t)t];
                batchSampleCount += threadSampleCount[(size_t)t];
            }
            
            // Average gradients over samples in this micro-batch
            if (batchSampleCount > 0) {
                accumulatedGradients.Scale(1.0f / (float)batchSampleCount);
            }
            
            // Single Adam update with accumulated grads
            trainer.ApplyGradients(model, accumulatedGradients, 1.0f);
            
            // Bookkeeping
            runningLossSum     += batchLossSum;
            runningSampleCount += (int)batchSampleCount;
            runningAvgLoss      = (runningSampleCount > 0) ? (runningLossSum / (float)runningSampleCount) : 0.0f;
            
            // Early stop when reaching target loss
            if (runningAvgLoss < lossDropout) {
                keepTraining = false;
                break;
            }
            
            // Progress toward dropout target
            float percentTowardDropout = 100.0f * (lossDropout / std::max(runningAvgLoss, 1e-12f));
            if (percentTowardDropout > 100.0f) percentTowardDropout = 100.0f;
            if (percentTowardDropout < 0.0f)   percentTowardDropout = 0.0f;
            
            // Progress to epoch
            float epochProgress = 100.0f * (float)runningSampleCount / (float)inputSequences.size();
            
            std::cout << "\r"
                      << "Epoch  " << std::right << std::setw(1) << currentEpoch << "  "
                      << "   Rate  " << std::setprecision(8) << std::setw(1) << currentLearningRate << "  "
                      << "   Loss  " << std::fixed << std::setprecision(8) << std::setw(1) << runningAvgLoss << "   "
                      << "   " << std::setprecision(3) << percentTowardDropout << " %       ";
            
            if (epochProgress < 100.0f) {
                std::cout << "   " << std::setprecision(3) << epochProgress << " %       ";
            } else {
                std::cout << "                   "; // Clear the line
            }
            
            std::cout << std::flush;
            
            // Save weights + vocab + optimizer + epoch/LR
            if (time.check()) 
                SaveModelPackage(modelFilename, model, vocab, trainer,
                                (uint64_t)currentEpoch, currentLearningRate, runningAvgLoss);
            
            // Allow user to abort
            if (KeyPressedNonBlocking()) {
                keepTraining = false;
                std::cout << "\n\n";
                return;
            }
            // LR decay (same logic)
            float epochAverage = (runningSampleCount > 0) ? (runningLossSum / (float)runningSampleCount) : 0.0f;
            if (epochAverage < learningDecayBegin) {
                float next = learningRateDecay * currentLearningRate;
                if (next < learningRateMin) next = learningRateMin;
                currentLearningRate = next;
            }
        }
        
        currentEpoch++;
        
        std::cout << "\n";
        
        
    }
}
