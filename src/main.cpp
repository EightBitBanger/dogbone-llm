#include "main.h"
#include <iostream>

#define RUN_UNIT_TESTS

int main() {
    std::string trainingFilename  = "ts.txt";
    std::string modelFilename     = "";
    
    // Setup window size
    SizePx displaySz = DisplayGetSize();
    WindowResizePx(displaySz.width * 0.7f, displaySz.height * 0.65f);
    
    // Model
    const int   n_ctx             = 64;                  // Max input sequence length
    const int   d_model           = 64;                  // Model node width
    const int   n_layers          = 24;                  // Model depth
    const int   d_ff              = 4 * d_model;         // Should be 4 * d_model
    
    // Optimizer
    const float lr                = 0.001f;              // Initial learning rate
    const float lr_min            = 0.0001f;             // Minimum learning rate
    const float lr_decay          = 0.98f;               // Rate decay multiplier
    const float lr_decay_begin    = 5.0f;                // Target to begin decaying
    const float target_loss       = 3.0f;                // End training at the target loss
    
    // Sampling
    const float temperature       = 0.3f;                // Scale the probability distribution. Higher = random
    const int   top_k             = 200;                 // Keep only the k most likely tokens and drop the rest
    const float top_p             = 0.8f;                // Keep the smallest set of tokens whose probabilities add up to p
    const int   context_size      = n_ctx;               // Number of tokens to 'remember'
    // Penalties
    const float presencePenalty   = 0.2f;                // 0.5 - 1.0  Degrade tokens that have been seen already
    const float frequencyPenalty  = 0.2f;                // 0.5 - 1.0  Degrade tokens by how many times they repeat
    
    // Attention
    const int   n_heads           = 2;
    const int   d_head            = d_model / n_heads;   // d_model must be divisible by n_heads
    
    // Sentence structuring
    const int tokenCountMax       = 1024;                // Absolute max number of tokens to return as the response
    const int wordsPerSentenceMax = 24;                  // Maximum number of words per sentence
    const int wordsPerSentenceMin = 3;                   // Minimum number of words per sentence
    const int sentenceCountMax    = 1;                   // Max number of sentences or strings of tokens broken by a period
    
    const bool usingGraphicsAcceleration = false;        // Use the graphics card as a math accelerator/co-processor
    
    // Rough checks
    if (n_ctx < 1)                 {std::cerr << "n_ctx must be >= 1\n"; return 1;}
    if (n_heads < 1)               {std::cerr << "n_heads must be >= 1\n"; return 1;}
    if ((d_model % n_heads) != 0)  {std::cerr << "d_model must be divisible by n_heads\n"; return 1;}
    
    Timer timer;
    uint64_t epoch = 0;
    float avgLoss;
    float learnRate = lr;
    float lossDrop = target_loss;
    
    std::srand(42);
    
    Tokenizer      vocab;
    LanguageModel  model;
    
    Tokenizer      vocabB;
    LanguageModel  modelB;
    
    bool duoMode = false;            // toggle duel model back-and-forth mode
    
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
    RunTests();
    std::cout << "\n\n";
#endif
    
    // Set the activation function
    Activation.SetActivationFunction( ActivationType::SWIGLU );
    
    // Setup sentence structuring
    SentenceStructure sentenceStruct;
    sentenceStruct.sentenceCountMax = sentenceCountMax;
    sentenceStruct.wordsPerSentenceMax = wordsPerSentenceMax;
    sentenceStruct.wordsPerSentenceMin = wordsPerSentenceMin;
    
    // Fire up the sampler
    SamplingParams sampler;
    sampler.temperature       = temperature;
    sampler.top_k             = std::min((int)vocab.Size(), top_k);
    sampler.top_p             = top_p;
    sampler.presence_penalty  = presencePenalty;
    sampler.frequency_penalty = frequencyPenalty;
    sampler.seed              = 42;     // Seed for random runs
    
    // Begin a context window
    ContextWindow context(context_size);
    
    std::cout << "\n";
    while (true) {
        std::cout << "> ";
        std::string keyboard_string;
        
        // User prompt
        if (!std::getline(std::cin, keyboard_string)) break;
        if (keyboard_string.empty() || keyboard_string[0] == ' ') 
            continue;
        std::vector<std::string> keyboard_splt = StringExplode(keyboard_string, ' ');
        
        // Check system function calls
        if (keyboard_splt[0][0] == '/') {
            // Remove leading slash
            keyboard_splt[0].erase(keyboard_splt[0].begin());
            
            // ================
            // Help section
            if (keyboard_splt[0] == "?") {
                std::cout << "source 'filename'     Set the source corpus document.\n";
                std::cout << "name 'filename'       Set the name for saving the model file.\n";
                std::cout << "load 'filename'       Load from a '.model' file.\n";
                std::cout << "save                  Save to a '.model' file.\n";
                std::cout << "unload                Unload and clear model memory.\n";
                std::cout << "clear                 Clear the context window.\n";
                std::cout << "train                 Begin a training cycle\n";
                continue;
            }
            
            // ================
            // Set a source corpus
            if (keyboard_splt[0] == "source") {
                if (keyboard_splt.size() > 1) {
                    trainingFilename = keyboard_splt[1];
                    if (!FileExists(trainingFilename)) {
                        std::cout << "Training source not found... '" << trainingFilename << "'\n\n";
                        continue;
                    }
                    std::cout << "Training source file set '" << trainingFilename << "'\n\n";
                }
                continue;
            }
            
            // ================
            // Set the model filename
            if (keyboard_splt[0] == "name") {
                if (keyboard_splt.size() > 1) {
                    modelFilename = keyboard_splt[1] + ".model";
                    std::cout << "Model filename set '" << modelFilename << "'\n\n";
                }
                continue;
            }
            
            // ================
            // Load a model
            if (keyboard_splt[0] == "load") {
                if (keyboard_splt.size() > 1) {
                    modelFilename = keyboard_splt[1] + std::string(".model");
                    if (!FileExists(modelFilename)) {
                        std::cout << "Model not found\n\n";
                        continue;
                    } else {
                        vocab.Clear();
                        model = LanguageModel();
                        std::cout << "Loading model package '" << modelFilename << "'\n";
                        
                        if (LoadModelPackage(modelFilename, model, vocab, trainer, epoch, learnRate, avgLoss)) {
                            std::cout << "Model ready...\n\n";
                            vocabB = vocab;
                            modelB = model;
                            continue;
                        }
                    }
                }
                if (modelFilename == "") {
                    std::cout << "Model filename not set '" << modelFilename << "'\n\n";
                    continue;
                }
                std::cout << "Error loading model package '" << modelFilename << "'\n\n";
                continue;
            }
            
            // ================
            // Save a model
            if (keyboard_splt[0] == "save") {
                if (keyboard_splt.size() > 1) 
                    modelFilename = keyboard_splt[1] + std::string(".model");
                std::cout << "Saving model package '" << modelFilename << "'\n\n";
                if (SaveModelPackage(modelFilename, model, vocab, trainer, epoch, learnRate, avgLoss)) 
                    continue;
                std::cout << "Error saving model package '" << modelFilename << "'\n\n";
                continue;
            }
            
            // ================
            // Unload the model
            if (keyboard_splt[0] == "unload") {
                std::cout << "Unloading and zeroing the model...\n\n";
                vocab.Clear();
                model = LanguageModel();
                context = ContextWindow(context_size);
                continue;
            }
            
            // ================
            // Clear context
            if (keyboard_splt[0] == "clear") {
                std::cout << "Resetting the context...\n\n";
                context = ContextWindow(context_size);
                continue;
            }
            
            // ================
            // Kick off a learning cycle
            if (keyboard_splt[0] == "train" || keyboard_splt[0] == "learn") {
                
                if (modelFilename == "") {
                    std::cout << "Model filename not set...\n\n";
                    continue;
                }
                
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
            if (keyboard_splt[0] == "rate") {
                if (keyboard_splt.size() > 1) {
                    std::istringstream iss(keyboard_splt[1]);
                    iss >> learnRate;
                    uint64_t ep = 0;
                    float oldLR = 0.0f;
                    if (FileExists(modelFilename)) 
                        UpdateModelLROnDisk(modelFilename, learnRate, &ep, &oldLR);
                }
                std::cout << "Learning rate  " << std::setprecision(4) << learnRate;
                std::cout << "\n\n";
                continue;
            }
            
            // ================
            // Set/get loss drop out
            if (keyboard_splt[0] == "loss") {
                if (keyboard_splt.size() > 1) {
                    std::istringstream iss(keyboard_splt[1]);
                    iss >> lossDrop;
                }
                std::cout << "Minimum loss drop out  " << std::setprecision(4) << lossDrop;
                std::cout << "\n\n";
                continue;
            }
            
            // ================
            // Toggle duo mode
            if (keyboard_splt[0] == "duo") {
                if (keyboard_splt.size() > 1) {
                    std::string v = keyboard_splt[1];
                    if (v == "on")  duoMode = true;
                    if (v == "off") duoMode = false;
                } else {
                    duoMode = !duoMode;
                }
                std::cout << "Duo mode: " << (duoMode ? "ON" : "OFF") << "\n\n";
                continue;
            }
            
            // ================
            // Get model details
            if (keyboard_splt[0] == "info") {
                std::cout << "---- Model details ----\n\n";
                
                std::cout << "n_ctx                " << model.n_ctx << "\n";
                std::cout << "d_model              " << model.d_model << "\n";
                std::cout << "n_layers             " << model.n_layers << "\n\n";
                
                std::cout << "Feed size            " << model.d_ff << "\n";
                std::cout << "attention heads      " << model.d_model / model.n_heads << "\n\n";
                
                std::cout << "Context window       " << context_size << "\n";
                std::cout << "Vocabulary           " << model.vocab_size << "\n\n";
                
                uint64_t paramCount = CalculateModelParameterCount(model.d_model, model.n_layers, model.d_ff, model.vocab_size, model.n_ctx, true, true);
                std::cout << "Total parameters     " << paramCount << "\n\n";
                
                continue;
            }
        }
        
        // Check if the main model is zeroed
        if (model.d_model == 0 || vocab.Size() == 0) {
            std::cout << "Model not loaded...\nUse /load 'name' to load a model\n\n";
            continue;
        }
        
        // Single-model or duo-model flow
        if (!duoMode) {
            ContextWindow userContextWindow(context_size);
            std::vector<Token>& userContext = userContextWindow.GetContext();
            userContext = Encode(vocab, keyboard_string);
            for (unsigned int id=0; id < userContext.size(); ++id) 
                context.Add( userContext[id] );
            
            (void)GenerateOnce(model, vocab, sampler, context, sentenceStruct, tokenCountMax, context_size);
            printLn();
            printLn();
            continue;
        }
        
        ContextWindow ctxA(context_size);
        ContextWindow ctxB(context_size);
        
        std::vector<Token> seedA = Encode(vocab, keyboard_string);
        for (size_t i = 0; i < seedA.size(); ++i) ctxA.Add(seedA[i]);
        
        // Kick off a duel between two models
        std::cout << "\n";
        int turn = 0;
        while (true) {
            // Model A
            print("A) ");
            std::string outA = GenerateOnce(model,  vocab,  sampler, ctxA, sentenceStruct, tokenCountMax, context_size);
            printLn();
            
            std::vector<Token> feedB = Encode(vocabB, outA);
            for (size_t i = 0; i < feedB.size(); ++i) ctxB.Add(feedB[i]);
            
            // Model B
            print("B) ");
            std::string outB = GenerateOnce(modelB, vocabB, sampler, ctxB, sentenceStruct, tokenCountMax, context_size);
            printLn();
            
            std::vector<Token> feedA = Encode(vocab, outB);
            for (size_t i = 0; i < feedA.size(); ++i) ctxA.Add(feedA[i]);
            
            if (KeyPressedNonBlocking()) 
                break;
        }
        
        printLn();
        continue;
    }
    return 0;
}


static void TrainModelGPU(std::string& trainingFilename, std::string& modelFilename,
                          LanguageModel& model, Tokenizer& vocab, Timer& time,
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
        vocab.FitToCorpus(corpus);
        model   = LanguageModel((int)vocab.Size(), layerWidth, headCount, feedWidth, layerDepth, contextSize);
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
            
            // Auto save
            if (time.check()) 
                SaveModelPackage(modelFilename, model, vocab, trainer,
                                (uint64_t)currentEpoch, currentLearningRate, runningAvgLoss);
            
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
                          LanguageModel& model, Tokenizer& vocab, Timer& time,
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
        vocab.FitToCorpus(corpus);
        model   = LanguageModel((int)vocab.Size(), layerWidth, headCount, feedWidth, layerDepth, contextSize);
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
                    
                    // Stride work assignment across the micro-batch
                    for (size_t j = startIdx + (size_t)t; j < endIdx; j += (size_t)threadsUsed) {
                        // Training pass
                        float sampleLoss = trainer.Step(model, inputSequences[j], targetTokens[j],
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
                
                SaveModelPackage(modelFilename, model, vocab, trainer, (uint64_t)currentEpoch, currentLearningRate, runningAvgLoss);
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
            
            // Auto save
            if (time.check()) 
                SaveModelPackage(modelFilename, model, vocab, trainer,
                                (uint64_t)currentEpoch, currentLearningRate, runningAvgLoss);
            
            // Allow user to abort
            if (KeyPressedNonBlocking()) {
                keepTraining = false;
                std::cout << "\n\n";
                return;
            }
            
        }
        
        // LR decay (same logic)
        float epochAverage = (runningSampleCount > 0) ? (runningLossSum / (float)runningSampleCount) : 0.0f;
        if (epochAverage < learningDecayBegin) {
            float next = learningRateDecay * currentLearningRate;
            if (next < learningRateMin) next = learningRateMin;
            currentLearningRate = next;
        }
        
        currentEpoch++;
        
        std::cout << "\n";
        
        
    }
}
