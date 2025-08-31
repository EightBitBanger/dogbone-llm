#include "main.h"


int main() {
    std::srand(42);
    
    std::string trainingFilename  = "training.txt";
    std::string modelFilename     = "dataset.model";
    
    // Model dimensions
    const int   inputWidth            = 128;             // Input context layer width
    const int   layerWidth            = 128;             // Hidden layer width 'd_model'
    const int   layerDepth            = 4;               // Number of hidden layers
    const int   feedWidth             = 4 * layerWidth;  // Internal hidden layer expansion (width > feedWidth > width)
    
    // Optimizer
    const float learningRate          = 0.001f;          // Rate of training to be applied per epoch
    const float learningRateMin       = 0.001f;          // Minimum learning rate after decay
    const float learningRateDecay     = 1.0f;            // Rate of learning decay per epoch
    const float learningLossDropout   = 3.0f;            // Stop training at this loss level
    
    // Sampler
    const float samplerTemp           =  0.8f;           // Temperature  0.7  1.3
    const int   samplerTopK           =  8;              // 0 will disable
    const float samplerTopP           =  0.7f;           // 1.0 will disable
    
    // Context
    const int   contextLength         = inputWidth;      // How many tokens should be remembered (must be less than or equal to inputWidth)
    const int   contextHeadCount      = 8;               // Context attention heads
    
    // Sentence structuring
    const int tokenCountMax           = 80;              // Absolute max number of tokens to return as the response
    const int wordsPerSentence        = 16;              // Max number of words per sentence
    const int sentenceCountMax        = 2;               // Max number of sentences or strings of tokens broken by a period
    
    if (inputWidth < 1) {
        std::cerr << "inputWidth must be >= 1\n";
        return 1;
    }
    if (contextHeadCount < 1) {
        std::cerr << "contextHeadCount must be >= 1\n";
        return 1;
    }
    if ((layerWidth % contextHeadCount) != 0) {
        std::cerr << "layerWidth must be divisible by contextHeadCount\n";
        return 1;
    }
    
    uint64_t epoch;
    float avgLoss;
    float learnMul = learningRate;
    float lossDrop = learningLossDropout;
    
    Vocabulary vocab;
    LauguageModel model;
    
    // Set the activation function
    Activation.SetActivationFunction( ActivationType::SWIGLU );
    
    // Check for the model file to load raw training text
    std::string trainingText;
    if (!LoadModelPackage(modelFilename, model, vocab, trainer, epoch, learnMul, avgLoss)) {
        // Check source training text
        if (!FileTextLoad(trainingFilename, trainingText) || trainingText.empty()) {
            std::cout << "Training data file not found or empty 'training.txt'..." << "\n";
        }
        std::cout << "Model package not found..." << "\n";
        
        TrainModel(trainingFilename, modelFilename, model, vocab,
                        layerWidth, contextHeadCount, feedWidth, layerDepth, inputWidth,
                        learnMul, learningRateMin, learningRateDecay, avgLoss, lossDrop);
        
    } else {
        
        std::cout << "Model package loaded" << "\n";
    }
    
    // Setup sentence structuring
    SentenceStructure sentenceStruct;
    sentenceStruct.sentenceCountMax = sentenceCountMax;
    sentenceStruct.wordsPerSentence = wordsPerSentence;
    
    // Fire up the sampler
    SamplingParams sampler;
    sampler.temperature       = samplerTemp;
    sampler.top_k             = std::min(vocab.Size(), samplerTopK);
    sampler.top_p             = samplerTopP;
    sampler.presence_penalty  = 0.7f;   // 0.5 - 1.0  if you see loops/repeats
    sampler.frequency_penalty = 0.5f;   // 0.5 - 1.0  for stronger anti-repetition
    sampler.seed              = 42;     // Seed for random runs
    
    ContextWindow context(contextLength);
    
    std::cout << "\n";
    while (true) {
        std::cout << "> ";
        std::string keyboard_string;
        if (!std::getline(std::cin, keyboard_string)) break;
        
        // Check system function calls
        if (keyboard_string[0] == '/') {
            keyboard_string.erase(keyboard_string.begin());
            
            // ================
            // Purge the model
            if (keyboard_string == "reset") {
                std::cout << "Resetting the model...\n";
                FileDelete(modelFilename);
                FileDelete(modelFilename + std::string(".state"));
                vocab = Vocabulary();
                model = LauguageModel();
                std::cout << "Use /train to retrain a new model.\n\n";
                continue;
            }
            
            // ================
            // Kick off a learning cycle
            if (keyboard_string == "train" || keyboard_string == "learn") {
                TrainModel(trainingFilename, modelFilename, model, vocab,
                        layerWidth, contextHeadCount, feedWidth, layerDepth, inputWidth,
                        learnMul, learningRateMin, learningRateDecay, avgLoss, lossDrop);
                continue;
            }
            
            // ================
            // Set/get the learning rate
            if (keyboard_string.rfind("rate", 0) == 0) {
                std::istringstream iss(keyboard_string);
                std::string cmd;
                iss >> cmd;
                if (iss >> learnMul) {
                    std::cout << "Learning rate  " << std::setprecision(4) << learnMul;
                    uint64_t ep = 0; float oldLR = 0.0f;
                    if (FileExists(modelFilename)) 
                        UpdateModelLROnDisk(modelFilename, learnMul, &ep, &oldLR);
                } else {
                    std::cout << "Learning rate  " << std::setprecision(4) << learnMul;
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
                std::cout << "Loss minimum " << std::setprecision(4) << lossDrop;
                std::cout << "\n\n";
                continue;
            }
            
            std::cout << "Unknown command '" << keyboard_string << "'\n\n";
            continue;
        }
        
        if (keyboard_string == "") 
            continue;
        
        // Encode user prompt
        ContextWindow userContextWindow(contextLength);
        std::vector<Token>& userContext = userContextWindow.GetContext();
        userContext = Encode(vocab, keyboard_string, true, false);
        for (unsigned int id=0; id < userContext.size(); id++) 
            context.Add( userContext[id] );
        
        // Reset the context and counters
        userContextWindow.Clear();
        
        sentenceStruct.sentenceCounter = 0;
        sentenceStruct.wordsCounter = 0;
        
        // Kick off a response stream
        for (unsigned int tokenCount=0; tokenCount < tokenCountMax; tokenCount++) {
            
            // Process the token stream
            if (!semantic.ProcessTokenStream(model, vocab, sampler, context, userContextWindow, sentenceStruct)) 
                break;
            
            // Terminate the token steam early
            if (KeyPressedNonBlocking()) 
                break;
        }
        
        std::cout << "\n\n";
    }
    return 0;
}





static void TrainModel(std::string& trainingFilename, std::string& modelFilename,
                       LauguageModel& model, Vocabulary& vocab,
                       int layerWidth, int headCount, int feedWidth, int layerDepth, int contextSize, 
                       float learningRate, float learningRateMin, float learningRateDecay, 
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
        FitVocab(vocab, corpus);
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
                  << "   Rate  " << std::setprecision(4) << currentLearningRate << "  "
                  << "   Loss  " << std::fixed << std::setprecision(8) << std::setw(1) << avgLoss << "\n";
    }
    
    // Determine logical processors
    int logicalProcs = 0;
    SYSTEM_INFO si; GetSystemInfo(&si);
    logicalProcs = (int)si.dwNumberOfProcessors;
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
        
        trainer.opt.lr = currentLearningRate;
        
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
                    localTrainer.opt.lr = trainer.opt.lr;
                    
                    // Stride work assignment across the micro-batch
                    for (size_t j = startIdx + (size_t)t; j < endIdx; j += (size_t)threadsUsed) {
                        // Training pass
                        float sampleLoss = trainer.Step(model, inputSequences[j], targetTokens[j],
                                                        vocab.pad_id, &threadGrads[(size_t)t], false);
                        
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
                      << "   Rate  " << std::setprecision(4) << currentLearningRate << "  "
                      << "   Loss  " << std::fixed << std::setprecision(8) << std::setw(1) << runningAvgLoss << "   "
                      << "   " << std::setprecision(3) << percentTowardDropout << " %       ";
            
            if (epochProgress < 100.0f) {
                std::cout << "   " << std::setprecision(3) << epochProgress << " %       ";
            } else {
                std::cout << "                   "; // Clear the line
            }
            
            std::cout << std::flush;
            
            // Save weights + vocab + optimizer + epoch/LR
            SaveModelPackage(modelFilename, model, vocab, trainer,
                            (uint64_t)currentEpoch, currentLearningRate, runningAvgLoss);
            
            // Allow user to abort
            if (KeyPressedNonBlocking()) {
                keepTraining = false;
                std::cout << "\n\n";
                return;
            }
        }
        
        currentEpoch++;
        
        std::cout << "\n";
        
        // LR decay (same logic)
        float epochAverage = (runningSampleCount > 0) ? (runningLossSum / (float)runningSampleCount) : 0.0f;
        if (epochAverage < 1.0f) {
            float next = learningRateDecay * currentLearningRate;
            if (next < learningRateMin) next = learningRateMin;
            currentLearningRate = next;
        }
    }
}
