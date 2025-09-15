#include "LanguageModelContextStream.h"

LanguageModelContextStream::LanguageModelContextStream(int context_size) : 
    mContextSize(context_size), 
    context(context_size) {}


Tokenizer& LanguageModelContextStream::Vocab() {
    return vocabulary;
}

LanguageModel& LanguageModelContextStream::Model() {
    return model;
}

ContextWindow& LanguageModelContextStream::Context() {
    return context;
}

int LanguageModelContextStream::GetMaxContext() const {
    return mContextSize;
}

void LanguageModelContextStream::SetMaxContext(int new_size) {
    mContextSize = new_size;
    context = ContextWindow(mContextSize);
}

void LanguageModelContextStream::ClearContext() {
    context = ContextWindow(mContextSize);
}

void LanguageModelContextStream::Unload() {
    vocabulary.Clear();
    model = LanguageModel();
    context = ContextWindow(mContextSize);
}

bool LanguageModelContextStream::LoadPackage(const std::string& filename, NeuralNetwork& trainer, uint64_t& epoch, float& learnRate, float& avgLoss) {
    bool ok = LoadModelPackage(filename, model, vocabulary, trainer, epoch, learnRate, avgLoss);
    return ok;
}

bool LanguageModelContextStream::SavePackage(const std::string& filename, NeuralNetwork& trainer, uint64_t epoch, float learnRate, float avgLoss) {
    return SaveModelPackage(filename, model, vocabulary, trainer, epoch, learnRate, avgLoss);
}

size_t LanguageModelContextStream::FeedString(const std::string& text) {
    std::vector<Token> toks = Encode(vocabulary, text);
    for (size_t i = 0; i < toks.size(); ++i) context.Add(toks[i]);
    return toks.size();
}

std::string TokensToString(const Tokenizer& vocab, const std::vector<Token>& tokens, size_t startIndex) {
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

std::string LanguageModelContextStream::GenerateStream(TokenSampler& Sampler, SamplingParams& sampler, ContextWindow& contextWindow, SentenceStructure& structure, int tokenCountMax) {
    // Sample until a word token is found to kick off the stream.
    // Prevents starting with a period and immediately ending the response stream.
    unsigned int repeat = 8;
    for(unsigned int counter=0; counter < repeat; counter++) {
        std::vector<TokenCandidate> candidate = Sampler.GetProbableTokens(model, contextWindow.GetContext(), sampler, vocabulary.Size(), 0.0f, true);
        // Check candidates list for an appropriate token
        for (unsigned int c=0; c < candidate.size(); c++) {
            Token token = candidate[c].id;
            std::string word = vocabulary[token];
            if (semantic.is_wordish(word)) {
                counter = repeat;
                break;
            }
        }
    }
    
    ContextWindow userContext( contextWindow.Size() );
    structure.sentenceCounter = 0;
    structure.wordsCounter = 0;
    userContext.Clear();
    
    std::vector<Token>& context = contextWindow.GetContext();
    size_t startSize = context.size();
    
    for (int t = 0; t < tokenCountMax; t++) {
        if (!semantic.ProcessTokenStream(model, vocabulary, Sampler, sampler, contextWindow, userContext, structure)) break;
        if (KeyPressedNonBlocking()) break;
    }
    return TokensToString(vocabulary, context, startSize);
}

bool LanguageModelContextStream::IsReady() const {
    return (model.d_model != 0) && (vocabulary.Size() > 0);
}


