#include "main.h"

int main() {
    std::srand(42);
    
    std::vector<std::string> corpus;
    corpus.push_back("in the beginning was the word");
    corpus.push_back("and the word was with god");
    corpus.push_back("and the word was god");
    
    Vocabulary vocab;
    FitVocab(vocab, corpus);
    
    std::string prompt = "in the beginning";
    std::vector<int> prompt_ids = Encode(vocab, prompt, true, false);
    
    int d_model = 64;
    int n_heads = 4;
    int d_ff    = 4 * d_model;
    int layers  = 2;
    int maxT    = 128;
    
    TransformerLM model(vocab.Size(), d_model, n_heads, d_ff, layers, maxT);
    
    // 4) (Optional) Training would go here:
    //    - Build (inputs, targets) pairs (shifted by 1)
    //    - Forward -> logits
    //    - Compute CE loss
    //    - Backprop (you’ll implement grads per layer)
    //    - Adam step
    
    // 5) Sampling (untrained = nonsense, but the flow is correct)
    std::vector<int> out_ids = Generate(model, prompt_ids, 10, vocab.eos_id);
    std::string out_text = Decode(vocab, out_ids);
    std::cout << out_text << "\n";
    
    return 0;
}
