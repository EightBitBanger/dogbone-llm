#ifndef SAMPLER_H
#define SAMPLER_H

static int ArgMax(const float* row, int V) {
    int best = 0;
    float bv = row[0];
    for (int v = 1; v < V; v++) {
        if (row[v] > bv) { bv = row[v]; best = v; }
    }
    return best;
}

static int GreedyNextToken(const TransformerLM& model, const std::vector<int>& context_ids) {
    Tensor2D logits = model.Forward(context_ids);
    const float* row = logits.Row(logits.R - 1);
    return ArgMax(row, logits.C);
}

static std::vector<int> Generate(const TransformerLM& model,
                                 const std::vector<int>& prompt_ids,
                                 int max_new_tokens, int eos_id) {
    std::vector<int> ids = prompt_ids;
    for (int i = 0; i < max_new_tokens; i++) {
        int nxt = GreedyNextToken(model, ids);
        if ((int)prompt_ids.size() > maxT) {
            prompt_ids.erase(prompt_ids.begin(), prompt_ids.end() - maxT);
        }
        ids.push_back(nxt);
        if (nxt == eos_id) break;
    }
    return ids;
}

#endif
