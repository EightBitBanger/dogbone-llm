#ifndef CROSS_ENTROPY_LOSS_H
#define CROSS_ENTROPY_LOSS_H

static float CrossEntropyLoss(const Tensor2D& logits, const std::vector<int>& targets, int pad_id) {
    int T = logits.R;
    int V = logits.C;
    float total = 0.0f;
    int count = 0;
    for (int t = 0; t < T; t++) {
        int y = targets[(size_t)t];
        if (y == pad_id) continue;
        const float* row = logits.Row(t);
        float maxv = -std::numeric_limits<float>::infinity();
        for (int v = 0; v < V; v++) if (row[v] > maxv) maxv = row[v];
        float denom = 0.0f;
        for (int v = 0; v < V; v++) denom += std::exp(row[v] - maxv);
        float logprob = (row[y] - maxv) - std::log(denom);
        total += -logprob;
        count += 1;
    }
    if (count == 0) return 0.0f;
    return total / (float)count;
}

#endif
