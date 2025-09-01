#ifndef CROSS_ENTROPY_LOSS_H
#define CROSS_ENTROPY_LOSS_H

#include <limits>
#include <cmath>
#include <cassert>

static float CrossEntropyLoss(const Tensor2D& logits, const std::vector<int>& targets, int pad_id) {
    const int T = logits.R;
    const int V = logits.C;
    assert((int)targets.size() >= T && "targets shorter than sequence length");

    float total = 0.0f;
    int   count = 0;

    for (int t = 0; t < T; ++t) {
        const int y = targets[(size_t)t];
        if (y == pad_id) continue;
        assert(y >= 0 && y < V && "target index out of range");

        const float* row = logits.Row(t);

        // max for numerical stability
        float maxv = -std::numeric_limits<float>::infinity();
        for (int v = 0; v < V; ++v) if (row[v] > maxv) maxv = row[v];

        // log-sum-exp
        double denom = 0.0;
        for (int v = 0; v < V; ++v) denom += std::exp((double)row[v] - (double)maxv);
        const float logsumexp = maxv + (float)std::log(denom);

        // nll for the correct class
        total += (logsumexp - row[y]);
        ++count;
    }

    return (count == 0) ? 0.0f : (total / (float)count);
}

#endif
