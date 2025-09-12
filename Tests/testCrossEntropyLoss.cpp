#include "test.h"
#include "../Transformer/CrossEntropyLoss.h"

#include <iostream>
#include <cmath>
#include <cstdint>

static inline bool almost_equal(float a, float b, float tol = 1e-6f) {
    return std::fabs(a - b) <= tol;
}

bool TestCrossEntropyLoss() {
    // ---- Case 1: No padding, closed-form expected ----
    {
        Tensor2D logits(2, 3);
        // Row 0: [0,0,0]  -> loss = log(3)
        logits(0,0)=0.0f; logits(0,1)=0.0f; logits(0,2)=0.0f;
        // Row 1: [0,1,0]  -> loss = log(2+e) - 1
        logits(1,0)=0.0f; logits(1,1)=1.0f; logits(1,2)=0.0f;

        std::vector<int> targets = {0, 1};
        const int pad_id = -1;

        float loss = CrossEntropyLoss(logits, targets, pad_id);

        const double e1 = std::log(3.0);                     // row0
        const double e2 = std::log(2.0 + std::exp(1.0)) - 1.0; // row1
        const float expected = (float)((e1 + e2) / 2.0);

        if (!almost_equal(loss, expected)) return false;
    }

    // ---- Case 2: With padding (ignored in average) ----
    {
        Tensor2D logits(3, 2);
        // Row 0: [0,0] -> loss = log(2)  (target=1)
        logits(0,0)=0.0f; logits(0,1)=0.0f;
        // Row 1: [5,0] -> loss = log(exp(5)+1) - 5  (target=0)
        logits(1,0)=5.0f; logits(1,1)=0.0f;
        // Row 2: anything; target is PAD -> ignored
        logits(2,0)=1.2f; logits(2,1)=-0.7f;

        std::vector<int> targets = {1, 0, -1};
        const int pad_id = -1;

        float loss = CrossEntropyLoss(logits, targets, pad_id);

        const double l0 = std::log(2.0);
        const double l1 = std::log(std::exp(5.0) + 1.0) - 5.0;
        const float expected = (float)((l0 + l1) / 2.0);

        if (!almost_equal(loss, expected)) return false;
    }

    // ---- Case 3: All padded -> returns 0.0 ----
    {
        Tensor2D logits(2, 4);
        logits(0,0)=0.5f; logits(0,1)=-1.0f; logits(0,2)=2.0f; logits(0,3)=0.0f;
        logits(1,0)=-0.3f; logits(1,1)=0.4f; logits(1,2)=-0.2f; logits(1,3)=1.7f;

        std::vector<int> targets = {-1, -1};
        const int pad_id = -1;

        float loss = CrossEntropyLoss(logits, targets, pad_id);
        if (!almost_equal(loss, 0.0f, 0.0f)) return false; // exact 0
    }

    return true;
}
