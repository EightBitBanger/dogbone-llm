#ifndef ADAPTIVE_MOMENT_ESTIMATION_H
#define ADAPTIVE_MOMENT_ESTIMATION_H

#include <vector>
#include <algorithm>
#include <cmath>

struct AdamState { std::vector<float> m, v; };

struct Adam {
    float lr;
    float beta1;
    float beta2;
    float eps;
    int t;
    
    Adam();
    Adam(float learning_rate);
    
    static void StepInPlace(std::vector<float>& w, const std::vector<float>& g, AdamState& s, 
                            float lr, float b1, float b2, float eps, int t) {
        if (s.m.size() != w.size()) s.m.assign(w.size(), 0.0f);
        if (s.v.size() != w.size()) s.v.assign(w.size(), 0.0f);
        for (size_t i = 0; i < w.size(); i++) {
            s.m[i] = b1 * s.m[i] + (1.0f - b1) * g[i];
            s.v[i] = b2 * s.v[i] + (1.0f - b2) * (g[i] * g[i]);
            float mhat = s.m[i] / (1.0f - std::pow(b1, (float)t));
            float vhat = s.v[i] / (1.0f - std::pow(b2, (float)t));
            w[i] -= lr * mhat / (std::sqrt(vhat) + eps);
        }
    }
};

#endif
