#ifndef ADAPTIVE_MOMENT_ESTIMATION_H
#define ADAPTIVE_MOMENT_ESTIMATION_H

#include <vector>
#include <algorithm>
#include <cmath>

struct AdamState { std::vector<float> m, v; };

class Adam {
public:
    
    float learning_rate;
    float beta_m;
    float beta_v;
    float epsilon;
    int step;
    
    Adam();
    Adam(float lr);
    
    static void StepInPlace(std::vector<float>& w, const std::vector<float>& g, AdamState& s, 
                            float lr, float b1, float b2, float eps, int t) {
        if (s.m.size() != w.size()) s.m.assign(w.size(), 0.0f);
        if (s.v.size() != w.size()) s.v.assign(w.size(), 0.0f);
        // Guard step: Adam bias correction requires t >= 1
        if (t < 1) t = 1;
        // Global grad-norm clipping (max 1.0)
        double gn2 = 0.0;
        for (size_t i = 0; i < g.size(); ++i) { double gi = (double)g[i]; gn2 += gi*gi; }
        double gnorm = std::sqrt(gn2);
        float clip = 1.0f;
        float scale = (gnorm > clip && gnorm > 0.0) ? (clip / (float)gnorm) : 1.0f;
        for (size_t i = 0; i < w.size(); i++) {
            float gi = g[i] * scale;
            s.m[i] = b1 * s.m[i] + (1.0f - b1) * gi;
            s.v[i] = b2 * s.v[i] + (1.0f - b2) * (gi * gi);
            float mhat = s.m[i] / (1.0f - std::pow(b1, (float)t));
            float vhat = s.v[i] / (1.0f - std::pow(b2, (float)t));
            w[i] -= lr * mhat / (std::sqrt(vhat) + eps);
        }
    }
};

#endif
