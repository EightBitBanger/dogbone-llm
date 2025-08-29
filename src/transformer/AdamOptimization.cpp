#include "AdamOptimization.h"

Adam::Adam() : lr(1e-3f), beta1(0.9f), beta2(0.999f), eps(1e-8f), t(0) {}

Adam::Adam(float learning_rate) : lr(learning_rate), beta1(0.9f), beta2(0.999f), eps(1e-8f), t(0) {}

