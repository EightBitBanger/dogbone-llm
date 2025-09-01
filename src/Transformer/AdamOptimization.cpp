#include "AdamOptimization.h"

Adam::Adam() : learning_rate(1e-3f), beta_m(0.9f), beta_v(0.999f), epsilon(1e-8f), step(0) {}

Adam::Adam(float lr) : learning_rate(lr), beta_m(0.9f), beta_v(0.999f), epsilon(1e-8f), step(0) {}

