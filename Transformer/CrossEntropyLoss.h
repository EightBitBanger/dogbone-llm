#ifndef CROSS_ENTROPY_LOSS_H
#define CROSS_ENTROPY_LOSS_H

#include <limits>
#include <cmath>
#include <cassert>
#include "Tensor2D.h"

float CrossEntropyLoss(const Tensor2D& logits, const std::vector<int>& targets, int pad_id);

#endif
