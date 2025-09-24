#ifndef BUILD_TOKEN_DATA_SETS_H
#define BUILD_TOKEN_DATA_SETS_H

#include <string>
#include <vector>

#include "Tokenizer.h"

// Fixed-length causal data set blocks
void BuildFixedLengthCausalBlocks(const Tokenizer& vocab, const std::vector<std::string>& corpus, 
                                  std::vector<std::vector<int>>& inputs, std::vector<std::vector<int>>& targets, 
                                  int block_len, int stride, 
                                  bool drop_tail, bool add_bos_eos);

void BuildNextTokenDataset(const Tokenizer& vocab, const std::vector<std::string>& corpus, std::vector<std::vector<int>>& inputs, std::vector<std::vector<int>>& targets,
                           int block_len, int stride);

#endif
