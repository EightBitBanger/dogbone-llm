#include "BuildDataSet.h"

#include <random>

void BuildNextTokenDataset(const Tokenizer& vocab, const std::vector<std::string>& corpus, std::vector<std::vector<int>>& inputs, std::vector<std::vector<int>>& targets,
                           int block_len, int stride) {
    inputs.clear();
    targets.clear();
    if (block_len < 2) return;

    for (const std::string& line : corpus) {
        std::vector<int> ids = Encode(vocab, line);
        if (ids.size() < 2) continue;

        // Slide a window of `block_len` over ids with the given stride
        // Note: We create x = ids[t : t+block_len-1], y = ids[t+1 : t+block_len]
        // and right-pad with pad_id if needed (handled by Trainer/CE via pad_id).
        // Here we simply drop last incomplete windows smaller than 2 tokens;
        // exact padding to block_len is done by Trainer when it sees pad_id.
        const int n = (int)ids.size();
        for (int t = 0; t < n - 1; t += stride) {
            int end = t + block_len;
            if (end > n) end = n;
            int xlen = end - t - 1; // because y is shifted by +1
            if (xlen < 1) break;

            std::vector<int> x;
            std::vector<int> y;
            x.reserve((size_t)block_len);
            y.reserve((size_t)block_len);
            for (int i = 0; i < xlen; ++i) {
                x.push_back(ids[(size_t)(t + i)]);
                y.push_back(ids[(size_t)(t + i + 1)]);
            }
            inputs.push_back(std::move(x));
            targets.push_back(std::move(y));

            if (end == n) break;
        }
    }
}


void BuildFixedLengthCausalBlocks(const Tokenizer& vocab, const std::vector<std::string>& corpus, 
                                  std::vector<std::vector<int>>& inputs, std::vector<std::vector<int>>& targets, 
                                  int block_len, int stride, 
                                  bool drop_tail, bool add_bos_eos) {
    inputs.clear();
    targets.clear();
    if (block_len < 2 || stride < 1) return;
    
    const int pad = vocab.token.pad_id;
    const int bos = vocab.token.bos_id;
    const int eos = vocab.token.eos_id;
    
    // 1) Flatten corpus with BOS/EOS boundaries
    std::vector<int> ids_all;
    ids_all.reserve(1024 * 1024); // adjust if needed
    for (size_t i = 0; i < corpus.size(); ++i) {
        if (add_bos_eos) ids_all.push_back(bos);
        std::vector<int> ids = Encode(vocab, corpus[i]);
        ids_all.insert(ids_all.end(), ids.begin(), ids.end());
        if (add_bos_eos) ids_all.push_back(eos);
    }
    
    const int N = (int)ids_all.size();
    if (N < 2) return;
    
    // 2) Slide a window; typically stride == block_len (no overlap) or smaller for overlap
    for (int t = 0; t < N - 1; t += stride) {
        int end = t + block_len;
        std::vector<int> x((size_t)block_len, pad);
        std::vector<int> y((size_t)block_len, pad);
        
        if (end <= N) {
            // Full block
            for (int i = 0; i < block_len; ++i) {
                x[(size_t)i] = ids_all[(size_t)(t + i)];
                y[(size_t)i] = ids_all[(size_t)(t + i + 1)];
            }
            inputs.push_back(std::move(x));
            targets.push_back(std::move(y));
        } else {
            // Tail
            int avail = (N - 1) - t; // max valid pairs
            if (avail <= 0) break;
            if (drop_tail) break;
            
            for (int i = 0; i < avail; ++i) {
                x[(size_t)i] = ids_all[(size_t)(t + i)];
                y[(size_t)i] = ids_all[(size_t)(t + i + 1)];
            }
            inputs.push_back(std::move(x));
            targets.push_back(std::move(y));
            break; // done after tail
        }
    }
    
    // 3) Shuffle examples together (stable seeding; change seed if desired)
    if (!inputs.empty()) {
        std::vector<size_t> idx(inputs.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        std::mt19937 rng(42);
        std::shuffle(idx.begin(), idx.end(), rng);
        
        std::vector<std::vector<int>> inputs_shuf, targets_shuf;
        inputs_shuf.reserve(inputs.size());
        targets_shuf.reserve(targets.size());
        for (size_t k = 0; k < idx.size(); ++k) {
            inputs_shuf.push_back(std::move(inputs[idx[k]]));
            targets_shuf.push_back(std::move(targets[idx[k]]));
        }
        inputs.swap(inputs_shuf);
        targets.swap(targets_shuf);
    }
}
