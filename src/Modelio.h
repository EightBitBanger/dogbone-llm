#ifndef MODELIO_H
#define MODELIO_H

#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

#include "Transformer/Transformer.h"
#include "tokenizer.h"

// Package format byte layout
//   [8]   magic        = "MODELPKG"
//   [4]   version      = 1 (LE32)
//   [4]   flags        = 0 (LE32)
//   [8]   off_model    (LE64, from file start)
//   [8]   size_model   (LE64)
//   [8]   off_vocab    (LE64)
//   [8]   size_vocab   (LE64)
//   [64]  reserved     (zeros)
//   [...] model blob   (your TRANSMLP block)
//   [...] vocab blob   (your VOCB block)

// Section directory entry
struct ModelPkgSection {
    char     id[4];     // e.g. 'H','P','R','M'
    uint64_t offset;    // from file start
    uint64_t size;      // bytes
};

// Small POD for hyper parameters
struct ModelDims {
    int32_t vocab_size;
    int32_t d_model;
    int32_t n_heads;
    int32_t d_ff;
    int32_t n_layers;
    int32_t max_T;
};

inline bool StreamWriteLE32(std::ostream& out, uint32_t v);
inline bool StreamReadLE32(std::istream& in, uint32_t& v);

// Write/read the ModelDims payload (little-endian int32s)
inline bool WriteModelDims(std::ostream& out, const ModelDims& d);
inline bool ReadModelDims(std::istream& in, ModelDims& d);

inline bool StreamWriteLE64(std::ostream& out, uint64_t v);
inline bool StreamReadLE64(std::istream& in, uint64_t& v);

bool SaveVocabBinaryToStream(const Vocabulary& vocab, std::ostream& out);
bool LoadVocabBinaryFromStream(Vocabulary& vocab, std::istream& in);

// Optimizer state save/load
void EnsureAdamSize(AdamState& s, size_t n);

// write: [u32 len][len floats m][len floats v]
bool WriteAdamState(std::ostream& out, const AdamState& s, size_t expected_len);

// read: [u32 len][len m][len v] -> sized to expected_len (truncate/pad with 0)
bool ReadAdamState(std::istream& in, AdamState& s, size_t expected_len);

// Save model + vocab + optimizer state into a single package.
bool SaveModelPackage(const std::string& path, const TransformerLauguageModel& model, const Vocabulary& vocab,
                      const NeuralNetwork& trainer, uint64_t epoch, float current_lr, float last_avg_loss);

// Load model + vocab + optimizer if present.
// Returns true on success. If OPTS missing, model+vocab still loaded and trainer remains default.
bool LoadModelPackage(const std::string& path, TransformerLauguageModel& model, Vocabulary& vocab, NeuralNetwork& trainer, 
                      uint64_t& epoch_out, float& current_lr_out, float& last_loss_out);

// Persist-only LR update: loads package, updates trainer.opt.lr, re-saves package.
// Leaves weights, epoch, Adam moments, etc. untouched.
bool UpdateModelLROnDisk(const std::string& modelPath, float newLR, uint64_t* outEpoch = NULL, float* outOldLR = NULL);

// Basic file functions
bool FileTextLoad(std::string& path, std::string& out);
bool FileExists(std::string filename);
bool FileDelete(std::string filename);

#endif
