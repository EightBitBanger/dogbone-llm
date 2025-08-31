# Transformer Language Model (C++ / CPU/GPU)

> A compact, transformer stack written in c++ with a simple repl interface, multithreaded CPU training, optional GPU acceleration

<br>

## Features
- Transformer
  - Multi-Head self attention
  - Adam (adaptive moment estimation) optimization
  - Multi-threaded gradient accumulator
- Activation
  - GELU (Gaussian error linear unit)
  - RELU (Rectified linear unit)
  - SiLU (Sigmoid linear unit)
  - Mish (Self regularized non-monotonic)
  - SwiGLU (Swish and GLU)
- Sampler
  - Top-P/Top-K
