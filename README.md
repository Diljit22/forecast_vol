# Intraday Volatility Deep Learning
This repository contains an intraday volatility deep learning pipeline that processes high-frequency market data, extracts features, and trains various models.

Makes extensive use of parallelization, multi-processing, and balances chunking vs vectorization to handle a large dataset.

Attention-based architecture:
  - FeedForward: a 2-layer MLP sub-layer used in each AttentionBlock
  - AttentionBlock: multi-head self-attention + residual + layernorm + feed-forward
  - AttentionPredictor: stacks multiple AttentionBlocks, plus final linear output.

Implements GaussianHMM-based regime detection for each ticker's time-series data.

Trains GNN (self-supervised) to add weights.

Multi-scale fractal dimension estimation using wavelet decomposition.
GARCH(1,1) fitting approach via the arch

Implements wavelet-based Hurst exponent estimation for Rough FSV,
plus MLE fitting for (H, mu, sigma), and simulation.

Performs advanced hyperparameter tuning via optuna.
