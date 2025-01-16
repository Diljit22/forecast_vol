# Intraday Volatility Deep Learning
This repository contains an intraday volatility deep learning pipeline that processes high-frequency market data, extracts features, and trains various models
(including LSTM, CNN, and Attention-based architectures).
It also includes a multi-asset synergy module (e.g., GNNs, HMM-based regime detection) and a stochastic modeling module (e.g., GARCH, RFSV).

# Features
Data Processing and Preprocessing
Merges raw CSV data by ticker, filters active sessions, resamples to 1-minute bars, and extracts features (e.g., returns, log returns, volatility measures).

Multi-Asset Synergy
Aligns multiple tickers, builds correlation graphs, and runs GNN-based embedding to capture cross-asset relationships.

Stochastic Models
Supports GARCH, multi-scale fractal analysis, and wavelet-based RFSV for volatility estimation.

Deep Learning Models

LSTM
CNN
Attention
Hyperparameter Tuning (Optuna)

Notes
Still training the model! - it did well a smaller sample training set though. 
