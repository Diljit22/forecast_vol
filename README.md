# Volatility Forecasting with Attention Networks

This repository contains a pipeline to perform **time-series volatility forecasting** using an **Attention-based deep learning model**. 
Fully enriched data has ~130M entries so parallelization, multi-processing, and vectorization while chunking are implemented whenever possible.

The pipeline handles:

1. **Data Loading & Preprocessing**  
   - Reads raw and intermediate CSV files to produce a single **final enriched CSV**.
   - Prepares columns needed for the forecasting model (e.g., `rv` for realized volatility - the "target" col.).
   - Splits data chronologically into train, validation, and test sets.
   - Converts features to float32, handles boolean columns, and optionally drops non-numeric columns (e.g. `ticker`).

2. **Multi-Asset Synergy**
   - Implements GaussianHMM-based regime detection for each ticker's time-series data.
   - Trains GNN (self-supervised) to add weights.
   - Runs in-memory hyperparam tuning for GNN implementation.
  
3. **Stochastic Models**
   - GARCH(1,1) fitting is done via arch.
   - Multi-scale fractal dimension estimation using wavelet decomposition is perfromed.
   - Implements wavelet-based Hurst exponent estimation for Rough FSV.
   - Runs in-memory hyperparam tuning for fitting (H, mu, sigma), and path simulation.
  
2. **Attention-Based Volatility Model**  
   - An **AttentionPredictor** that processes multi-timestep sequences:
     - Multiple self-attention blocks, each containing multi-head attention + small feed-forward sub-layer.
     - Final linear layer to predict next-step volatility (`rv`).
   - Configurable hyperparameters (layer count, hidden dimension, dropout, feed-forward dimension, etc.).

3. **Hyperparameter Search (Optuna)**  
   - Runs in-memory hyperparam tuning for the attention model, using the train & validation splits.
   - Configurable search space (e.g., `d_model`, `num_layers`, `ff_dim`, `dropout`, `learning_rate`).
   - Picks best combination (lowest validation MSE), merges them back into the model config, final training occurs on train+val combined.

4. **End-to-End Pipeline**  
     - Load a final enriched CSV from disk.
     - Split into `(train_ds, val_ds, test_ds)` with time-based partitioning.
     - Multi-asset Synergy is done.
     - Stochastic models are fitted.
     - Perform hyperparam search on `(train_ds, val_ds)`.
     - Retrain the **AttentionPredictor** with best hyperparams.
     - Evaluate on the test set (logging final MSE).

## Notes

Key Configuration Files
- `configs/` contains information for all configurations.

Windows Multiprocessing; when using spawn start method, you may see repeated logs about config merges.
To reduce set logging to WARN in child processes.

Hyperparameter Tuning
- By default, we set n_jobs=1 for Optuna to avoid errors with large data + Windows spawn.
- The tuning logic is in src/deep_learning_models/hyperparam_search.py, inside run_in_memory_hpo().
- You can customize the search space for dropout, layers, learning rate, etc.

Large Datasets
- For 2,500,000+ rows, see HPC notes: use a higher num_workers in DataLoader, pinned memory if on GPU.
- The code uses chunk-based iteration for training.
- Fully enriched data has ~130M entries.
