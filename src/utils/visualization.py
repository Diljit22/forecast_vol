import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from typing import List

logger = logging.getLogger(__name__)


def plot_split_distribution(df_splitted: pd.DataFrame, out_dir="plots"):
    """
    Bar chart of how many rows in each split: train/val/test.
    """
    os.makedirs(out_dir, exist_ok=True)
    split_counts = df_splitted["split"].value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(
        y=split_counts.values, hue=split_counts.index, palette="Set2", legend=False
    )
    plt.title("Data Split Distribution")
    plt.xlabel("Split")
    plt.ylabel("Count of Rows")
    out_path = os.path.join(out_dir, "split_distribution.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Saved split distribution plot => {out_path}")


def plot_training_loss(loss_history: List[float], out_dir="plots"):
    """
    Plots the training loss over epochs.
    loss_history is a list of float losses for each epoch.
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Training Loss", color="blue")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    out_path = os.path.join(out_dir, "training_loss.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Saved training loss plot => {out_path}")


def plot_final_predictions(
    df_test: pd.DataFrame,
    seq_len: int,
    all_features: List[str],
    target_cols: List[str],
    preds: np.ndarray,
    out_dir="plots",
):
    """
    Compares final predicted values vs. actual for the test set.
    """
    os.makedirs(out_dir, exist_ok=True)
    if len(preds) == 0:
        logger.warning("No predictions to plot. Skipping final predictions plot.")
        return

    out_path = os.path.join(out_dir, "final_predictions.png")

    skip_rows = seq_len
    df_plot = df_test.iloc[skip_rows:].copy().reset_index(drop=True)
    df_plot["pred_0"] = preds[:, 0]

    if preds.shape[1] > 1:
        df_plot["pred_1"] = preds[:, 1]

    # actual rv vs. pred_0
    actual_col = target_cols[0]  # e.g. "rv"
    if actual_col in df_plot.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(
            df_plot["timestamp"],
            df_plot[actual_col],
            label=f"Actual {actual_col}",
            color="red",
        )
        plt.plot(df_plot["timestamp"], df_plot["pred_0"], label="Pred 0", color="blue")
        if (
            preds.shape[1] > 1
            and len(target_cols) > 1
            and target_cols[1] in df_plot.columns
        ):
            plt.plot(
                df_plot["timestamp"],
                df_plot[target_cols[1]],
                label=f"Actual {target_cols[1]}",
                color="green",
            )
            plt.plot(
                df_plot["timestamp"], df_plot["pred_1"], label="Pred 1", color="purple"
            )

        plt.title("Final Predictions vs. Actual")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved final predictions plot => {out_path}")
    else:
        logger.warning(
            f"Target col {actual_col} not found in df_test. Skip final predictions plot."
        )
