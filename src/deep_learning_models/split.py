import logging
import time
import pandas as pd
import torch
from typing import List, Dict, Any
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


def time_split_three_way(
    df: pd.DataFrame,
    time_col: str = "t",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    target_cols: List[str] = ["rv"]
):
    """
    Splits df chronologically into train, val, test sets:
     - train_frac => fraction of rows for train
     - val_frac   => fraction for val
     - remainder  => test
    Returns: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("[time_split_three_way] Sorting by time column for chronological split ...")
    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)

    n = len(df_sorted)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_df = df_sorted.iloc[:n_train]
    val_df   = df_sorted.iloc[n_train : n_train + n_val]
    test_df  = df_sorted.iloc[n_train + n_val :]

    X_train = train_df.drop(columns=target_cols)
    y_train = train_df[target_cols]

    X_val   = val_df.drop(columns=target_cols)
    y_val   = val_df[target_cols]

    X_test  = test_df.drop(columns=target_cols)
    y_test  = test_df[target_cols]

    logger.info(f"[time_split_three_way] Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


class TimeSeriesDataset(Dataset):
    """
    On-the-fly dataset for time-series sequences of length seq_len.
    shape per item:
       X => [seq_len, num_features]
       y => [num_targets]
    """
    def __init__(
        self,
        X_df: pd.DataFrame,
        y_df: pd.DataFrame,
        seq_len: int,
        time_col: str = "t"
    ):
        super().__init__()
        self.seq_len = seq_len

        bool_cols = X_df.select_dtypes("bool").columns
        for bc in bool_cols:
            X_df[bc] = X_df[bc].astype(float)

        self.X_df = X_df.reset_index(drop=True)
        self.y_df = y_df.reset_index(drop=True)

        self.X_np = self.X_df.values  # shape => [N, num_features]
        self.Y_np = self.y_df.values  # shape => [N, num_targets]

        self.N = len(self.X_df) - seq_len
        if self.N < 0:
            logger.warning(f"[TimeSeriesDataset] seq_len={seq_len} > data rows => empty dataset.")

    def __len__(self):
        return max(0, self.N)

    def __getitem__(self, idx: int):
        seq_start = idx
        seq_end   = idx + self.seq_len

        X_window = self.X_np[seq_start:seq_end]   # shape => [seq_len, num_features]
        y_target = self.Y_np[seq_end]             # shape => [num_targets]

        X_tensor = torch.tensor(X_window, dtype=torch.float32)
        y_tensor = torch.tensor(y_target, dtype=torch.float32)

        return X_tensor, y_tensor


def pipeline_train_val_test_on_the_fly(
    config: Dict[str, Any],
    seq_len: int = 30,
    target_cols: List[str] = ["rv"],
    time_col: str = "t"
):
    t0 = time.time()
    logger.info("[pipeline_train_val_test_on_the_fly] Starting pipeline ...")

    csv_path = config["data"].get("path_final_enriched", "data/final/enriched.csv")
    logger.info(f"[LOAD] Reading CSV from {csv_path} ...")
    df = pd.read_csv(csv_path)
    logger.info(f"[LOAD] Loaded shape={df.shape}, columns={df.columns.tolist()}")

    # e.g. drop 'ticker' column if it's object
    if "ticker" in df.columns and df["ticker"].dtype == object:
        logger.info("[pipeline] dropping 'ticker' col (object dtype)...")
        df.drop(columns=["ticker"], inplace=True)

    # Further cleaning can be done here if needed.

    # 2) time-split
    X_train, X_val, X_test, y_train, y_val, y_test = time_split_three_way(
        df,
        time_col=time_col,
        train_frac=0.8,
        val_frac=0.1,
        target_cols=target_cols
    )

    # 3) Build TimeSeriesDatasets
    train_ds = TimeSeriesDataset(X_train, y_train, seq_len=seq_len, time_col=time_col)
    val_ds   = TimeSeriesDataset(X_val,   y_val,   seq_len=seq_len, time_col=time_col)
    test_ds  = TimeSeriesDataset(X_test,  y_test,  seq_len=seq_len, time_col=time_col)

    logger.info(f"[pipeline] train_ds={len(train_ds)}, val_ds={len(val_ds)}, test_ds={len(test_ds)}")
    logger.info(f"[pipeline_train_val_test_on_the_fly] Done in {time.time() - t0:.2f}s.")
    return train_ds, val_ds, test_ds


def create_data_loaders(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    dl_cfg: Dict[str, Any],
    device: str = "cpu"
):
    """
    - multiple workers
    - pinned memory if GPU
    """
    logger.info("[create_data_loaders] Building DataLoaders ...")
    batch_size = dl_cfg.get("batch_size", 32)
    num_workers = dl_cfg.get("num_workers", 4)
    pin_mem = (device == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem
    )
    return train_loader, val_loader, test_loader
