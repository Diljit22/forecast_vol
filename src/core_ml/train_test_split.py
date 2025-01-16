import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_train_test(
    df: pd.DataFrame, target_col: str, test_ratio: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the given processed DataFrame into training and test sets (e.g., 80/20).
    This function is designed for a final training dataset, leaving inference data separate.
    """

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, shuffle=True
    )

    logger.info(
        "Split data into train (%d rows) and test (%d rows). Ratio=%.2f (Train=%.2f%%, Test=%.2f%%)",
        X_train.shape[0],
        X_test.shape[0],
        test_ratio,
        (1 - test_ratio) * 100,
        test_ratio * 100,
    )

    return X_train, X_test, y_train, y_test
