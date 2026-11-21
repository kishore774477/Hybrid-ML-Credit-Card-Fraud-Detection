"""Data loading, splitting and scaling.

The scaler is fitted on the training split only, so no information from the
validation or test sets leaks into training.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.config import (
    DATA_PATH,
    FEATURE_COLS,
    RANDOM_STATE,
    RAW_COLS,
    SCALE_COLS,
    TARGET_COL,
)


def load_data(filepath: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and check it has the expected columns."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. Run `python -m src.download_data` first."
        )
    df = pd.read_csv(filepath)
    missing = [c for c in RAW_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2):
    """Stratified train / validation / test split on the raw columns.

    Returns (X_train, X_val, X_test, y_train, y_val, y_test). val_size is a
    fraction of the whole dataset.
    """
    X = df[RAW_COLS]
    y = df[TARGET_COL].astype(int)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=rel_val, stratify=y_trainval, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_scaler(X_train_raw: pd.DataFrame) -> RobustScaler:
    """Fit a RobustScaler on Time and Amount (Amount is heavily skewed)."""
    scaler = RobustScaler()
    scaler.fit(X_train_raw[SCALE_COLS].to_numpy())
    return scaler


def transform(X_raw: pd.DataFrame, scaler) -> pd.DataFrame:
    """Turn raw columns (Time, V1..V28, Amount) into model features."""
    missing = [c for c in RAW_COLS if c not in X_raw.columns]
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")
    out = X_raw[RAW_COLS].astype(float).copy()
    scaled = scaler.transform(out[SCALE_COLS].to_numpy())
    out["scaled_time"] = scaled[:, 0]
    out["scaled_amount"] = scaled[:, 1]
    return out[FEATURE_COLS]


if __name__ == "__main__":
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    print(f"Rows: {len(df):,}  |  Fraud rate: {df[TARGET_COL].mean():.4%}")
    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
