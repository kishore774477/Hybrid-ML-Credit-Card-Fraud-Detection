"""Hybrid inference: Isolation Forest anomaly score + Random Forest probability.

hybrid_score = w * rf_probability + (1 - w) * normalised_anomaly_score
A transaction is flagged as fraud when hybrid_score >= threshold.
Both w and the threshold are tuned on a validation split during training and
stored in models/metadata.json.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ISO_FOREST_FILE,
    METADATA_FILE,
    MODELS_DIR,
    RANDOM_FOREST_FILE,
    RAW_COLS,
    SCALER_FILE,
)
from src.data_preprocessing import transform


def anomaly_scores(iso_forest, X: pd.DataFrame) -> np.ndarray:
    """Raw Isolation Forest anomaly score: higher means more unusual."""
    return -iso_forest.score_samples(X)


def normalise_anomaly(raw: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map raw anomaly scores to [0, 1] using bounds learnt on training data."""
    return np.clip((raw - lo) / (hi - lo), 0.0, 1.0)


def hybrid_score(rf_prob: np.ndarray, iso_norm: np.ndarray, weight: float) -> np.ndarray:
    return weight * rf_prob + (1.0 - weight) * iso_norm


def risk_level(score: float, threshold: float) -> str:
    if score >= threshold:
        return "High"
    if score >= threshold / 2:
        return "Medium"
    return "Low"


class HybridFraudDetector:
    def __init__(self, models_dir: Path | str = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.iso_forest = self._load_pickle(ISO_FOREST_FILE)
        self.random_forest = self._load_pickle(RANDOM_FOREST_FILE)
        self.scaler = self._load_pickle(SCALER_FILE)
        meta_path = self.models_dir / METADATA_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"Model metadata not found: {meta_path}")
        self.metadata = json.loads(meta_path.read_text())
        self.weight = float(self.metadata["hybrid_weight"])
        self.threshold = float(self.metadata["hybrid_threshold"])
        self.iso_lo = float(self.metadata["iso_score_lo"])
        self.iso_hi = float(self.metadata["iso_score_hi"])

    def _load_pickle(self, filename: str):
        path = self.models_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict_batch(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Score many transactions. raw_df needs Time, V1..V28, Amount."""
        X = transform(raw_df, self.scaler)
        rf_prob = self.random_forest.predict_proba(X)[:, 1]
        iso_raw = anomaly_scores(self.iso_forest, X)
        iso_norm = normalise_anomaly(iso_raw, self.iso_lo, self.iso_hi)
        iso_flag = self.iso_forest.predict(X) == -1
        score = hybrid_score(rf_prob, iso_norm, self.weight)

        return pd.DataFrame(
            {
                "rf_fraud_probability": rf_prob,
                "anomaly_score": iso_norm,
                "isolation_forest_flag": iso_flag,
                "hybrid_score": score,
                "is_fraud": score >= self.threshold,
                "risk_level": [risk_level(s, self.threshold) for s in score],
            },
            index=raw_df.index,
        )

    def predict(self, input_data: dict) -> dict:
        """Score one transaction given as {Time, V1..V28, Amount}."""
        missing = [c for c in RAW_COLS if c not in input_data]
        if missing:
            raise ValueError(f"Missing input fields: {missing}")
        row = self.predict_batch(pd.DataFrame([{c: input_data[c] for c in RAW_COLS}])).iloc[0]
        return {
            "is_fraud": bool(row["is_fraud"]),
            "hybrid_score": float(row["hybrid_score"]),
            "fraud_probability": float(row["rf_fraud_probability"]),
            "anomaly_score": float(row["anomaly_score"]),
            "risk_level": row["risk_level"],
            "isolation_forest_prediction": "Anomaly" if row["isolation_forest_flag"] else "Normal",
            "random_forest_prediction": "Fraud" if row["rf_fraud_probability"] >= 0.5 else "Normal",
            "threshold": self.threshold,
        }


if __name__ == "__main__":
    detector = HybridFraudDetector()
    sample = {"Time": 0.0, "Amount": 100.0, **{f"V{i}": 0.0 for i in range(1, 29)}}
    print("Prediction for an all-zero PCA transaction of $100:")
    print(detector.predict(sample))
