"""End-to-end smoke tests on a small synthetic dataset (no download needed).

Run with:  python -m pytest
"""
import numpy as np
import pandas as pd
import pytest

from src.config import PCA_COLS, RAW_COLS
from src.data_preprocessing import fit_scaler, split_data, transform
from src.hybrid_predict import HybridFraudDetector
from src.train_models import train


def make_synthetic(n_normal=3000, n_fraud=60, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    normal = pd.DataFrame(rng.normal(0, 1, (n_normal, 28)), columns=PCA_COLS)
    fraud = pd.DataFrame(rng.normal(3, 1.5, (n_fraud, 28)), columns=PCA_COLS)
    normal["Amount"] = rng.gamma(2, 40, n_normal)
    fraud["Amount"] = rng.gamma(1, 300, n_fraud)
    normal["Class"], fraud["Class"] = 0, 1
    df = pd.concat([normal, fraud], ignore_index=True)
    df["Time"] = rng.uniform(0, 172800, len(df))
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    data_path = root / "synthetic.csv"
    make_synthetic().to_csv(data_path, index=False)
    models_dir = root / "models"
    meta = train(data_path=data_path, models_dir=models_dir, rf_estimators=30)
    return meta, HybridFraudDetector(models_dir)


def test_split_is_stratified():
    df = make_synthetic()
    *_, y_train, y_val, y_test = split_data(df)
    rates = [y.mean() for y in (y_train, y_val, y_test)]
    assert max(rates) - min(rates) < 0.01


def test_transform_columns_and_order():
    df = make_synthetic()
    X = transform(df, fit_scaler(df))
    assert list(X.columns) == PCA_COLS + ["scaled_time", "scaled_amount"]
    assert not X.isna().any().any()


def test_training_writes_metadata(trained):
    meta, _ = trained
    assert 0.5 <= meta["hybrid_weight"] <= 0.95
    assert 0 < meta["hybrid_threshold"] < 1
    assert set(meta["test_metrics"]) == {"isolation_forest", "random_forest", "hybrid"}
    assert meta["test_metrics"]["hybrid"]["roc_auc"] > 0.9


def test_single_prediction(trained):
    _, detector = trained
    normal = {"Time": 1000.0, "Amount": 50.0, **{c: 0.0 for c in PCA_COLS}}
    fraud = {"Time": 1000.0, "Amount": 500.0, **{c: 4.0 for c in PCA_COLS}}
    r_normal, r_fraud = detector.predict(normal), detector.predict(fraud)
    assert r_normal["is_fraud"] is False
    assert r_fraud["is_fraud"] is True
    assert r_fraud["hybrid_score"] > r_normal["hybrid_score"]


def test_batch_prediction(trained):
    _, detector = trained
    df = make_synthetic(200, 20, seed=1)
    out = detector.predict_batch(df[RAW_COLS])
    assert len(out) == len(df)
    assert out["hybrid_score"].between(0, 1).all()
    caught = (out["is_fraud"] & (df["Class"] == 1)).sum()
    assert caught >= 15


def test_missing_field_raises(trained):
    _, detector = trained
    with pytest.raises(ValueError):
        detector.predict({"Time": 0.0, "Amount": 1.0})
