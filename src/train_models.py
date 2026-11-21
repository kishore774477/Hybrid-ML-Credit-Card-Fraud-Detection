"""Train the hybrid fraud detector and save models + metrics to models/.

Usage:
    python -m src.train_models

Pipeline
1. Stratified 60/20/20 train / validation / test split.
2. Scaler fitted on the training split only.
3. Isolation Forest fitted on *normal* training transactions only, so it learns
   what legitimate behaviour looks like.
4. Random Forest (class-balanced) fitted on all training transactions.
5. Hybrid weight and decision threshold tuned on the validation split to
   maximise F1.
6. All three approaches evaluated on the untouched test split.
"""
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    DATA_PATH,
    FEATURE_COLS,
    ISO_FOREST_FILE,
    METADATA_FILE,
    MODELS_DIR,
    RANDOM_FOREST_FILE,
    RANDOM_STATE,
    SCALER_FILE,
)
from src.data_preprocessing import fit_scaler, load_data, split_data, transform
from src.hybrid_predict import anomaly_scores, hybrid_score, normalise_anomaly

WEIGHT_GRID = np.round(np.arange(0.5, 0.951, 0.05), 2)  # IF always contributes
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def train_isolation_forest(X_normal: pd.DataFrame, contamination: float) -> IsolationForest:
    model = IsolationForest(
        n_estimators=200,
        max_samples=min(4096, len(X_normal)),
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_normal)
    return model


def train_random_forest(X: pd.DataFrame, y: pd.Series, n_estimators: int = 200) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def evaluate(y_true, y_pred, y_score) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def tune_hybrid(y_val, rf_prob, iso_norm):
    best = (-1.0, 0.8, 0.5)
    for w in WEIGHT_GRID:
        scores = hybrid_score(rf_prob, iso_norm, w)
        for t in THRESHOLD_GRID:
            f1 = f1_score(y_val, scores >= t, zero_division=0)
            if f1 > best[0]:
                best = (f1, float(w), float(t))
    return best


def save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  saved {path.relative_to(path.parent.parent)}")


def train(data_path=DATA_PATH, models_dir=MODELS_DIR, rf_estimators: int = 200) -> dict:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"Loading {data_path} ...")
    df = load_data(data_path)
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_data(df)
    print(f"  train={len(X_train_raw):,}  val={len(X_val_raw):,}  test={len(X_test_raw):,}  "
          f"fraud rate={y_train.mean():.4%}")

    scaler = fit_scaler(X_train_raw)
    X_train = transform(X_train_raw, scaler)
    X_val = transform(X_val_raw, scaler)
    X_test = transform(X_test_raw, scaler)

    print("Training Isolation Forest on normal transactions ...")
    fraud_rate = float(y_train.mean())
    iso = train_isolation_forest(X_train[y_train == 0], contamination=max(fraud_rate, 0.001))
    train_raw_scores = anomaly_scores(iso, X_train)
    iso_lo = float(np.percentile(train_raw_scores, 1))
    iso_hi = float(np.percentile(train_raw_scores, 99.9))

    print("Training Random Forest ...")
    rf = train_random_forest(X_train, y_train, n_estimators=rf_estimators)

    print("Tuning hybrid weight and threshold on validation set ...")
    val_rf = rf.predict_proba(X_val)[:, 1]
    val_iso = normalise_anomaly(anomaly_scores(iso, X_val), iso_lo, iso_hi)
    val_f1, weight, threshold = tune_hybrid(y_val, val_rf, val_iso)
    print(f"  weight={weight}  threshold={threshold}  val F1={val_f1:.4f}")

    print("Evaluating on test set ...")
    test_rf = rf.predict_proba(X_test)[:, 1]
    test_iso_raw = anomaly_scores(iso, X_test)
    test_iso = normalise_anomaly(test_iso_raw, iso_lo, iso_hi)
    test_hybrid = hybrid_score(test_rf, test_iso, weight)

    metrics = {
        "isolation_forest": evaluate(y_test, iso.predict(X_test) == -1, test_iso_raw),
        "random_forest": evaluate(y_test, test_rf >= 0.5, test_rf),
        "hybrid": evaluate(y_test, test_hybrid >= threshold, test_hybrid),
    }

    importances = sorted(
        zip(FEATURE_COLS, rf.feature_importances_.tolist()), key=lambda kv: kv[1], reverse=True
    )

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "n_rows": int(len(df)),
        "n_fraud": int(df["Class"].sum()),
        "split": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "hybrid_weight": weight,
        "hybrid_threshold": threshold,
        "validation_f1": float(val_f1),
        "iso_score_lo": iso_lo,
        "iso_score_hi": iso_hi,
        "feature_columns": FEATURE_COLS,
        "feature_importances": dict(importances),
        "test_metrics": metrics,
        "training_seconds": round(time.time() - t0, 1),
    }

    print("Saving artefacts ...")
    save_pickle(iso, models_dir / ISO_FOREST_FILE)
    save_pickle(rf, models_dir / RANDOM_FOREST_FILE)
    save_pickle(scaler, models_dir / SCALER_FILE)
    (models_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2))
    print(f"  saved models/{METADATA_FILE}")

    print("\nTest-set results")
    print(f"{'model':<18}{'precision':>10}{'recall':>9}{'f1':>8}{'roc_auc':>9}{'pr_auc':>8}")
    for name, m in metrics.items():
        print(f"{name:<18}{m['precision']:>10.4f}{m['recall']:>9.4f}{m['f1']:>8.4f}"
              f"{m['roc_auc']:>9.4f}{m['pr_auc']:>8.4f}")
    print(f"\nDone in {metadata['training_seconds']}s")
    return metadata


if __name__ == "__main__":
    train()
