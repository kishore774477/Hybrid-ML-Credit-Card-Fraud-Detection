"""Central paths and constants used across the project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = DATA_DIR / "creditcard.csv"

# Public mirror of the Kaggle "Credit Card Fraud Detection" dataset (ULB),
# hosted by TensorFlow for its tutorials.
DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

PCA_COLS = [f"V{i}" for i in range(1, 29)]
RAW_COLS = ["Time"] + PCA_COLS + ["Amount"]
SCALE_COLS = ["Time", "Amount"]
FEATURE_COLS = PCA_COLS + ["scaled_time", "scaled_amount"]
TARGET_COL = "Class"

RANDOM_STATE = 42

ISO_FOREST_FILE = "isolation_forest.pkl"
RANDOM_FOREST_FILE = "random_forest.pkl"
SCALER_FILE = "scaler.pkl"
METADATA_FILE = "metadata.json"
