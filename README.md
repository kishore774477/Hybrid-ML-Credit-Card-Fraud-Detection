# 💳 Hybrid ML Credit Card Fraud Detection

A fraud detection system that combines **Isolation Forest** (unsupervised anomaly detection) and **Random Forest** (supervised classification) into one weighted **hybrid score**, served through an interactive **Streamlit** dashboard.

Trained on the [ULB Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): 284,807 European card transactions, of which 492 (0.17%) are fraudulent.

## ✨ Features

- **Hybrid model:** `hybrid_score = w · RF_probability + (1 − w) · IF_anomaly_score`. The weight `w` and the decision threshold are tuned on a validation split to maximise F1.
- **Leak-free pipeline:** stratified 60/20/20 train/validation/test split, with the scaler fitted on training data only.
- **Isolation Forest trained on normal transactions only**, so it models legitimate behaviour.
- **Streamlit app** with four pages:
  - **Dashboard:** dataset statistics, class balance, amount distribution, fraud rate by hour
  - **Model Performance:** precision, recall, F1, ROC-AUC and PR-AUC for IF vs RF vs Hybrid, confusion matrices, feature importances
  - **Single Prediction:** enter a transaction, or load a random real normal or fraud one, and get a risk level
  - **Batch Prediction:** upload a CSV, score every row, download the results
- **Pre-trained models included**, so the prediction pages work right after cloning
- **Automated tests** on a synthetic dataset, so no download is needed

## 📂 Project Structure

```
Hybrid-ML-Credit-Card-Fraud-Detection/
├── app.py                    # Streamlit web app
├── src/
│   ├── config.py             # Paths & constants
│   ├── download_data.py      # Downloads data/creditcard.csv
│   ├── data_preprocessing.py # Load, split, scale
│   ├── train_models.py       # Train, tune, evaluate, save
│   └── hybrid_predict.py     # HybridFraudDetector (inference)
├── models/                   # Trained models + metadata.json (metrics)
├── data/                     # creditcard.csv goes here (git-ignored)
├── tests/test_pipeline.py    # pytest suite
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
# 1. Create a virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

# 2. Run the app (pre-trained models are already in models/)
streamlit run app.py
```

### Optional: get the dataset and retrain

```bash
python -m src.download_data     # ~150 MB -> data/creditcard.csv
python -m src.train_models      # retrains and rewrites models/
```

The dataset is needed for the **Dashboard** page and for the "random transaction" buttons. Training takes about a minute on a laptop.

### Run the tests

```bash
python -m pytest
```

## 📈 Results (held-out test set, 56,962 transactions)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Isolation Forest | 0.262 | 0.347 | 0.298 | 0.960 | 0.218 |
| Random Forest | **0.936** | 0.745 | 0.830 | 0.967 | **0.852** |
| **Hybrid (w = 0.8, threshold = 0.29)** | 0.832 | **0.857** | **0.844** | **0.967** | 0.829 |

The hybrid catches **84 of 98** frauds in the test set, against 73 for Random Forest alone, while keeping precision above 83%. That trade is worth making in fraud detection, where a missed fraud usually costs more than a false alarm.

PR-AUC (average precision) is the most meaningful threshold-free metric here, because accuracy is above 99.8% even for a model that never predicts fraud.

## 🔍 How It Works

1. **Preprocessing.** `V1–V28` are already PCA components. `Time` and `Amount` are scaled with a `RobustScaler` fitted on the training split.
2. **Isolation Forest.** Fitted on normal training transactions. Its anomaly score is normalised to [0, 1] using bounds learnt from the training data.
3. **Random Forest.** 200 trees with `class_weight="balanced_subsample"` to handle the 1:578 class imbalance. Outputs the probability of fraud.
4. **Hybrid.** Weighted blend of the two scores. A transaction is flagged when the score reaches the tuned threshold. Risk level is **High** at or above the threshold, **Medium** at or above half of it, and **Low** otherwise.

## 🛠 Tech Stack

Python · scikit-learn · pandas · NumPy · Streamlit · Matplotlib · Seaborn · pytest
