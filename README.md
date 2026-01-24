# Hybrid Credit Card Fraud Detection

A comprehensive fraud detection system using a hybrid approach with **Isolation Forest** (Unsupervised Anomaly Detection) and **Random Forest** (Supervised Classification).

## 📂 Project Structure
```
Hybrid-Credit-Card-Fraud-Detection/
│
├── data/
│   └── creditcard.csv       # Dataset (transactions)
│
├── models/                  # Saved models (generated after training)
│   ├── isolation_forest.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
│
├── src/
│   ├── data_preprocessing.py  # Data loading and scaling
│   ├── train_models.py        # Model training script
│   ├── hybrid_predict.py      # Inference logic
│
├── app.py                   # Streamlit Web App
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

## 🚀 Setup & Installation

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Dataset**
    Ensure `creditcard.csv` is placed inside the `data/` folder.

## 🧠 Training the Models

Before running the app, you must train the models. This process handles data preprocessing, model training, and saving the artifacts.

```bash
python src/train_models.py
```
*This will create/populate the `models/` directory.*

## 💻 Running the Application

Launch the Streamlit dashboard to explore data and make predictions.

```bash
python -m streamlit run app.py
```

## 🔍 How it Works
1.  **Isolation Forest**: Detects anomalies based on how "different" a transaction is from the norm.
2.  **Random Forest**: Classifies transactions based on learned patterns from labeled fraud data.
3.  **Hybrid Logic**: Combines insights from both models to provide a robust fraud probability score.
