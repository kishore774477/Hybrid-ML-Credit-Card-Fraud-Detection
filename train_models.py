import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
from data_preprocessing import load_and_preprocess_data, split_data

def train_isolation_forest(X_train):
    """
    Trains Isolation Forest (Unsupervised)
    """
    print("Training Isolation Forest...")
    # Contamination is the proportion of outliers in the data set
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)
    return iso_forest

def train_random_forest(X_train, y_train):
    """
    Trains Random Forest (Supervised)
    """
    print("Training Random Forest...")
    # Using class_weight='balanced' to handle class imbalance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

def save_model(model, filename):
    os.makedirs('models', exist_ok=True)
    filepath = os.path.join('models', filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

if __name__ == "__main__":
    try:
        data_path = 'data/creditcard.csv'
        print("Loading data...")
        df = load_and_preprocess_data(data_path)
        X_train, X_test, y_train, y_test = split_data(df)

        # Train Isolation Forest
        # Note: Isolation Forest is trained on X_train (unsupervised), but it's good to remove known frauds for training 'normal' behavior if we treat it as strictly anomaly detection on normal data. 
        # However, for standard usage on mixed data, we fit on the full train set.
        iso_forest = train_isolation_forest(X_train)
        save_model(iso_forest, 'isolation_forest.pkl')

        # Train Random Forest
        rf = train_random_forest(X_train, y_train)
        save_model(rf, 'random_forest.pkl')
        
        # Simple Evaluation
        print("\nEvaluating Random Forest on Test Set:")
        y_pred_rf = rf.predict(X_test)
        print(classification_report(y_test, y_pred_rf))

    except Exception as e:
        print(f"Error during training: {e}")
