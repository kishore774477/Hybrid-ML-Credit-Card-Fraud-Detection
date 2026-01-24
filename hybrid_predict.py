import pickle
import numpy as np
import pandas as pd
import os

class HybridFraudDetector:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.iso_forest = self._load_model('isolation_forest.pkl')
        self.random_forest = self._load_model('random_forest.pkl')
        self.scaler = self._load_model('scaler.pkl')

    def _load_model(self, filename):
        filepath = os.path.join(self.models_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def preprocess_input(self, input_data):
        """
        Preprocesses raw input data for prediction.
        input_data: dict containing keys matching the original CSV structure 
                    (Time, V1..V28, Amount)
        """
        # Convert dictionary to DataFrame
        # We need a defined order. The original CSV had Time, V1...V28, Amount, Class
        # Our training data (X) has: V1...V28, scaled_time, scaled_amount (the order depends on drop/add sequence)
        
        # Let's verify the column order from data_preprocessing.py:
        # 1. Load CSV (Time, V1..V28, Amount, Class)
        # 2. Add scaled_time, scaled_amount
        # 3. Drop Time, Amount
        # 4. Drop Class -> This leaves V1..V28, scaled_time, scaled_amount
        
        # NOTE: Pandas preserves order. So X_train columns are V1..V28, scaled_time, scaled_amount.
        
        input_df = pd.DataFrame([input_data])
        
        # Scale Time and Amount
        to_scale = input_df[['Time', 'Amount']]
        scaled = self.scaler.transform(to_scale)
        
        input_df['scaled_time'] = scaled[:, 0]
        input_df['scaled_amount'] = scaled[:, 1]
        
        # Select features in the correct order
        feature_cols = [f'V{i}' for i in range(1, 29)] + ['scaled_time', 'scaled_amount']
        
        return input_df[feature_cols]

    def predict(self, input_data):
        """
        Predicts fraud using the hybrid approach.
        input_data: dict map of input features.
        """
        processed_data = self.preprocess_input(input_data)
        
        # Model 1: Isolation Forest (Anomaly Detection)
        # IF returns -1 for outliers (fraud) and 1 for inliers (normal)
        # We invoke decision_function to get a score if needed, or just predict
        iso_pred = self.iso_forest.predict(processed_data)[0]
        iso_fraud = 1 if iso_pred == -1 else 0
        
        # Model 2: Random Forest (Supervised)
        # RF returns 0 (normal) or 1 (fraud)
        rf_pred = self.random_forest.predict(processed_data)[0]
        rf_prob = self.random_forest.predict_proba(processed_data)[0][1] # Probability of fraud
        
        # Hybrid Logic (Simple Ensemble)
        # If both agree, high confidence. If simple OR/AND logic.
        # Let's return details.
        
        is_fraud = (iso_fraud == 1) or (rf_pred == 1) # Conservative: if either thinks it's fraud
        
        return {
            'is_fraud': bool(is_fraud),
            'isolation_forest_prediction': 'Fraud' if iso_fraud else 'Normal',
            'random_forest_prediction': 'Fraud' if rf_pred else 'Normal',
            'fraud_probability': float(rf_prob)
        }

if __name__ == "__main__":
    # Test with dummy data
    detector = HybridFraudDetector()
    dummy_input = {
        'Time': 0,
        'Amount': 100.0
    }
    for i in range(1, 29):
        dummy_input[f'V{i}'] = 0.0
        
    try:
        print("Testing prediction with dummy data...")
        result = detector.predict(dummy_input)
        print("Result:", result)
    except Exception as e:
        print(f"Error: {e}")
