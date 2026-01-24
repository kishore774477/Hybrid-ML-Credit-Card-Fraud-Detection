import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

def load_and_preprocess_data(filepath):
    """
    Loads credit card data, normalizes 'Time' and 'Amount',
    and splits into train/test sets.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)

    # Normalize Time and Amount
    # We fit the scaler on both columns together so we can save a single object
    scaler = StandardScaler()
    df[['scaled_time', 'scaled_amount']] = scaler.fit_transform(df[['Time', 'Amount']])
    
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    # Save scaler for future use
    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return df

def split_data(df):
    """
    Splits data into X and y, then train and test sets.
    """
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data_path = 'data/creditcard.csv'
    try:
        print("Loading and preprocessing data...")
        df = load_and_preprocess_data(data_path)
        X_train, X_test, y_train, y_test = split_data(df)
        
        # Save processed data for training to avoid re-processing (optional, but good for larger workflows)
        # For now, we'll just print success
        print("Data preprocessing complete.")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
