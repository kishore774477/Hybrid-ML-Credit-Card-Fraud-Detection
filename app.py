import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.hybrid_predict import HybridFraudDetector
import os

# Page config
st.set_page_config(page_title="Hybrid Credit Card Fraud Detection", layout="wide")

# Title and Description
st.title("💳 Hybrid Credit Card Fraud Detection System")
st.markdown("""
This application uses a detailed Hybrid Machine Learning approach combining **Isolation Forest** (Anomaly Detection) 
and **Random Forest** (Supervised Classification) to detect fraudulent credit card transactions.
""")

# Sidebar
st.sidebar.header("Navigation")
options = st.sidebar.radio("Go to", ["Dashboard", "Fraud Prediction"])

@st.cache_data
def load_data():
    if os.path.exists('data/creditcard.csv'):
        return pd.read_csv('data/creditcard.csv')
    return None

def load_detector():
    try:
        return HybridFraudDetector()
    except FileNotFoundError:
        return None

if options == "Dashboard":
    st.header("📊 Dataset Overview")
    df = load_data()
    
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Transactions", len(df))
        with col2:
            fraud_count = df[df['Class'] == 1].shape[0]
            st.metric("Fraudulent Transactions", fraud_count)
            
        st.subheader("Data Sample")
        st.dataframe(df.head())
        
        st.subheader("Class Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='Class', data=df, ax=ax)
        plt.title('Class Distribution (0: Normal, 1: Fraud)')
        st.pyplot(fig)
        
        st.subheader("Transaction Amount Distribution (Fraud vs Normal)")
        fig, ax = plt.subplots(figsize=(10, 6))
        # Filter 0 or negative values for log scale
        plot_df = df[df['Amount'] > 0]
        sns.histplot(data=plot_df, x='Amount', hue='Class', kde=True, log_scale=True, ax=ax)
        plt.title("Log-Scaled Transaction Amount Distribution")
        st.pyplot(fig)
        
    else:
        st.error("Data file not found in `data/creditcard.csv`. Please ensure the file exists.")

elif options == "Fraud Prediction":
    st.header("🔍 Predict Fraud")
    
    detector = load_detector()
    
    if detector:
        st.info("Enter transaction details below to check for fraud.")
        
        # Input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                time_val = st.number_input("Time (Seconds since first transaction)", min_value=0.0, value=0.0)
                amount_val = st.number_input("Amount ($)", min_value=0.0, value=100.0)
            
            with col2:
                st.markdown("**PCA Feature Inputs (V1-V28)**")
                # Create inputs for V1-V28
                v_features = {}
                # Using an expander to hide the 28 inputs
                with st.expander("Enter V1-V28 Features (PCA Components)", expanded=False):
                    for i in range(1, 29):
                        v_features[f'V{i}'] = st.number_input(f"V{i}", value=0.0)
            
            submit_button = st.form_submit_button("Predict")
            
        if submit_button:
            # Construct input dictionary
            input_data = {
                'Time': time_val,
                'Amount': amount_val
            }
            input_data.update(v_features)
            
            # Predict
            try:
                result = detector.predict(input_data)
                
                st.subheader("Prediction Result")
                
                if result['is_fraud']:
                    st.error(f"🚨 FRAUD DETECTED! (Probability: {result['fraud_probability']:.2f})")
                else:
                    st.success(f"✅ Normal Transaction (Probability of Fraud: {result['fraud_probability']:.2f})")
                
                st.write("### Model Insights")
                c1, c2 = st.columns(2)
                c1.metric("Isolation Forest (Anomaly)", result['isolation_forest_prediction'])
                c2.metric("Random Forest (Classifier)", result['random_forest_prediction'])
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                
    else:
        st.warning("Models not found! Please train the models first by running `src/train_models.py`.")
