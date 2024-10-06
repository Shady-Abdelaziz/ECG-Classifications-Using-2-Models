import streamlit as st
import pandas as pd
import pickle
import zipfile
import os

# Load models
@st.cache_resource
def load_models():
    # Unzip the file
    with zipfile.ZipFile('Models.zip', 'r') as zip_ref:
        zip_ref.extractall()

    # Load the XGBoost model for binary classification
    with open('xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)

    # Load the Abnormal model for abnormal classification
    with open('Abnormal_model.pkl', 'rb') as f:
        Abnormal_model = pickle.load(f)

    return xgb_model, Abnormal_model

# Load the models
xgb_model, Abnormal_model = load_models()

# Streamlit App Code
st.title("ECG Classification: Binary & Abnormal Detection")

# File uploader for test data
uploaded_file = st.file_uploader("Upload CSV test file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    test_data = pd.read_csv(uploaded_file, header=None)
    X_test = test_data.iloc[:, :100]

    # First, use XGBoost to predict if the ECG is abnormal
    if st.button("Predict"):
        binary_pred = xgb_model.predict(X_test)
        
        # Display the result of the XGBoost binary classification
        if binary_pred == 0:
            st.write("The ECG is classified as **normal**.")
        else:
            st.write("The ECG is classified as **abnormal**.")
            
            # If abnormal, further classify using the Abnormal model
            abnormal_pred = Abnormal_model.predict(X_test)+1
            st.write("Detailed abnormal classification prediction:", abnormal_pred)

# About section with updated information
st.sidebar.title("About")
st.sidebar.info("""
This app first classifies ECG signals using a pre-trained XGBoost model to determine if the signal is abnormal.
If the signal is abnormal, a second model (RandomForest) provides further classification.
Upload a CSV file with test data (100 columns for features) and receive a prediction.
""")
