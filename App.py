import streamlit as st
import pandas as pd
import pickle
import zipfile
import os


@st.cache(allow_output_mutation=True)
def load_models():
    
    # Unzip the file
    with zipfile.ZipFile('Models.zip', 'r') as zip_ref:
        zip_ref.extractall()

    # Load the RandomForest model
    with open('Abnormal_model.pkl', 'rb') as f:
        Abnormal_model = pickle.load(f)

    # Load the XGBoost model
    with open('xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)

    return Abnormal_model, xgb_model

Abnormal_model, xgb_model = load_models()

# Streamlit App Code
st.title("ECG Classification using ML Models")

# File uploader for test data
uploaded_file = st.file_uploader("Upload CSV test file", type=["csv"])

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file, header=None)
    X_test = test_data.iloc[:, :100]

    # Radio buttons for model selection
    model_choice = st.radio("Choose a model", ('Binary', 'Abnormal'))

    # Predict button
    if st.button("Predict"):
        if model_choice == 'Binary':
            y_pred = xgb_model.predict(X_test)
        else:
            y_pred = Abnormal_model.predict(X_test)
        st.write("Predictions:", y_pred)

# About section
st.sidebar.title("About")
st.sidebar.info("""
This app allows users to classify ECG signals using pre-trained RandomForest and XGBoost models.
Upload a CSV file with test data and select a model to get predictions.
""")