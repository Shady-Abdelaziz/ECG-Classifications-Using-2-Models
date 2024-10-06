import streamlit as st
import pandas as pd
import pickle
import zipfile

# Load models
@st.cache_resource
def load_models():
    with zipfile.ZipFile('Models.zip', 'r') as zip_ref:
        zip_ref.extractall()

    with open('xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)

    with open('Abnormal_model.pkl', 'rb') as f:
        Abnormal_model = pickle.load(f)

    return xgb_model, Abnormal_model

# Load the models
xgb_model, Abnormal_model = load_models()

# Streamlit App Code
st.title("ECG Classification: Binary & Abnormal Detection")

# File uploader for test data
uploaded_file = st.file_uploader("Upload CSV test file", type=["csv"])

# Placeholder for results to allow clearing
results_placeholder = st.empty()

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file
        test_data = pd.read_csv(uploaded_file, header=None)
        num_columns = test_data.shape[1]

        if num_columns < 100:
            st.error("Uploaded file must contain at least 100 columns.")
        else:
            # Use only the first 100 columns if there are more
            X_test = test_data.iloc[:, :100]

            # First, use XGBoost to predict if the ECG is abnormal
            if st.button("Predict"):
                # Clear previous results
                results_placeholder.empty()

                # Predict using XGBoost
                binary_pred = xgb_model.predict(X_test)
                results = []

                # Loop through each prediction in the array
                for idx, pred in enumerate(binary_pred):
                    result = "normal" if pred == 0 else "abnormal"
                    results.append(result)

                    # If the prediction is abnormal, predict further using the Abnormal model
                    if result == "abnormal":
                        abnormal_pred = Abnormal_model.predict(X_test.iloc[idx].values.reshape(1, -1)) + 1
                        results_placeholder.write(f"Sample {idx}: Detailed abnormal classification prediction: {abnormal_pred[0]}")

                # Display the summary of predictions
                results_placeholder.write("Summary of predictions: " + ", ".join(results))

    except Exception as e:
        st.error(f"Error reading the file: {e}")

# About section with updated information
st.sidebar.title("About")
st.sidebar.info("""
This app first classifies ECG signals using a pre-trained XGBoost model to determine if the signal is abnormal.
If the signal is abnormal, a second model (RandomForest) provides further classification.
Upload a CSV file with test data (100 columns for features) and receive a prediction.
""")
