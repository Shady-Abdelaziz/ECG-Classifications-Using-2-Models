# ECG Heartbeat Categorization Model Using Two Models

## Overview

This repository contains a machine learning model designed to categorize ECG (Electrocardiogram) heartbeat data in relation to Executive Core Qualifications (ECQs). The model utilizes data from an Employee database to perform classification and aggregation, aiding organizations in identifying and evaluating core competencies through heartbeat analysis.

## Project Structure

- **`data/`**: Contains the heartbeat dataset. [Kaggle Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
- **`Colab/`**: [Jupyter notebooks](https://colab.research.google.com/drive/1Of8TNBl7Z0vlubzs6yhexJhtO-pCHB2R?usp=sharing) that detail data processing, model training, and evaluation steps.
- **`requirements.txt`**: Lists the required Python packages to run the code.
- **`Models.zip`**: Contains two models:
  - **XGBoost Model**: For binary classification (0: Normal, 1: Abnormal).
  - **Abnormal Model**: Further classifies abnormal data into four categories (1, 2, 3, 4).

## Getting Started

### Prerequisites

To run the code in this repository, ensure you have the following installed:

- Python 3.x
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Shady-Abdelaziz/ECG-Classifications-Using-2-Models.git
   cd ECG-Classifications-Using-2-Models

### Deployment

The ECG Heartbeat Categorization model is deployed as a web application using Streamlit. You can access the deployed app at the following link:

[ECG Heartbeat Categorization Streamlit App
](https://ecg-classifications-using-2-models-vm8tfwbvgcddrrjpqncg5p.streamlit.app/)

## Usage

After deploying the application, follow the on-screen instructions to upload ECG data and obtain categorizations based on the trained models.
Contributing

