# Heart Disease Prediction MLOps Pipeline

This repository demonstrates a complete, end-to-end MLOps pipeline for training, tuning, and deploying a machine learning model to predict heart disease based on clinical data. The project leverages **MLflow** for experiment tracking, model registry, and serving, providing a robust framework for managing the machine learning lifecycle.

## Overview

The goal of this project is to build a classification model that accurately predicts the presence or absence of heart disease. More importantly, it serves as a practical implementation of core MLOps principles, taking a model from raw data all the way to a monitored production endpoint.

The pipeline is broken down into modular steps:
1.  **Data Preparation:** Loading, cleaning, and splitting the dataset.
2.  **Baseline Training:** Establishing a performance benchmark with a standard Random Forest classifier.
3.  **Hyperparameter Tuning:** Using `hyperopt` to search the parameter space and find the best-performing XGBoost model.
4.  **Model Registration:** Automatically selecting the best model from our experiments and registering it for production.
5.  **Deployment & Monitoring:** Serving the model as a REST API and simulating a production environment to monitor incoming data and model performance.

## Project Structure

```text
MLOpsProject/
├── data/
│   └── Heart_Disease_Prediction.csv  # The raw clinical dataset
├── src/
│   ├── data_prep.py                  # Data loading and preprocessing logic
│   ├── train.py                      # Baseline Random Forest training script
│   ├── tune.py                       # Hyperparameter tuning with XGBoost & Hyperopt
│   ├── register_model.py             # Logic to find and promote the best model
│   └── evaluate.py                   # Simulates production monitoring against the deployed model
├── requirements.txt                  # Project dependencies
└── README.md                         # You are here
```

## Setup Instructions

### 1. Environment Setup

It is highly recommended to use a virtual environment (Python 3.11+).

```bash
# Create a virtual environment
python -m venv .venv

# Activate the environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### 2. Start the MLflow Tracking Server

Before running any scripts, you need to start the local MLflow tracking server. This server stores our experiment metadata (parameters, metrics, models).

Open a **new terminal tab**, activate your environment, and run:

```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 127.0.0.1 \
    --port 5000
```
*(Leave this terminal window running in the background.)*

## Running the Pipeline

With the tracking server running, execute the following steps in your main terminal to walk through the pipeline.

### Step 1: Train the Baseline Model

Run the baseline training script. This script prepares the data, trains a Random Forest model, and logs the baseline metrics and a confusion matrix to MLflow.

```bash
python src/train.py
```

### Step 2: Hyperparameter Tuning

Next, run the tuning script to try and beat the baseline. This uses Hyperopt to find the optimal parameters for an XGBoost model, logging all trials and saving the absolute best model.

```bash
python src/tune.py
```

*Tip: You can view the results of Steps 1 and 2 by opening your browser and navigating to `http://127.0.0.1:5000` to view the MLflow UI.*

### Step 3: Register the Best Model

This script queries the MLflow server, finds the run with the highest F1-score across all our experiments, registers it as `HeartDiseaseModel`, and tags it as "Production".

```bash
python src/register_model.py
```

### Step 4: Serve the Model

Now that we have a production model, we need to serve it. Open a **third terminal tab**, activate your environment, and run the following command to start a REST API endpoint.

```bash
mlflow models serve -m "models:/HeartDiseaseModel/Production" --port 5001 --env-manager local
```
*(Leave this terminal window running in the background.)*

### Step 5: Simulate Production Monitoring

Finally, simulate a production environment. This script chunks the test dataset and sends it as sequential requests to the running model endpoint, logging the ongoing performance back to MLflow.

```bash
python src/evaluate.py
```

Check the MLflow UI one last time (`http://127.0.0.1:5000`). You will see a new experiment called `Model_Performance_Monitoring` tracking how the model handled the simulated data stream.
