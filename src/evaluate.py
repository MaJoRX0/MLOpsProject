import time
import json
import requests
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import mlflow

# Import the data loader
from data_prep import load_and_prep_data


def monitor_production_model():
    """
    Simulates monitoring a production model by sending data batches
    to a deployed API endpoint and logging performance metrics.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Model_Performance_Monitoring")

    _, X_test, _, y_test = load_and_prep_data()

    api_url = "http://127.0.0.1:5001/invocations"
    headers = {"Content-Type": "application/json"}

    print("Starting production data stream simulation...")

    with mlflow.start_run(run_name="Production_Drift_Log"):
        # Split data to simulate incoming batches
        X_batches = np.array_split(X_test, 5)
        y_batches = np.array_split(y_test, 5)

        for i, (X_batch, y_batch) in enumerate(zip(X_batches, y_batches)):
            print(f"\nProcessing Batch {i + 1}/{len(X_batches)}...")

            # Prepare data for the API request
            data = {
                "dataframe_split": {
                    "columns": X_batch.columns.tolist(),
                    "data": X_batch.values.tolist()
                }
            }

            try:
                response = requests.post(api_url, headers=headers, data=json.dumps(data))
                response.raise_for_status()  # Raise an exception for bad status codes

                predictions = response.json()["predictions"]

                # Calculate and log batch metrics
                f1 = f1_score(y_batch, predictions)
                accuracy = accuracy_score(y_batch, predictions)

                print(f"Batch {i + 1} -> F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
                mlflow.log_metrics({"production_f1_score": f1, "production_accuracy": accuracy}, step=i)

                time.sleep(1)  # Simulate time between batches

            except requests.exceptions.RequestException as e:
                print(f"[ERROR] API request failed: {e}")
                return

    print("\nMonitoring simulation complete.")


if __name__ == "__main__":
    monitor_production_model()