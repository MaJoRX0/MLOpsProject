import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.sklearn

from data_prep import load_and_prep_data

def train_baseline_model():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Heart_Disease_Classification")

    X_train, X_test, y_train, y_test = load_and_prep_data()

    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
    }

    with mlflow.start_run(run_name="Baseline_RandomForest"):
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }

        print("\n--- Baseline Model Metrics ---")
        for k, v in metrics.items():
            print(f"{k.capitalize()}: {v:.4f}")

        mlflow.log_metrics(metrics)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Absence", "Presence"])
        disp.plot(cmap=plt.cm.Blues)

        plot_path = "confusion_matrix.png"
        plt.savefig(plot_path)
        plt.close()

        mlflow.log_artifact(plot_path)

        if os.path.exists(plot_path):
            os.remove(plot_path)

        mlflow.sklearn.log_model(model, artifact_path="model")

        print("\nRun tracked and assets logged to MLflow.")

if __name__ == "__main__":
    train_baseline_model()