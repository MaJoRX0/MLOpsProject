import mlflow
from mlflow.tracking import MlflowClient


def register_best_model():
    server_uri = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(server_uri)
    client = MlflowClient(tracking_uri=server_uri)

    experiment_name = "Fraud_Hyper"
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    print(f"Searching experiment '{experiment_name}' for the best model...")

    # 1. Fetch ALL runs for this experiment (ignoring MLflow's restricted query language)
    all_runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    # 2. Filter and score the runs using native Python
    valid_runs = []
    for run in all_runs:
        run_name = run.data.tags.get("mlflow.runName", "")

        # Only look at our main parent runs
        if run_name in ["Hyperopt_XGBoost_Tuning", "Baseline_RandomForest"]:

            # Extract the correct metric based on the script that created it
            if run_name == "Hyperopt_XGBoost_Tuning":
                score = run.data.metrics.get("champion_f1_score", 0.0)
                model_path = "best_model"
            else:
                score = run.data.metrics.get("f1_score", 0.0)
                model_path = "model"

            valid_runs.append({
                "run_id": run.info.run_id,
                "run_name": run_name,
                "score": score,
                "model_path": model_path
            })

    if not valid_runs:
        print("No valid main runs found. Did you run train.py or tune.py?")
        return

    # 3. Sort by our extracted score (Highest first)
    valid_runs.sort(key=lambda x: x["score"], reverse=True)
    best_candidate = valid_runs[0]

    print(f"\nFound best run! Run Name: {best_candidate['run_name']}")
    print(f"Run ID: {best_candidate['run_id']}")
    print(f"Best F1-Score: {best_candidate['score']:.4f}")

    # 4. Formally register the model
    model_name = "FraudModel"
    model_uri = f"runs:/{best_candidate['run_id']}/{best_candidate['model_path']}"

    print(f"\nRegistering model under the name '{model_name}' from URI: {model_uri}...")
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

    version = model_details.version
    print(f"Successfully registered model! Version assigned: {version}")

    # 5. Promote to Production
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="stage",
        value="Production"
    )

    print(f"\nPhase 4 Complete! '{model_name}' Version {version} is now in Production.")


if __name__ == "__main__":
    register_best_model()