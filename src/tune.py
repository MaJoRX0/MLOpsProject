import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score
import mlflow
import mlflow.xgboost
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from data_prep import load_and_prep_data

X_train, X_test, y_train, y_test = load_and_prep_data()


def objective(space):
    params = {
        'n_estimators': int(space['n_estimators']),
        'max_depth': int(space['max_depth']),
        'learning_rate': space['learning_rate'],
        'subsample': space['subsample'],
        'random_state': 42,
        'eval_metric': 'logloss'
    }

    with mlflow.start_run(run_name=f"Trial_{params['n_estimators']}_{params['max_depth']}", nested=True):
        mlflow.log_params(params)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metrics({"f1_score": f1, "accuracy": acc})

        return {'loss': 1 - f1, 'status': STATUS_OK}


def tune_hyperparameters():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Heart_Disease_Classification")

    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 250, 10),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'subsample': hp.uniform('subsample', 0.6, 1.0)
    }

    with mlflow.start_run(run_name="Hyperopt_XGBoost_Tuning"):
        print("Starting hyperparameter tuning...")
        trials = Trials()
        
        best_hyperparams = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=15,
            trials=trials
        )

        best_params = {
            "n_estimators": int(best_hyperparams['n_estimators']),
            "max_depth": int(best_hyperparams['max_depth']),
            "learning_rate": float(best_hyperparams['learning_rate']),
            "subsample": float(best_hyperparams['subsample'])
        }

        print(f"\nBest hyperparameters: {best_params}")

        # Train final model with best params
        best_model = XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)
        final_f1 = f1_score(y_test, y_pred)
        final_acc = accuracy_score(y_test, y_pred)

        mlflow.log_params(best_params)
        mlflow.log_metrics({"champion_f1_score": final_f1, "champion_accuracy": final_acc})

        mlflow.xgboost.log_model(best_model, artifact_path="best_model")

        print(f"\nTuning complete. Champion model F1: {final_f1:.4f}")


if __name__ == "__main__":
    tune_hyperparameters()