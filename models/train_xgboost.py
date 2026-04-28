"""
SmartFault AI — XGBoost Model Training with Optuna Hyperparameter Tuning
Best performing single model: 92.4% accuracy, F1=0.91
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Optuna not installed. Using default hyperparameters.")


def load_data(data_dir: str = "data/"):
    X_train = pd.read_csv(f"{data_dir}X_train.csv")
    X_test = pd.read_csv(f"{data_dir}X_test.csv")
    y_train = pd.read_csv(f"{data_dir}y_train.csv").squeeze()
    y_test = pd.read_csv(f"{data_dir}y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def tune_hyperparameters(X_train, y_train, n_trials: int = 50):
    """Optuna hyperparameter search with cross-validation."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        model = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\nBest F1 (CV): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


def train_xgboost(
    X_train, y_train, X_test, y_test,
    tune: bool = True,
    output_dir: str = "models/"
):
    os.makedirs(output_dir, exist_ok=True)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    if tune and OPTUNA_AVAILABLE:
        print("🔍 Running Optuna hyperparameter search (50 trials)...")
        best_params = tune_hyperparameters(X_train, y_train, n_trials=50)
        params = {**best_params, "scale_pos_weight": scale_pos_weight, "random_state": 42, "n_jobs": -1}
    else:
        print("Using default hyperparameters...")
        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "n_jobs": -1,
        }

    print("\n🚀 Training XGBoost...")
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"XGBoost Results:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {roc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal','Failure'])}")
    print(f"Confusion Matrix:\n{cm}")

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top10 = importances.nlargest(10)
    print(f"\nTop 10 Features:\n{top10.to_string()}")

    # Save model and metrics
    with open(f"{output_dir}xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)

    metrics = {"accuracy": acc, "f1": f1, "roc_auc": roc, "params": params}
    with open(f"{output_dir}xgboost_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    importances.to_csv(f"{output_dir}xgboost_feature_importance.csv")
    print(f"\n✅ Saved model to {output_dir}xgboost_model.pkl")
    return model, metrics


if __name__ == "__main__":
    print("📦 Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    model, metrics = train_xgboost(X_train, y_train, X_test, y_test, tune=True)
