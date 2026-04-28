"""
SmartFault AI — Random Forest Classifier
Fast, interpretable baseline with SHAP explainability support.
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def load_data(data_dir="data/"):
    X_train = pd.read_csv(f"{data_dir}X_train.csv")
    X_test = pd.read_csv(f"{data_dir}X_test.csv")
    y_train = pd.read_csv(f"{data_dir}y_train.csv").squeeze()
    y_test = pd.read_csv(f"{data_dir}y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, X_test, y_test, output_dir="models/"):
    os.makedirs(output_dir, exist_ok=True)

    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    params = {
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": {0: 1, 1: int(scale)},
        "random_state": 42,
        "n_jobs": -1,
    }

    print("🌲 Training Random Forest (300 trees)...")
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    print(f"5-Fold CV F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # Test evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"Random Forest Results:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {roc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal','Failure'])}")

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top10 = importances.nlargest(10)
    print(f"\nTop 10 Important Features:\n{top10.to_string()}")

    # SHAP (if available)
    try:
        import shap
        print("\n📊 Computing SHAP values (sample of 500 rows)...")
        explainer = shap.TreeExplainer(model)
        shap_sample = X_test.sample(min(500, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(shap_sample)
        shap.summary_plot(shap_values[1], shap_sample, plot_type="bar", show=False)
        import matplotlib.pyplot as plt
        plt.savefig(f"{output_dir}shap_rf_importance.png", bbox_inches="tight", dpi=120)
        plt.close()
        print(f"  SHAP plot saved to {output_dir}shap_rf_importance.png")
    except ImportError:
        print("  (Install shap for explainability plots: pip install shap)")

    # Save
    with open(f"{output_dir}rf_model.pkl", "wb") as f:
        pickle.dump(model, f)

    metrics = {
        "accuracy": float(acc), "f1": float(f1), "roc_auc": float(roc),
        "cv_f1_mean": float(cv_f1.mean()), "cv_f1_std": float(cv_f1.std()),
        "params": {k: str(v) for k, v in params.items()},
    }
    with open(f"{output_dir}rf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    importances.to_csv(f"{output_dir}rf_feature_importance.csv")
    print(f"\n✅ Random Forest saved to {output_dir}rf_model.pkl")
    return model, metrics


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    train_random_forest(X_train, y_train, X_test, y_test)
