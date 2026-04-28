import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, Optional, Tuple


class EnsemblePredictor:

    WEIGHTS = {"xgboost": 0.50, "rf": 0.30, "lstm": 0.20}
    RISK_THRESHOLDS = {"LOW": 0.30, "MEDIUM": 0.60, "HIGH": 0.80, "CRITICAL": 0.90}
    MAINTENANCE_MAP = {
        "LOW": "Monitor normally. Next scheduled check sufficient.",
        "MEDIUM": "Flag for inspection within 48 hours.",
        "HIGH": "Schedule maintenance within 6-12 hours.",
        "CRITICAL": "Immediate shutdown recommended. Failure imminent.",
    }

    def __init__(self, model_dir: str = "models/"):
        self.model_dir = model_dir
        self.models: Dict = {}
        self.scaler = None
        self.feature_cols = None
        self._load_models()

    def _load_models(self):
        # XGBoost
        xgb_path = os.path.join(self.model_dir, "xgboost_model.pkl")
        if os.path.exists(xgb_path):
            with open(xgb_path, "rb") as f:
                self.models["xgboost"] = pickle.load(f)
            print("✅ XGBoost loaded")

        # Random Forest
        rf_path = os.path.join(self.model_dir, "rf_model.pkl")
        if os.path.exists(rf_path):
            with open(rf_path, "rb") as f:
                self.models["rf"] = pickle.load(f)
            print("✅ Random Forest loaded")

        # LSTM
        lstm_path = os.path.join(self.model_dir, "lstm_model.keras")
        if os.path.exists(lstm_path):
            import tensorflow as tf
            self.models["lstm"] = tf.keras.models.load_model(lstm_path)

            scaler_path = os.path.join(self.model_dir, "lstm_scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)

            print("✅ LSTM loaded")

        # Feature columns (IMPORTANT)
        feat_path = os.path.join(self.model_dir, "..", "data", "feature_cols.pkl")
        if os.path.exists(feat_path):
            with open(feat_path, "rb") as f:
                self.feature_cols = pickle.load(f)

        if not self.models:
            raise RuntimeError("No models found. Train models first.")

        print(f"\nLoaded {len(self.models)} model(s): {list(self.models.keys())}")

    # 🔥 CORE FIX: ALIGN FEATURES
    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_cols:
            for col in self.feature_cols:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[self.feature_cols]
        return X

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = X.copy()

        # 🔥 Align BEFORE anything
        X = self._align_features(X)

        probs = []
        active_weights = {k: v for k, v in self.WEIGHTS.items() if k in self.models}
        weight_sum = sum(active_weights.values())

        for name, model in self.models.items():
            w = active_weights[name] / weight_sum

            if name == "lstm":
                if self.scaler:
                    # 🔥 CRITICAL: match scaler feature order EXACTLY
                    feature_order = list(self.scaler.feature_names_in_)

                    for col in feature_order:
                        if col not in X.columns:
                            X[col] = 0.0

                    X_lstm = X[feature_order]

                    x_scaled = self.scaler.transform(X_lstm)
                else:
                    x_scaled = X.values

                x_seq = np.expand_dims(x_scaled, axis=1)
                p = model.predict(x_seq, verbose=0).flatten()

            else:
                p = model.predict_proba(X)[:, 1]

            probs.append(w * p)

        final_prob = np.sum(probs, axis=0)
        return (final_prob >= 0.5).astype(int), final_prob

    def predict_single(
        self,
        machine_id: str,
        temperature: float,
        vibration: float,
        pressure: float,
        rpm: float,
        extra_features: Optional[Dict] = None,
    ) -> Dict:

        base = {
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "rpm": rpm,

            # basic engineered placeholders
            "temp_delta": 0.0,
            "pres_delta": 0.0,
            "vib_delta": 0.0,
            "vib_fft_peak": 0.0,
            "temp_pressure_ratio": temperature / max(pressure, 0.001),
            "vib_rpm_ratio": vibration / max(rpm, 0.001),
            "hour_sin": 0.0,
            "hour_cos": 1.0,
        }

        if extra_features:
            base.update(extra_features)

        X = pd.DataFrame([base])

        # 🔥 Align features BEFORE prediction
        X = self._align_features(X)

        _, prob = self.predict(X)
        failure_prob = float(prob[0])

        # Risk level
        risk_level = "LOW"
        for level, threshold in sorted(self.RISK_THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
            if failure_prob >= threshold:
                risk_level = level
                break

        hours_to_failure = None
        if failure_prob >= 0.3:
            hours_to_failure = round(max(1.0, (1 - failure_prob) * 24), 1)

        return {
            "machine_id": machine_id,
            "failure_probability": round(failure_prob, 4),
            "risk_level": risk_level,
            "predicted_failure_in_hours": hours_to_failure,
            "recommended_action": self.MAINTENANCE_MAP[risk_level],
            "models_used": list(self.models.keys()),
            "confidence": round(
                min(
                    1.0,
                    failure_prob * 1.1 + 0.05 if failure_prob > 0.5
                    else (1 - failure_prob) * 1.1 + 0.05
                ),
                4
            ),
        }


if __name__ == "__main__":
    predictor = EnsemblePredictor()
    result = predictor.predict_single(
        machine_id="M001",
        temperature=85,
        vibration=0.7,
        pressure=50,
        rpm=1400
    )

    print("\nSample Prediction:")
    for k, v in result.items():
        print(f"{k}: {v}")