"""
SmartFault AI — LSTM Temporal Failure Prediction
Captures gradual sensor degradation over 30-step (7.5hr) windows.
Architecture: LSTM(128) → Dropout → LSTM(64) → Dense(1, sigmoid)
"""

import numpy as np
import pandas as pd
import pickle
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not found. Install with: pip install tensorflow")


SEQUENCE_LENGTH = 30  # 30 × 15min = 7.5 hours of history
FEATURE_COLS = [
    "temperature", "vibration", "pressure", "rpm",
    "temp_vib_ratio", "temp_delta", "pres_delta", "vib_delta",
    "vib_fft_peak", "temp_pressure_ratio", "vib_rpm_ratio",
    "hour_sin", "hour_cos",
]


def build_sequences(df: pd.DataFrame, feature_cols: list, seq_len: int = 30):
    """Convert tabular sensor data to 3D sequences for LSTM."""
    X_seqs, y_seqs = [], []
    for _, group in df.groupby("machine_id") if "machine_id" in df.columns else [("all", df)]:
        arr = group[feature_cols].values
        labels = group["failure_label"].values
        for i in range(seq_len, len(arr)):
            X_seqs.append(arr[i - seq_len:i])
            y_seqs.append(labels[i])
    return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)


def build_lstm_model(seq_len: int, n_features: int) -> "tf.keras.Model":
    model = Sequential([
        LSTM(128, input_shape=(seq_len, n_features), return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        Dense(32, activation="relu"),
        Dropout(0.1),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="roc_auc")]
    )
    model.summary()
    return model


def train_lstm(
    train_csv: str = "data/X_train.csv",
    train_labels: str = "data/y_train.csv",
    test_csv: str = "data/X_test.csv",
    test_labels: str = "data/y_test.csv",
    output_dir: str = "models/",
    epochs: int = 50,
    batch_size: int = 64,
):
    if not TF_AVAILABLE:
        print("❌ TensorFlow required. pip install tensorflow")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Load
    X_train_df = pd.read_csv(train_csv)
    y_train = pd.read_csv(train_labels).squeeze()
    X_test_df = pd.read_csv(test_csv)
    y_test = pd.read_csv(test_labels).squeeze()

    # Use available feature columns
    available = [c for c in FEATURE_COLS if c in X_train_df.columns]
    if not available:
        available = X_train_df.columns.tolist()

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df[available])
    X_test_scaled = scaler.transform(X_test_df[available])

    # Rebuild DataFrames with labels for sequence building
    train_df = pd.DataFrame(X_train_scaled, columns=available)
    train_df["failure_label"] = y_train.values
    test_df = pd.DataFrame(X_test_scaled, columns=available)
    test_df["failure_label"] = y_test.values

    print(f"Building sequences (window={SEQUENCE_LENGTH})...")
    X_tr, y_tr = build_sequences(train_df, available, SEQUENCE_LENGTH)
    X_te, y_te = build_sequences(test_df, available, SEQUENCE_LENGTH)
    print(f"  Train sequences: {X_tr.shape} | Test sequences: {X_te.shape}")

    # Class weights
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    class_weight = {0: 1.0, 1: float(pos_weight)}

    # Build & train
    model = build_lstm_model(SEQUENCE_LENGTH, len(available))

    callbacks = [
        EarlyStopping(monitor="val_roc_auc", patience=8, mode="max", restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-5),
        ModelCheckpoint(f"{output_dir}lstm_best.keras", monitor="val_roc_auc", mode="max", save_best_only=True),
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    y_prob = model.predict(X_te, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    roc = roc_auc_score(y_te, y_prob)

    print(f"\n{'='*50}")
    print(f"LSTM Results:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {roc:.4f}")
    print(f"\n{classification_report(y_te, y_pred, target_names=['Normal','Failure'])}")

    # Save
    model.save(f"{output_dir}lstm_model.keras")
    with open(f"{output_dir}lstm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    metrics = {
        "accuracy": float(acc), "f1": float(f1), "roc_auc": float(roc),
        "sequence_length": SEQUENCE_LENGTH, "features": available,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()}
    }
    with open(f"{output_dir}lstm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ LSTM model saved to {output_dir}")
    return model, metrics


if __name__ == "__main__":
    train_lstm()
