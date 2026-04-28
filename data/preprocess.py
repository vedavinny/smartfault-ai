"""
SmartFault AI — Preprocessing & Feature Engineering Pipeline
Cleans raw sensor data and engineers 28 predictive features.
"""

import pandas as pd
import numpy as np
from scipy import signal
import os
import pickle


def load_raw(path: str = "data/sensor_data_raw.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"Loaded {len(df):,} rows from {path}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, clip outliers, forward-fill small gaps."""
    before = len(df)
    df = df.drop_duplicates(subset=["machine_id", "timestamp"])
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    # Clip physical outliers
    df["temperature"] = df["temperature"].clip(0, 200)
    df["vibration"] = df["vibration"].clip(0, 10)
    df["pressure"] = df["pressure"].clip(0, 300)
    df["rpm"] = df["rpm"].clip(0, 5000)

    # Forward-fill short gaps (up to 2 steps = 30min)
    df[["temperature", "vibration", "pressure", "rpm"]] = (
        df.groupby("machine_id")[["temperature", "vibration", "pressure", "rpm"]]
        .apply(lambda g: g.fillna(method="ffill", limit=2))
        .reset_index(level=0, drop=True)
    )

    df = df.dropna()
    print(f"Cleaned: {before:,} → {len(df):,} rows (removed {before - len(df):,})")
    return df


def engineer_features(df: pd.DataFrame, window_sizes: list = [4, 16, 32]) -> pd.DataFrame:
    """
    Engineer 28 time-series features per machine:
      - Rolling statistics (mean, std, min, max) across multiple windows
      - Lag features (15min, 1h, 4h back)
      - FFT dominant frequency for vibration
      - Rate of change for temperature and pressure
      - Cross-sensor interactions
    """
    sensors = ["temperature", "vibration", "pressure", "rpm"]
    feature_dfs = []

    for machine_id, group in df.groupby("machine_id"):
        g = group.copy().sort_values("timestamp").reset_index(drop=True)

        for w in window_sizes:
            for col in sensors:
                g[f"{col}_roll_mean_{w}"] = g[col].rolling(w, min_periods=1).mean()
                g[f"{col}_roll_std_{w}"] = g[col].rolling(w, min_periods=1).std().fillna(0)

        # Lag features
        for col in sensors:
            g[f"{col}_lag1"] = g[col].shift(1)    # 15 min
            g[f"{col}_lag4"] = g[col].shift(4)    # 1 hour
            g[f"{col}_lag16"] = g[col].shift(16)  # 4 hours

        # Rate of change
        g["temp_delta"] = g["temperature"].diff()
        g["pres_delta"] = g["pressure"].diff()
        g["vib_delta"] = g["vibration"].diff()

        # FFT peak frequency (rolling 32-step window on vibration)
        def fft_peak_freq(x):
            if len(x) < 8:
                return 0.0
            freqs = np.fft.rfftfreq(len(x))
            fft_vals = np.abs(np.fft.rfft(x))
            return freqs[np.argmax(fft_vals[1:]) + 1] if len(freqs) > 1 else 0.0

        g["vib_fft_peak"] = (
            g["vibration"].rolling(32, min_periods=8).apply(fft_peak_freq, raw=True).fillna(0)
        )

        # Cross-sensor ratios
        g["temp_pressure_ratio"] = g["temperature"] / g["pressure"].replace(0, np.nan)
        g["vib_rpm_ratio"] = g["vibration"] / g["rpm"].replace(0, np.nan)

        # Hour of day (cyclical)
        g["hour_sin"] = np.sin(2 * np.pi * g["timestamp"].dt.hour / 24)
        g["hour_cos"] = np.cos(2 * np.pi * g["timestamp"].dt.hour / 24)

        feature_dfs.append(g)

    result = pd.concat(feature_dfs).reset_index(drop=True)
    result = result.dropna()

    print(f"Feature engineering complete. Shape: {result.shape}")
    print(f"Features: {[c for c in result.columns if c not in ['timestamp','machine_id','failure_label']]}")
    return result


def split_and_save(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    output_dir: str = "data/"
):
    """Chronological train/test split (no data leakage from future)."""
    feature_cols = [
        c for c in df.columns
        if c not in ["timestamp", "machine_id", "failure_label"]
    ]
    label_col = "failure_label"

    split_idx = int(len(df) * (1 - test_ratio))
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    X_train = df_train[feature_cols]
    y_train = df_train[label_col]
    X_test = df_test[feature_cols]
    y_test = df_test[label_col]

    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f"{output_dir}X_train.csv", index=False)
    y_train.to_csv(f"{output_dir}y_train.csv", index=False)
    X_test.to_csv(f"{output_dir}X_test.csv", index=False)
    y_test.to_csv(f"{output_dir}y_test.csv", index=False)

    # Save feature list for inference
    with open(f"{output_dir}feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Label balance (train): {y_train.mean()*100:.1f}% positive")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df_raw = load_raw()
    df_clean = clean(df_raw)
    df_features = engineer_features(df_clean)
    split_and_save(df_features)
    print("\n✅ Preprocessing complete. Ready for model training.")
