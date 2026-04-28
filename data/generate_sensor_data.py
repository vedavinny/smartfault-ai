"""
SmartFault AI — Synthetic Sensor Data Generator
Generates realistic IoT sensor data with injected failure patterns.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_sensor_data(
    n_machines: int = 10,
    days: int = 90,
    samples_per_hour: int = 4,
    failure_rate: float = 0.08,
    seed: int = 42,
    output_path: str = "data/sensor_data_raw.csv"
):
    """
    Generate synthetic multi-sensor time-series data for N machines over D days.
    
    Sensor channels:
        - temperature (°C): Normal ~65-75°C, spikes before failure
        - vibration (g): Normal ~0.2-0.4g, increases with wear
        - pressure (bar): Normal ~100-115 bar, drops before failure
        - rpm: Motor speed, irregular near failure
    
    Failure injection:
        - ~8% of time windows are labeled as pre-failure (next 8h)
        - Gradual sensor degradation pattern ~12h before failure
    """
    np.random.seed(seed)
    records = []

    total_steps = days * 24 * samples_per_hour
    time_index = [
        datetime(2024, 1, 1) + timedelta(minutes=15 * i)
        for i in range(total_steps)
    ]

    for machine_id in range(1, n_machines + 1):
        # Machine baseline variation
        temp_base = np.random.uniform(62, 72)
        vib_base = np.random.uniform(0.18, 0.35)
        pres_base = np.random.uniform(102, 112)
        rpm_base = np.random.uniform(1380, 1500)

        # Generate failure event indices
        n_failures = max(1, int(total_steps * failure_rate / 32))  # ~8h windows
        failure_steps = sorted(
            np.random.choice(range(200, total_steps - 50), n_failures, replace=False)
        )
        failure_set = set()
        pre_failure_set = set()
        for fs in failure_steps:
            failure_set.add(fs)
            for offset in range(-48, 0):  # 12h pre-failure window (48 × 15min)
                if fs + offset >= 0:
                    pre_failure_set.add(fs + offset)

        for step, ts in enumerate(time_index):
            is_failure = step in failure_set
            is_pre_failure = step in pre_failure_set

            # Distance to nearest failure for degradation ramp
            min_dist = min(
                (abs(step - fs) for fs in failure_steps), default=9999
            )
            ramp = max(0, 1 - min_dist / 48) ** 2  # quadratic ramp

            # Sensor readings with realistic noise
            temperature = (
                temp_base
                + np.random.normal(0, 1.2)
                + ramp * np.random.uniform(18, 30)   # temp spike
                + np.sin(step / (4 * 24)) * 3        # diurnal cycle
            )
            vibration = (
                vib_base
                + np.random.normal(0, 0.03)
                + ramp * np.random.uniform(0.4, 0.9)
            )
            pressure = (
                pres_base
                + np.random.normal(0, 2.0)
                - ramp * np.random.uniform(10, 25)   # pressure drop
            )
            rpm = (
                rpm_base
                + np.random.normal(0, 15)
                + ramp * np.random.uniform(-200, 50)
            )

            # Derived features
            temp_vib_ratio = temperature / max(vibration, 0.01)

            label = 1 if (is_failure or is_pre_failure) else 0

            records.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "machine_id": f"MACHINE_{machine_id:03d}",
                "temperature": round(temperature, 3),
                "vibration": round(vibration, 4),
                "pressure": round(pressure, 3),
                "rpm": round(rpm, 1),
                "temp_vib_ratio": round(temp_vib_ratio, 3),
                "failure_label": label,
            })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    total = len(df)
    failures = df["failure_label"].sum()
    print(f"✅ Generated {total:,} records for {n_machines} machines over {days} days")
    print(f"   Failure events: {failures:,} ({failures/total*100:.1f}%)")
    print(f"   Saved to: {output_path}")
    return df


if __name__ == "__main__":
    df = generate_sensor_data()
    print(df.head())
    print(df.describe())
