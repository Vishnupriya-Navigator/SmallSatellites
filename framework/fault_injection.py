import pandas as pd
import numpy as np
import os


def inject_faults(df):
    df = df.copy()
    df["fault_label"] = 0  # start with all normal

    # Inject voltage spikes every 200th row
    for i in range(200, len(df), 200):
        df.loc[i, "voltage"] += np.random.uniform(0.5, 1.0)
        df.loc[i, "fault_label"] = 1

    # Inject temperature drift in a range
    drift_start = 500
    drift_end = 550
    df.loc[drift_start:drift_end, "temperature"] += np.linspace(
        0, 10, drift_end - drift_start + 1
    )
    df.loc[drift_start:drift_end, "fault_label"] = 1

    # Inject signal dropout
    dropout_indices = np.random.choice(df.index, size=5, replace=False)
    for i in dropout_indices:
        df.loc[i, "signal_strength"] = 0
        df.loc[i, "fault_label"] = 1

    return df


def save_faulty_data(df, filename="data/simulated/telemetry_faulty.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Faulty telemetry saved to {filename}")


if __name__ == "__main__":
    # Load clean telemetry
    input_file = "data/simulated/telemetry_clean.csv"
    df = pd.read_csv(input_file)

    # Inject faults
    df_faulty = inject_faults(df)

    # Save output
    save_faulty_data(df_faulty)
