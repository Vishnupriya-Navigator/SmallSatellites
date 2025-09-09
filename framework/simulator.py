import numpy as np
import pandas as pd
import datetime

# Set random seed for reproducibility
np.random.seed(42)
# Simulation settings
NUM_ROWS = 1000
START_TIME = datetime.datetime.now()


def generate_telemetry():
    timestamps = [START_TIME + datetime.timedelta(seconds=i) for i in range(NUM_ROWS)]

    voltage = np.random.normal(loc=3.7, scale=0.05, size=NUM_ROWS)  # volts
    temperature = np.random.normal(loc=25.0, scale=1.0, size=NUM_ROWS)  # Celsius
    cpu_load = np.random.normal(loc=60, scale=10, size=NUM_ROWS)  # percent
    signal_strength = np.random.normal(loc=80, scale=5, size=NUM_ROWS)  # dBm

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "voltage": voltage,
            "temperature": temperature,
            "cpu_load": cpu_load,
            "signal_strength": signal_strength,
        }
    )

    return df


def save_telemetry(df, filename="data/simulated/telemetry_clean.csv"):
    # Make sure the folder exists
    import os

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    df.to_csv(filename, index=False)
    print(f" Telemetry saved to {filename}")


if __name__ == "__main__":
    telemetry_df = generate_telemetry()
    save_telemetry(telemetry_df)
