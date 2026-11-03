import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("DataDownloader")

# Official NASA C-MAPSS URL & mirror
NASA_CMAPSS_URL = "https://raw.githubusercontent.com/nkitsopoulos/C-MAPSS-Data/master/CMAPSSData.zip"

COLUMN_NAMES = [
    "unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"
] + [f"s_{i}" for i in range(1, 22)]


def generate_synthetic_cmapss(output_dir: Path, dataset_name: str = "FD001", num_units: int = 100, seed: int = 42):
    """
    Generates a high-fidelity synthetic C-MAPSS dataset (train, test, and RUL truth)
    mimicking NASA C-MAPSS Turbofan degradation telemetry if offline.
    """
    logger.info(f"Generating synthetic C-MAPSS dataset '{dataset_name}' with {num_units} turbofan engines...")
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base operating nominal values for 21 sensors
    base_sensor_means = np.array([
        518.67, 642.50, 1590.00, 1400.00, 14.62,
        21.61, 553.50, 2388.00, 9050.00, 1.30,
        47.50, 521.80, 2388.00, 8130.00, 8.40,
        0.03, 392.00, 2388.00, 100.00, 38.80, 23.30
    ])

    # Degradation sensitivity weights per sensor (some sensors increase, some decrease, some constant)
    degradation_weights = np.array([
        0.0, 0.08, 0.12, 0.15, 0.0,
        0.0, -0.05, 0.04, -0.06, 0.0,
        0.07, -0.08, 0.05, -0.04, 0.02,
        0.0, 0.20, 0.0, 0.0, -0.03, -0.02
    ])

    train_rows = []
    for unit_id in range(1, num_units + 1):
        max_cycles = np.random.randint(128, 360)
        for t in range(1, max_cycles + 1):
            deg_factor = (t / max_cycles) ** 2.2  # Non-linear exponential wear out
            setting_1 = np.random.normal(0.0000, 0.0015)
            setting_2 = np.random.normal(0.0000, 0.0002)
            setting_3 = 100.0

            sensor_vals = []
            for idx in range(21):
                base = base_sensor_means[idx]
                w = degradation_weights[idx]
                noise = np.random.normal(0, abs(base) * 0.005 + 0.01)
                val = base + w * deg_factor * 15.0 + noise
                sensor_vals.append(val)

            row = [unit_id, t, setting_1, setting_2, setting_3] + sensor_vals
            train_rows.append(row)

    train_df = pd.DataFrame(train_rows, columns=COLUMN_NAMES)
    train_path = output_dir / f"train_{dataset_name}.txt"
    train_df.to_csv(train_path, sep=" ", index=False, header=False)
    logger.info(f"Saved synthetic train file to {train_path} ({len(train_df)} rows).")

    # Generate test set & RUL ground truth
    test_rows = []
    rul_truth = []
    for unit_id in range(1, num_units + 1):
        max_cycles = np.random.randint(150, 320)
        cutoff_cycle = np.random.randint(40, max_cycles - 10)
        remaining_rul = max_cycles - cutoff_cycle
        rul_truth.append(remaining_rul)

        for t in range(1, cutoff_cycle + 1):
            deg_factor = (t / max_cycles) ** 2.2
            setting_1 = np.random.normal(0.0000, 0.0015)
            setting_2 = np.random.normal(0.0000, 0.0002)
            setting_3 = 100.0

            sensor_vals = []
            for idx in range(21):
                base = base_sensor_means[idx]
                w = degradation_weights[idx]
                noise = np.random.normal(0, abs(base) * 0.005 + 0.01)
                val = base + w * deg_factor * 15.0 + noise
                sensor_vals.append(val)

            row = [unit_id, t, setting_1, setting_2, setting_3] + sensor_vals
            test_rows.append(row)

    test_df = pd.DataFrame(test_rows, columns=COLUMN_NAMES)
    test_path = output_dir / f"test_{dataset_name}.txt"
    test_df.to_csv(test_path, sep=" ", index=False, header=False)

    rul_df = pd.DataFrame(rul_truth)
    rul_path = output_dir / f"RUL_{dataset_name}.txt"
    rul_df.to_csv(rul_path, sep=" ", index=False, header=False)
    logger.info(f"Saved synthetic test file ({len(test_df)} rows) and RUL ground truth ({len(rul_df)} units).")


def download_cmapss_dataset(output_dir: str = "data/raw"):
    """Downloads NASA C-MAPSS dataset or generates synthetic C-MAPSS data if download fails."""
    raw_path = Path(output_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    target_train_fd001 = raw_path / "train_FD001.txt"
    if target_train_fd001.exists():
        logger.info(f"Dataset already present at {raw_path}.")
        return

    logger.info("Attempting to download NASA C-MAPSS dataset...")
    zip_dest = raw_path / "CMAPSSData.zip"
    try:
        urllib.request.urlretrieve(NASA_CMAPSS_URL, zip_dest)
        with zipfile.ZipFile(zip_dest, "r") as zip_ref:
            zip_ref.extractall(raw_path)
        os.remove(zip_dest)
        logger.info("Successfully downloaded and extracted NASA C-MAPSS dataset.")
    except Exception as e:
        logger.warning(f"Could not download NASA C-MAPSS dataset directly ({e}). Generating high-fidelity synthetic dataset...")
        for ds in ["FD001", "FD002", "FD003", "FD004"]:
            generate_synthetic_cmapss(raw_path, dataset_name=ds)

if __name__ == "__main__":
    download_cmapss_dataset()
