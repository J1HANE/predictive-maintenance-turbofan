import os
from pathlib import Path
import datetime
import pandas as pd
import numpy as np
import yaml
from src.utils.logger import get_logger

logger = get_logger("DataLake")

class MedallionDataLake:
    """
    Implements a Medallion Architecture Data Lake (Bronze -> Silver -> Gold)
    for Turbofan Engine Telemetry Data.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.raw_dir = Path(self.config["paths"]["raw_data"])
        self.bronze_dir = Path(self.config["paths"]["bronze_data"])
        self.silver_dir = Path(self.config["paths"]["silver_data"])
        self.gold_dir = Path(self.config["paths"]["gold_data"])

        for d in [self.bronze_dir, self.silver_dir, self.gold_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.column_names = [
            "unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"
        ] + [f"s_{i}" for i in range(1, 22)]

    def run_bronze_ingestion(self, dataset_name: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Bronze Layer: Ingests raw text logs, tags metadata (ingestion_timestamp, source),
        and persists in Parquet format.
        """
        logger.info(f"[BRONZE] Ingesting raw dataset '{dataset_name}'...")
        train_file = self.raw_dir / f"train_{dataset_name}.txt"
        test_file = self.raw_dir / f"test_{dataset_name}.txt"
        rul_file = self.raw_dir / f"RUL_{dataset_name}.txt"

        if not train_file.exists():
            raise FileNotFoundError(f"Raw data file {train_file} not found. Please run download_data.py first.")

        # Read train data
        df_train = pd.read_csv(train_file, sep=r"\s+", header=None, names=self.column_names)
        df_train["ingestion_timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        df_train["dataset"] = dataset_name
        df_train["split"] = "train"

        # Read test data
        df_test = pd.read_csv(test_file, sep=r"\s+", header=None, names=self.column_names)
        df_test["ingestion_timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        df_test["dataset"] = dataset_name
        df_test["split"] = "test"

        # Read RUL ground truth for test
        df_rul = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["rul_truth"])
        df_rul["unit_nr"] = range(1, len(df_rul) + 1)
        df_rul["dataset"] = dataset_name

        # Persist Bronze Parquet
        train_bronze_path = self.bronze_dir / f"bronze_train_{dataset_name}.parquet"
        test_bronze_path = self.bronze_dir / f"bronze_test_{dataset_name}.parquet"
        rul_bronze_path = self.bronze_dir / f"bronze_rul_{dataset_name}.parquet"

        df_train.to_parquet(train_bronze_path, index=False)
        df_test.to_parquet(test_bronze_path, index=False)
        df_rul.to_parquet(rul_bronze_path, index=False)

        logger.info(f"[BRONZE] Saved Parquet to Bronze lake: train ({len(df_train)} rows), test ({len(df_test)} rows).")
        return df_train, df_test

    def run_silver_processing(self, dataset_name: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Silver Layer: Cleans telemetry data, removes constant non-informative sensors,
        imputes outliers, and structures sensor measurements.
        """
        logger.info(f"[SILVER] Processing dataset '{dataset_name}'...")
        df_train = pd.read_parquet(self.bronze_dir / f"bronze_train_{dataset_name}.parquet")
        df_test = pd.read_parquet(self.bronze_dir / f"bronze_test_{dataset_name}.parquet")

        # Compute max cycles per train engine to derive ground truth train RUL
        max_cycles_train = df_train.groupby("unit_nr")["time_cycles"].max().reset_index()
        max_cycles_train.rename(columns={"time_cycles": "max_cycles"}, inplace=True)
        df_train = df_train.merge(max_cycles_train, on="unit_nr", how="left")
        df_train["RUL"] = df_train["max_cycles"] - df_train["time_cycles"]

        # For test data, merge with ground truth RUL file
        df_rul = pd.read_parquet(self.bronze_dir / f"bronze_rul_{dataset_name}.parquet")
        max_cycles_test = df_test.groupby("unit_nr")["time_cycles"].max().reset_index()
        max_cycles_test.rename(columns={"time_cycles": "max_observed_cycles"}, inplace=True)
        df_test = df_test.merge(max_cycles_test, on="unit_nr", how="left")
        df_test = df_test.merge(df_rul, on="unit_nr", how="left")
        # Exact RUL for each test cycle
        df_test["RUL"] = (df_test["max_observed_cycles"] + df_test["rul_truth"]) - df_test["time_cycles"]

        # Drop constant/flat sensors with std close to 0 across dataset
        sensor_cols = self.config["data"]["sensor_cols"]
        stds = df_train[sensor_cols].std()
        informative_sensors = stds[stds > 1e-4].index.tolist()
        logger.info(f"[SILVER] Retained {len(informative_sensors)} informative sensors out of {len(sensor_cols)}.")

        # Event flag for Survival Analysis (1 if unit reaches end of life / max cycle, else 0)
        df_train["event_occurred"] = (df_train["RUL"] == 0).astype(int)
        # For right-censoring in test set: last observed cycle is censored or failure event
        df_test["event_occurred"] = (df_test["time_cycles"] == df_test["max_observed_cycles"]).astype(int)

        # Save to Silver Layer
        silver_train_path = self.silver_dir / f"silver_train_{dataset_name}.parquet"
        silver_test_path = self.silver_dir / f"silver_test_{dataset_name}.parquet"

        df_train.to_parquet(silver_train_path, index=False)
        df_test.to_parquet(silver_test_path, index=False)

        logger.info(f"[SILVER] Saved Silver data to {silver_train_path} and {silver_test_path}.")
        return df_train, df_test

    def run_gold_enrichment(self, dataset_name: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Gold Layer: Builds temporal rolling features, degradation indices,
        piecewise linear target RUL, and prepares ML-ready datasets.
        """
        logger.info(f"[GOLD] Enriching features for dataset '{dataset_name}'...")
        df_train = pd.read_parquet(self.silver_dir / f"silver_train_{dataset_name}.parquet")
        df_test = pd.read_parquet(self.silver_dir / f"silver_test_{dataset_name}.parquet")

        active_sensors = [col for col in self.config["data"]["active_sensors"] if col in df_train.columns]
        windows = self.config["features"]["rolling_windows"]
        piecewise_max = self.config["features"]["piecewise_rul_max"]

        def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            # Piecewise Linear RUL capping for early life (as recommended by NASA C-MAPSS literature)
            df["RUL_piecewise"] = df["RUL"].apply(lambda x: min(x, piecewise_max))

            # Compute rolling features grouped by unit_nr
            feature_dfs = [df]
            for window in windows:
                rolling_mean = df.groupby("unit_nr")[active_sensors].transform(lambda x: x.rolling(window, min_periods=1).mean())
                rolling_mean.columns = [f"{col}_roll_mean_{window}" for col in active_sensors]

                rolling_std = df.groupby("unit_nr")[active_sensors].transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
                rolling_std.columns = [f"{col}_roll_std_{window}" for col in active_sensors]

                feature_dfs.extend([rolling_mean, rolling_std])

            enriched = pd.concat(feature_dfs, axis=1)

            # Health Index (Composite Degradation Score using first 3 active sensors normalized)
            s_norm = (enriched[active_sensors[0]] - enriched[active_sensors[0]].min()) / (enriched[active_sensors[0]].max() - enriched[active_sensors[0]].min() + 1e-6)
            enriched["health_index"] = 1.0 - s_norm
            return enriched

        gold_train = enrich_df(df_train)
        gold_test = enrich_df(df_test)

        gold_train_path = self.gold_dir / f"gold_train_{dataset_name}.parquet"
        gold_test_path = self.gold_dir / f"gold_test_{dataset_name}.parquet"

        gold_train.to_parquet(gold_train_path, index=False)
        gold_test.to_parquet(gold_test_path, index=False)

        logger.info(f"[GOLD] Successfully built Gold dataset. Columns: {len(gold_train.columns)}.")
        return gold_train, gold_test

    def build_full_lake(self, dataset_name: str = "FD001"):
        """Executes full Bronze -> Silver -> Gold ETL pipeline."""
        self.run_bronze_ingestion(dataset_name)
        self.run_silver_processing(dataset_name)
        self.run_gold_enrichment(dataset_name)
        logger.info(f"Medallion Data Lake pipeline finished for {dataset_name}.")

if __name__ == "__main__":
    lake = MedallionDataLake()
    lake.build_full_lake("FD001")
