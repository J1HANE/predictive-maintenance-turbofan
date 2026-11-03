import pytest
from pathlib import Path
import pandas as pd
from src.data.download_data import download_cmapss_dataset
from src.data.data_lake import MedallionDataLake

def test_data_acquisition_and_lake():
    # 1. Download/generate dataset
    download_cmapss_dataset(output_dir="data/raw")
    assert Path("data/raw/train_FD001.txt").exists()

    # 2. Run Medallion Data Lake
    lake = MedallionDataLake()
    df_train_b, df_test_b = lake.run_bronze_ingestion("FD001")
    assert not df_train_b.empty
    assert "ingestion_timestamp" in df_train_b.columns

    df_train_s, df_test_s = lake.run_silver_processing("FD001")
    assert "RUL" in df_train_s.columns
    assert "event_occurred" in df_train_s.columns

    gold_train, gold_test = lake.run_gold_enrichment("FD001")
    assert "RUL_piecewise" in gold_train.columns
    assert any("roll_mean" in c for c in gold_train.columns)
