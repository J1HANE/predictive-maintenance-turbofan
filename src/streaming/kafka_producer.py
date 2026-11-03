import json
import time
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("KafkaProducerSim")

class EngineTelemetryProducer:
    """
    Simulates a real-time Kafka producer replaying turbofan engine telemetry data
    cycle-by-cycle for industrial predictive maintenance simulation.
    """
    def __init__(self, data_path: str = "data/silver/silver_test_FD001.parquet", delay_sec: float = 0.05):
        self.data_path = Path(data_path)
        self.delay_sec = delay_sec

    def stream_telemetry(self, max_cycles: int = 50, callback=None):
        """Streams engine sensor telemetry events."""
        if not self.data_path.exists():
            logger.warning(f"Data file {self.data_path} not found. Skipping streaming simulation.")
            return

        df = pd.read_parquet(self.data_path)
        logger.info(f"Starting simulated Kafka Telemetry Stream for {len(df)} records...")

        count = 0
        for idx, row in df.iterrows():
            if count >= max_cycles:
                break
            payload = {
                "unit_nr": int(row["unit_nr"]),
                "time_cycles": int(row["time_cycles"]),
                "setting_1": float(row["setting_1"]),
                "setting_2": float(row["setting_2"]),
                "setting_3": float(row["setting_3"]),
                "s_2": float(row.get("s_2", 0.0)),
                "s_3": float(row.get("s_3", 0.0)),
                "s_4": float(row.get("s_4", 0.0)),
                "s_7": float(row.get("s_7", 0.0)),
                "s_11": float(row.get("s_11", 0.0)),
                "s_12": float(row.get("s_12", 0.0)),
                "s_15": float(row.get("s_15", 0.0)),
                "s_20": float(row.get("s_20", 0.0)),
                "s_21": float(row.get("s_21", 0.0)),
                "timestamp": time.time()
            }

            if callback:
                callback(payload)

            count += 1
            time.sleep(self.delay_sec)

        logger.info(f"Completed streaming of {count} telemetry cycles.")
