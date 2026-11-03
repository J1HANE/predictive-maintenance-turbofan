import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from src.utils.logger import get_logger

logger = get_logger("DriftMonitor")

class TelemetryDriftMonitor:
    """
    Monitors statistical data drift between reference training baseline
    and current incoming sensor telemetry using Kolmogorov-Smirnov test.
    """
    def __init__(self, p_value_threshold: float = 0.05):
        self.p_threshold = p_value_threshold

    def calculate_data_drift(self, ref_df: pd.DataFrame, curr_df: pd.DataFrame, sensor_cols: list[str]) -> dict:
        """
        Computes 2-sample KS test for each sensor to detect distribution shift.
        """
        valid_cols = [c for c in sensor_cols if c in ref_df.columns and c in curr_df.columns]
        drift_results = {}
        drifted_count = 0

        for col in valid_cols:
            ref_vals = ref_df[col].dropna().values
            curr_vals = curr_df[col].dropna().values

            if len(ref_vals) == 0 or len(curr_vals) == 0:
                continue

            ks_stat, p_val = ks_2samp(ref_vals, curr_vals)
            has_drift = p_val < self.p_threshold

            if has_drift:
                drifted_count += 1

            drift_results[col] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_val),
                "drift_detected": bool(has_drift)
            }

        drift_share = drifted_count / max(1, len(valid_cols))
        overall_drift = drift_share > 0.3

        logger.info(f"[DRIFT MONITOR] Evaluated {len(valid_cols)} sensors: {drifted_count} drifted ({drift_share:.1%}). Overall Drift: {overall_drift}")

        return {
            "drift_share": float(drift_share),
            "drifted_sensors_count": drifted_count,
            "overall_data_drift": overall_drift,
            "sensor_details": drift_results
        }
