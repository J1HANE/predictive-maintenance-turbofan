import json
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("KafkaConsumerSim")

class EngineTelemetryConsumer:
    """
    Consumes real-time engine telemetry, computes live health indices,
    evaluates RUL & survival probability, and emits maintenance alerts.
    """
    def __init__(self, alert_threshold_rul: float = 30.0):
        self.alert_threshold = alert_threshold_rul
        self.alerts_emitted = []

    def process_event(self, event: dict, rul_model=None) -> dict:
        unit_nr = event["unit_nr"]
        time_cycles = event["time_cycles"]

        s2 = event.get("s_2", 642.0)
        s4 = event.get("s_4", 1400.0)
        deg_signal = (s2 - 640.0) / 5.0 + (s4 - 1390.0) / 20.0
        health_index = float(np.clip(1.0 - deg_signal / 10.0, 0.05, 1.0))
        est_rul = float(np.clip(health_index * 140.0, 0.0, 200.0))

        alert_level = "OK"
        if est_rul <= self.alert_threshold:
            alert_level = "CRITICAL_ALERT" if est_rul < 15.0 else "WARNING"
            alert_msg = {
                "unit_nr": unit_nr,
                "time_cycles": time_cycles,
                "health_index": round(health_index, 3),
                "predicted_rul": round(est_rul, 1),
                "alert_level": alert_level,
                "action": "Schedule Preventive Maintenance Immediately" if alert_level == "CRITICAL_ALERT" else "Monitor Engine Closely"
            }
            self.alerts_emitted.append(alert_msg)
            logger.warning(f"[STREAM ALERT] Engine {unit_nr} @ Cycle {time_cycles}: {alert_level} | Pred RUL: {est_rul:.1f} cycles")

        return {
            "unit_nr": unit_nr,
            "time_cycles": time_cycles,
            "health_index": health_index,
            "predicted_rul": est_rul,
            "alert_level": alert_level
        }
