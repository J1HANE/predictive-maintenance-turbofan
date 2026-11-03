import os
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("MLflowTracker")

class MLOpsTracker:
    """
    MLflow experiment tracking and model registry integration.
    """
    def __init__(self, experiment_name: str = "predictive_maintenance_c_mapss", tracking_uri: str = "file:./mlruns"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.has_mlflow = False

        try:
            import mlflow
            self.mlflow = mlflow
            self.mlflow.set_tracking_uri(tracking_uri)
            self.mlflow.set_experiment(experiment_name)
            self.has_mlflow = True
            logger.info(f"MLflow tracker initialized with experiment '{experiment_name}'.")
        except ImportError:
            logger.warning("MLflow library not installed. Running in mock MLOps mode.")

    def log_experiment(self, run_name: str, params: dict, metrics: dict, model=None):
        """Logs parameters, evaluation metrics, and model artifact to MLflow."""
        if not self.has_mlflow:
            logger.info(f"[MLOps Log Mock] Run: {run_name} | Params: {params} | Metrics: {metrics}")
            return

        with self.mlflow.start_run(run_name=run_name):
            for k, v in params.items():
                self.mlflow.log_param(k, v)

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.mlflow.log_metric(k, float(v))

            if model is not None:
                try:
                    self.mlflow.sklearn.log_model(model, "model")
                except Exception as e:
                    logger.debug(f"Could not log sklearn model object to MLflow: {e}")

            logger.info(f"Successfully logged run '{run_name}' to MLflow.")
