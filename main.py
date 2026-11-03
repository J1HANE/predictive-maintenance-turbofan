import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.logger import get_logger
from src.data.download_data import download_cmapss_dataset
from src.data.data_lake import MedallionDataLake
from src.features.feature_engineering import FeatureExtractor, prepare_ml_matrices
from src.models.survival import KaplanMeierEstimator, CoxProportionalHazardsModel, compute_concordance_index
from src.models.rul_regression import RULPredictor, nasa_asymmetric_score
from src.models.cost_optimization import MaintenanceCostEvaluator
from src.streaming.kafka_producer import EngineTelemetryProducer
from src.streaming.kafka_consumer import EngineTelemetryConsumer
from src.mlops.mlflow_tracker import MLOpsTracker
from src.mlops.drift_monitor import TelemetryDriftMonitor
from src.rag.maintenance_rag import MaintenanceRAGAssistant

logger = get_logger("MainPipeline")

def run_end_to_end_pipeline():
    logger.info("="*80)
    logger.info("[PIPELINE START] STARTING END-TO-END PREDICTIVE MAINTENANCE PIPELINE")
    logger.info("="*80)

    # ----------------------------------------------------
    # STEP 1: DATA ACQUISITION
    # ----------------------------------------------------
    logger.info("\n--- [STEP 1] Data Acquisition (NASA C-MAPSS) ---")
    download_cmapss_dataset(output_dir="data/raw")

    # ----------------------------------------------------
    # STEP 2: MEDALLION DATA LAKE (BRONZE -> SILVER -> GOLD)
    # ----------------------------------------------------
    logger.info("\n--- [STEP 2] Medallion Data Lake ETL Pipeline ---")
    lake = MedallionDataLake()
    gold_train, gold_test = lake.run_bronze_ingestion("FD001"), None
    gold_train, gold_test = lake.run_silver_processing("FD001")
    gold_train, gold_test = lake.run_gold_enrichment("FD001")

    # ----------------------------------------------------
    # STEP 3: FEATURE ENGINEERING & PCA HEALTH INDEX
    # ----------------------------------------------------
    logger.info("\n--- [STEP 3] Feature Engineering & PCA Health Index ---")
    sensor_cols = lake.config["data"]["active_sensors"]
    extractor = FeatureExtractor(n_pca_components=2)
    train_transformed = extractor.fit_transform_gold(gold_train, sensor_cols)
    test_transformed = extractor.transform_gold(gold_test, sensor_cols)

    feature_cols = [c for c in train_transformed.columns if "roll_" in c or "scaled" in c or "pca_" in c]
    X_train, y_train = prepare_ml_matrices(train_transformed, feature_cols, target_col="RUL_piecewise")

    # Extract test features at last observed cycle per unit
    test_latest = test_transformed.groupby("unit_nr").last().reset_index()
    X_test, y_test = prepare_ml_matrices(test_latest, feature_cols, target_col="RUL")

    # ----------------------------------------------------
    # STEP 4: SURVIVAL ANALYSIS
    # ----------------------------------------------------
    logger.info("\n--- [STEP 4] Survival Analysis (Kaplan-Meier & Cox PH) ---")
    km = KaplanMeierEstimator()
    km.fit(gold_train["time_cycles"].values, gold_train["event_occurred"].values)
    prob_surv_100 = km.predict_survival_at(100)
    logger.info(f"Kaplan-Meier Baseline Fleet Survival Probability at 100 cycles: {prob_surv_100*100:.1f}%")

    cox = CoxProportionalHazardsModel()
    cox.fit(X_train, gold_train["time_cycles"].values, gold_train["event_occurred"].values)
    hazards_test = cox.predict_partial_hazards(X_test)
    c_index = compute_concordance_index(y_test.values, hazards_test, test_latest["event_occurred"].values)
    logger.info(f"Cox Proportional Hazards C-Index: {c_index:.3f}")

    # ----------------------------------------------------
    # STEP 5: RUL REGRESSION MODEL & EVALUATION
    # ----------------------------------------------------
    logger.info("\n--- [STEP 5] RUL Regression Modeling & NASA Scoring ---")
    rf_model = RULPredictor(model_type="random_forest", n_estimators=100, max_depth=12)
    rf_model.fit(X_train, y_train)
    rf_metrics = rf_model.evaluate(X_test, y_test)
    preds_test = rf_model.predict(X_test)

    # ----------------------------------------------------
    # STEP 6: MAINTENANCE COST OPTIMIZATION
    # ----------------------------------------------------
    logger.info("\n--- [STEP 6] Financial Maintenance Policy Optimization ---")
    evaluator = MaintenanceCostEvaluator(cost_preventive=500.0, cost_corrective=5000.0, cost_downtime_per_cycle=100.0)
    optimal_policy = evaluator.optimize_threshold(y_test.values, preds_test, tau_range=(5, 50))

    # ----------------------------------------------------
    # STEP 7: REAL-TIME STREAMING SIMULATION
    # ----------------------------------------------------
    logger.info("\n--- [STEP 7] Real-time Streaming Simulation (Kafka Producer/Consumer) ---")
    producer = EngineTelemetryProducer(data_path="data/silver/silver_test_FD001.parquet", delay_sec=0.001)
    consumer = EngineTelemetryConsumer(alert_threshold_rul=optimal_policy["threshold_tau"])
    producer.stream_telemetry(max_cycles=30, callback=consumer.process_event)
    logger.info(f"Streaming finished. {len(consumer.alerts_emitted)} maintenance alerts triggered.")

    # ----------------------------------------------------
    # STEP 8: MLOPS & DATA DRIFT MONITORING
    # ----------------------------------------------------
    logger.info("\n--- [STEP 8] MLOps Tracking & Data Drift Monitoring ---")
    tracker = MLOpsTracker()
    tracker.log_experiment(
        run_name="rf_rul_baseline",
        params={"n_estimators": 100, "max_depth": 12, "features_count": len(feature_cols)},
        metrics=rf_metrics,
        model=rf_model.model
    )

    drift_monitor = TelemetryDriftMonitor()
    drift_report = drift_monitor.calculate_data_drift(gold_train, gold_test, sensor_cols[:7])

    # ----------------------------------------------------
    # STEP 9: GENAI / RAG MAINTENANCE REPORT DEMO
    # ----------------------------------------------------
    logger.info("\n--- [STEP 9] GenAI / RAG Maintenance Report Generation ---")
    rag = MaintenanceRAGAssistant()
    sample_report = rag.generate_report(
        unit_nr=1,
        current_cycle=int(test_latest.iloc[0]["time_cycles"]),
        predicted_rul=float(preds_test[0]),
        survival_prob_20=0.78,
        health_index=0.42
    )

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_file = reports_dir / "sample_maintenance_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(sample_report)
    logger.info(f"Generated sample GenAI RAG Maintenance Report at {report_file}.")

    logger.info("\n" + "="*80)
    logger.info("[PIPELINE SUCCESS] END-TO-END PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*80)

if __name__ == "__main__":
    run_end_to_end_pipeline()
