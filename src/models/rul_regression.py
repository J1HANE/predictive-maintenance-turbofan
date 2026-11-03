import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.logger import get_logger

logger = get_logger("RULRegression")

def nasa_asymmetric_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA C-MAPSS Official Asymmetric Penalty Score:
    d_i = y_pred - y_true
    s_i = exp(-d_i / 13) - 1 if d_i < 0 (early warning, mild penalty)
    s_i = exp( d_i / 10) - 1 if d_i >= 0 (late warning, harsh penalty)
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(scores))

class RULPredictor:
    """
    RUL Prediction Pipeline supporting Random Forest, Gradient Boosting,
    and XGBoost/LightGBM regressors with NASA evaluation metrics.
    """
    def __init__(self, model_type: str = "random_forest", n_estimators: int = 100, max_depth: int = 12):
        self.model_type = model_type
        if model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        else:
            try:
                import xgboost as xgb
                self.model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            except ImportError:
                logger.warning("XGBoost not available, falling back to GradientBoostingRegressor.")
                self.model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        logger.info(f"Training RUL regressor model ({self.model_type}) on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        logger.info("RUL Model training complete.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, a_min=0, a_max=None)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        y_pred = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        nasa_score = nasa_asymmetric_score(np.array(y_test), y_pred)

        metrics = {
            "model_type": self.model_type,
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "nasa_score": float(nasa_score)
        }
        logger.info(f"Evaluation Metrics [{self.model_type}]: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}, NASA Score={nasa_score:.1f}")
        return metrics
