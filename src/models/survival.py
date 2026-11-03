import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("SurvivalAnalysis")

class KaplanMeierEstimator:
    """
    Non-parametric Kaplan-Meier Survival Estimator:
    S(t) = Prod_{t_i <= t} (1 - d_i / n_i)
    """
    def __init__(self):
        self.survival_table = None

    def fit(self, durations: np.ndarray, events: np.ndarray):
        df = pd.DataFrame({"duration": durations, "event": events})
        grouped = df.groupby("duration").agg(
            n_events=("event", "sum"),
            n_censored=("event", lambda x: (x == 0).sum()),
            total=("event", "count")
        ).reset_index()

        grouped = grouped.sort_values("duration")
        grouped["at_risk"] = len(durations) - grouped["total"].cumsum().shift(1, fill_value=0)
        grouped["survival_prob"] = (1.0 - grouped["n_events"] / grouped["at_risk"]).cumprod()

        self.survival_table = grouped
        logger.info("Kaplan-Meier model fitted successfully.")
        return self

    def predict_survival_at(self, t: float) -> float:
        if self.survival_table is None:
            raise ValueError("Model not fitted.")
        valid = self.survival_table[self.survival_table["duration"] <= t]
        if valid.empty:
            return 1.0
        return float(valid.iloc[-1]["survival_prob"])


class CoxProportionalHazardsModel:
    """
    Cox Proportional Hazards / Parametric Survival Model.
    Hazard rate h(t | X) = h_0(t) * exp(X @ beta)
    """
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.beta = None
        self.mean_x = None
        self.std_x = None
        self.baseline_hazard = None

    def fit(self, X: pd.DataFrame, durations: np.ndarray, events: np.ndarray):
        X_mat = np.asarray(X, dtype=float)
        self.mean_x = np.mean(X_mat, axis=0)
        self.std_x = np.std(X_mat, axis=0) + 1e-6
        X_norm = (X_mat - self.mean_x) / self.std_x

        # Fit hazard coefficients via regularized pseudo-inverse alignment
        # In survival regression, log-hazard correlates with negative duration
        target_risk = - (durations - np.mean(durations)) / (np.std(durations) + 1e-6)
        cov = X_norm.T @ X_norm + self.alpha * np.eye(X_norm.shape[1])
        self.beta = np.linalg.solve(cov, X_norm.T @ target_risk)

        logger.info(f"Cox Proportional Hazards model fitted across {X_mat.shape[1]} features.")
        return self

    def predict_partial_hazards(self, X: pd.DataFrame) -> np.ndarray:
        X_mat = np.asarray(X, dtype=float)
        X_norm = (X_mat - self.mean_x) / self.std_x
        log_hazard = X_norm @ self.beta
        return np.exp(log_hazard)

    def predict_survival_probability(self, X: pd.DataFrame, time_ahead: float = 20.0) -> np.ndarray:
        """Estimates individual engine survival probability over time_ahead cycles."""
        hazards = self.predict_partial_hazards(X)
        # Exponential survival curves: S(t) = exp(- lambda * t)
        base_rate = 0.005
        survival_probs = np.exp(- base_rate * hazards * time_ahead)
        return np.clip(survival_probs, 0.0, 1.0)


def compute_concordance_index(y_time: np.ndarray, y_pred_risk: np.ndarray, events: np.ndarray) -> float:
    """
    Calculates Harrell's Concordance Index (C-Index) for Survival Analysis performance.
    C-Index > 0.5 indicates predictive power above random chance (1.0 = perfect order).
    """
    concordant = 0
    total_pairs = 0
    n = len(y_time)

    for i in range(n):
        if events[i] == 0:
            continue
        for j in range(n):
            if y_time[i] < y_time[j]:
                total_pairs += 1
                if y_pred_risk[i] > y_pred_risk[j]:
                    concordant += 1
                elif y_pred_risk[i] == y_pred_risk[j]:
                    concordant += 0.5

    if total_pairs == 0:
        return 0.5
    return concordant / total_pairs
