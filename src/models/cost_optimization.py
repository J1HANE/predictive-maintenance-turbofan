import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("CostOptimization")

class MaintenanceCostEvaluator:
    """
    Evaluates total operational maintenance cost across a fleet of turbofans
    and finds the cost-optimal decision threshold tau* (RUL cut-off).
    """
    def __init__(self, cost_preventive: float = 500.0, cost_corrective: float = 5000.0, cost_downtime_per_cycle: float = 100.0):
        self.c_prev = cost_preventive
        self.c_correc = cost_corrective
        self.c_downtime = cost_downtime_per_cycle

    def compute_fleet_cost(self, y_true: np.ndarray, y_pred: np.ndarray, threshold_tau: float) -> dict:
        """
        Computes total cost for a fleet under maintenance policy:
        'Trigger maintenance when predicted RUL <= threshold_tau'
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n_units = len(y_true)

        n_preventive = 0
        n_corrective = 0
        total_unused_cycles = 0.0
        total_downtime_cycles = 0.0

        for true_rul, pred_rul in zip(y_true, y_pred):
            if pred_rul <= threshold_tau:
                # Maintenance triggered preventively
                if true_rul >= 0:
                    n_preventive += 1
                    total_unused_cycles += max(0, true_rul - threshold_tau)
                else:
                    n_corrective += 1
            else:
                # Maintenance not triggered in time -> engine fails in operation!
                if true_rul <= threshold_tau:
                    n_corrective += 1
                    total_downtime_cycles += abs(true_rul - threshold_tau) + 5.0
                else:
                    n_preventive += 1

        cost_prev_total = n_preventive * self.c_prev
        cost_correc_total = n_corrective * self.c_correc
        cost_loss_unused = total_unused_cycles * (self.c_downtime * 0.5)
        cost_downtime_total = total_downtime_cycles * self.c_downtime

        total_cost = cost_prev_total + cost_correc_total + cost_loss_unused + cost_downtime_total

        return {
            "threshold_tau": threshold_tau,
            "total_cost": float(total_cost),
            "n_preventive": n_preventive,
            "n_corrective": n_corrective,
            "avg_cost_per_engine": float(total_cost / max(1, n_units)),
            "cost_breakdown": {
                "preventive": float(cost_prev_total),
                "corrective": float(cost_correc_total),
                "downtime_penalties": float(cost_downtime_total)
            }
        }

    def optimize_threshold(self, y_true: np.ndarray, y_pred: np.ndarray, tau_range: tuple[int, int] = (5, 60)) -> dict:
        """Searches for optimal RUL threshold tau* minimizing total fleet cost."""
        best_tau = None
        best_cost = float("inf")
        best_result = None

        search_grid = range(tau_range[0], tau_range[1] + 1)
        for tau in search_grid:
            res = self.compute_fleet_cost(y_true, y_pred, threshold_tau=float(tau))
            if res["total_cost"] < best_cost:
                best_cost = res["total_cost"]
                best_tau = tau
                best_result = res

        logger.info(f"Optimal Maintenance Threshold found: tau* = {best_tau} cycles (Min Fleet Cost = ${best_cost:,.2f})")
        return best_result
