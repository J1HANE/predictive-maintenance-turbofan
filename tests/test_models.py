import numpy as np
import pandas as pd
from src.models.survival import KaplanMeierEstimator, CoxProportionalHazardsModel, compute_concordance_index
from src.models.rul_regression import RULPredictor, nasa_asymmetric_score
from src.models.cost_optimization import MaintenanceCostEvaluator

def test_survival_models():
    durations = np.array([10, 20, 30, 40, 50])
    events = np.array([1, 1, 0, 1, 0])

    km = KaplanMeierEstimator()
    km.fit(durations, events)
    prob = km.predict_survival_at(25)
    assert 0.0 <= prob <= 1.0

    X = pd.DataFrame({"f1": [1, 2, 3, 4, 5], "f2": [5, 4, 3, 2, 1]})
    cox = CoxProportionalHazardsModel()
    cox.fit(X, durations, events)
    hazards = cox.predict_partial_hazards(X)
    assert len(hazards) == 5

    c_index = compute_concordance_index(durations, hazards, events)
    assert 0.0 <= c_index <= 1.0

def test_rul_and_cost():
    X_train = pd.DataFrame({"feat1": np.random.randn(50), "feat2": np.random.randn(50)})
    y_train = pd.Series(np.random.randint(10, 100, size=50))

    model = RULPredictor(model_type="random_forest", n_estimators=10)
    model.fit(X_train, y_train)
    preds = model.predict(X_train)
    assert len(preds) == 50

    score = nasa_asymmetric_score(y_train.values, preds)
    assert isinstance(score, float)

    evaluator = MaintenanceCostEvaluator()
    res = evaluator.optimize_threshold(y_train.values, preds, tau_range=(10, 30))
    assert "threshold_tau" in res
    assert res["total_cost"] >= 0
