import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.utils.logger import get_logger

logger = get_logger("FeatureEngineering")

class FeatureExtractor:
    """
    Extracts sensor features, normalizes signals, builds PCA health index,
    and formats train/test feature matrices for ML & Survival models.
    """
    def __init__(self, n_pca_components: int = 2):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca_components)
        self.is_fitted = False

    def fit_transform_gold(self, gold_df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
        """Fits scalers/PCA on training gold data and returns feature-transformed DataFrame."""
        df = gold_df.copy()
        valid_sensors = [col for col in sensor_cols if col in df.columns]

        # Scale sensor columns
        scaled_sensors = self.scaler.fit_transform(df[valid_sensors])
        scaled_df = pd.DataFrame(scaled_sensors, columns=[f"{c}_scaled" for c in valid_sensors], index=df.index)

        # PCA Degradation Profile
        pca_feats = self.pca.fit_transform(scaled_sensors)
        for i in range(self.pca.n_components):
            df[f"pca_component_{i+1}"] = pca_feats[:, i]

        # Health index derived from 1st PCA component
        pca1 = pca_feats[:, 0]
        pca1_norm = (pca1 - pca1.min()) / (pca1.max() - pca1.min() + 1e-6)
        df["health_index_pca"] = 1.0 - pca1_norm

        self.is_fitted = True
        df = pd.concat([df, scaled_df], axis=1)
        return df

    def transform_gold(self, gold_df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
        """Transforms test gold data using fitted scalers and PCA models."""
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before calling transform_gold.")

        df = gold_df.copy()
        valid_sensors = [col for col in sensor_cols if col in df.columns]

        scaled_sensors = self.scaler.transform(df[valid_sensors])
        scaled_df = pd.DataFrame(scaled_sensors, columns=[f"{c}_scaled" for c in valid_sensors], index=df.index)

        pca_feats = self.pca.transform(scaled_sensors)
        for i in range(self.pca.n_components):
            df[f"pca_component_{i+1}"] = pca_feats[:, i]

        pca1 = pca_feats[:, 0]
        pca1_norm = (pca1 - pca1.min()) / (pca1.max() - pca1.min() + 1e-6)
        df["health_index_pca"] = 1.0 - pca1_norm

        df = pd.concat([df, scaled_df], axis=1)
        return df

def prepare_ml_matrices(df: pd.DataFrame, feature_cols: list[str], target_col: str = "RUL_piecewise"):
    """Extracts X matrix and y target vector for ML training/testing."""
    valid_cols = [c for c in feature_cols if c in df.columns]
    X = df[valid_cols].fillna(0)
    y = df[target_col] if target_col in df.columns else None
    return X, y
