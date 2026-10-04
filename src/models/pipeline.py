from typing import Any, List, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from src.features.engineer import TelcoFeatureEngineer, TelcoDataTransformer

class NumpyLogisticRegression:
    """Production-grade Logistic Regression baseline with balanced class weights and L2 regularization."""

    def __init__(self, lr: float = 0.05, max_iter: int = 500, l2: float = 1e-3, class_weight: str = "balanced"):
        self.lr = lr
        self.max_iter = max_iter
        self.l2 = l2
        self.class_weight = class_weight
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        if self.class_weight == "balanced":
            n_pos = np.sum(y == 1)
            n_neg = np.sum(y == 0)
            sample_weight = np.where(y == 1, n_samples / (2.0 * max(n_pos, 1)), n_samples / (2.0 * max(n_neg, 1)))
        else:
            sample_weight = np.ones(n_samples)

        for _ in range(self.max_iter):
            z = np.clip(np.dot(X, self.weights) + self.bias, -20.0, 20.0)
            preds = 1.0 / (1.0 + np.exp(-z))
            err = (preds - y) * sample_weight
            dw = (np.dot(X.T, err) / n_samples) + self.l2 * self.weights
            db = float(np.sum(err) / n_samples)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        z = np.clip(np.dot(X, self.weights) + self.bias, -20.0, 20.0)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p1, p1])


class ChurnPipeline:
    """
    Unified end-to-end inference and training pipeline.
    Combines feature engineering, numerical/categorical data transformation, and trained model.
    """

    def __init__(
        self,
        feature_engineer: TelcoFeatureEngineer,
        transformer: TelcoDataTransformer,
        model: Any,
        model_type: str,
        feature_names: List[str]
    ):
        self.feature_engineer = feature_engineer
        self.transformer = transformer
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Run complete transformation and output calibrated probabilities [P(No), P(Yes)]."""
        df_eng = self.feature_engineer.transform(df)
        df_trans = self.transformer.transform(df_eng)

        if self.model_type in ["xgboost", "random_forest", "gradient_boosting"]:
            dmat = xgb.DMatrix(df_trans.values, feature_names=self.feature_names)
            p1 = self.model.predict(dmat)
            return np.column_stack([1.0 - p1, p1])
        else:
            return self.model.predict_proba(df_trans.values)

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Generate binary churn decisions."""
        proba = self.predict_proba(df)[:, 1]
        return (proba >= threshold).astype(int)
