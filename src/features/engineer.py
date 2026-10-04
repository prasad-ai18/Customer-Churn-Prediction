from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from src.logger import logger

class TelcoFeatureEngineer:
    """
    Domain-specific feature engineering for Telco Customer Churn.
    Transforms raw customer inputs into business-interpretable risk factors.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Ensure numeric types
        df["tenure"] = pd.to_numeric(df.get("tenure", 0), errors="coerce").fillna(0)
        df["MonthlyCharges"] = pd.to_numeric(df.get("MonthlyCharges", 0), errors="coerce").fillna(0)
        df["TotalCharges"] = pd.to_numeric(df.get("TotalCharges", 0), errors="coerce").fillna(0)

        # 1. Tenure Cohorts
        # Early months (0-12m) have highest churn risk; 49-72m represent high loyalty.
        bins = [-1, 12, 24, 48, 72]
        labels = ["0-12m", "13-24m", "25-48m", "49-72m"]
        df["tenure_cohort"] = pd.cut(df["tenure"], bins=bins, labels=labels).astype(str)

        # 2. Financial Velocity & Charge Ratio
        # Detects early tenure customers with disproportionate monthly commitment
        df["charge_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1.0)

        # Average historical monthly charges vs current monthly charges
        # If current MonthlyCharges is much higher than historical avg, indicates bill shock
        avg_historical_charge = df["TotalCharges"] / (df["tenure"] + 1.0)
        df["bill_shock_indicator"] = (df["MonthlyCharges"] - avg_historical_charge).clip(lower=0)

        # 3. Service Bundle Density (Count of active add-on features)
        service_cols = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        active_services = 0
        for col in service_cols:
            if col in df.columns:
                active_services += (df[col] == "Yes").astype(int)
        df["total_active_services"] = active_services

        # 4. Critical Friction Profiles
        # Fiber Optic without TechSupport or OnlineSecurity is known high-churn segment
        has_fiber = (df.get("InternetService", "") == "Fiber optic").astype(int)
        no_tech_support = (df.get("TechSupport", "") == "No").astype(int)
        df["fiber_without_tech_support"] = has_fiber * no_tech_support

        # 5. Contract & Payment Risk Indicators
        is_month_to_month = (df.get("Contract", "") == "Month-to-month").astype(int)
        is_electronic_check = (df.get("PaymentMethod", "") == "Electronic check").astype(int)
        df["high_risk_contract_payment"] = is_month_to_month * is_electronic_check

        # 6. Streamer Bundle Indicator
        has_tv = (df.get("StreamingTV", "") == "Yes").astype(int)
        has_movies = (df.get("StreamingMovies", "") == "Yes").astype(int)
        df["is_full_streamer"] = has_tv * has_movies

        return df


class TelcoDataTransformer:
    """
    Self-contained, production-grade scaler and categorical encoder.
    Guarantees consistent column schema between training and live inference.
    """

    def __init__(self):
        self.numeric_features = [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "charge_ratio",
            "bill_shock_indicator",
            "total_active_services"
        ]
        self.categorical_features = [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "tenure_cohort",
            "fiber_without_tech_support",
            "high_risk_contract_payment",
            "is_full_streamer"
        ]
        self.mean_stats: Dict[str, float] = {}
        self.std_stats: Dict[str, float] = {}
        self.encoded_column_names: List[str] = []
        self.category_levels: Dict[str, List[str]] = {}

    def fit(self, df: pd.DataFrame):
        """Learn feature scaling statistics and one-hot categories from training data."""
        # Compute scaling statistics for numeric columns
        for col in self.numeric_features:
            if col in df.columns:
                mean_val = float(df[col].mean())
                std_val = float(df[col].std())
                self.mean_stats[col] = mean_val
                self.std_stats[col] = std_val if std_val > 1e-6 else 1.0

        # Record categorical levels
        dummy_df_parts = []
        for col in self.categorical_features:
            if col in df.columns:
                levels = sorted([str(v) for v in df[col].dropna().unique()])
                self.category_levels[col] = levels
                dummy_cols = [f"{col}_{lvl}" for lvl in levels]
                dummy_df_parts.extend(dummy_cols)

        self.encoded_column_names = self.numeric_features + dummy_df_parts
        logger.info(f"Fitted data transformer: {len(self.numeric_features)} numeric features, {len(dummy_df_parts)} one-hot encoded columns (Total: {len(self.encoded_column_names)})")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned standardization and one-hot encoding."""
        out = pd.DataFrame(index=df.index)

        # Scale numeric features
        for col in self.numeric_features:
            mean = self.mean_stats.get(col, 0.0)
            std = self.std_stats.get(col, 1.0)
            vals = pd.to_numeric(df.get(col, mean), errors="coerce").fillna(mean)
            out[col] = (vals - mean) / std

        # One-hot encode categorical features against fixed vocabulary
        for col in self.categorical_features:
            series = df.get(col, pd.Series(["Unknown"] * len(df), index=df.index)).astype(str)
            levels = self.category_levels.get(col, [])
            for lvl in levels:
                dummy_name = f"{col}_{lvl}"
                out[dummy_name] = (series == lvl).astype(float)

        # Ensure all expected columns exist in strict order
        for col in self.encoded_column_names:
            if col not in out.columns:
                out[col] = 0.0

        return out[self.encoded_column_names]
