from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from src.logger import logger
from src.exceptions import InferenceError

class ChurnExplainer:
    """
    Model explainability engine implementing exact Tree SHAP and linear feature attributions.
    Produces local feature contributions for individual predictions and global feature rankings.
    """

    def __init__(self, pipeline: Any, background_sample: Optional[pd.DataFrame] = None):
        self.pipeline = pipeline
        self.feature_names: List[str] = pipeline.feature_names
        self.model = pipeline.model
        self.model_type = pipeline.model_type

    def explain_instance(self, input_df: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
        """
        Generate local SHAP explanations for a single customer prediction.
        Returns top positive (risk increasing) and negative (risk reducing) drivers.
        """
        try:
            # 1. Transform single customer through feature engineering & data transformer
            X_eng = self.pipeline.feature_engineer.transform(input_df)
            X_trans = self.pipeline.transformer.transform(X_eng)

            # 2. Extract SHAP values
            if self.model_type in ["xgboost", "random_forest"]:
                # Native XGBoost Tree SHAP calculation
                dmat = xgb.DMatrix(X_trans.values, feature_names=self.feature_names)
                contribs = self.model.predict(dmat, pred_contribs=True)
                # First M features are attributions; last column is base bias
                feature_contribs = contribs[0, :-1]
            else:
                # Linear attribution for Logistic Regression: weight * scaled_feature
                weights = getattr(self.model, "weights", np.zeros(len(self.feature_names)))
                feature_contribs = X_trans.values[0] * weights

            # Pair features with their SHAP contribution values
            contributions = []
            for name, val in zip(self.feature_names, feature_contribs):
                contributions.append({
                    "feature": name,
                    "shap_value": float(val),
                    "abs_shap": abs(float(val))
                })

            # Sort by absolute influence
            contributions.sort(key=lambda x: x["abs_shap"], reverse=True)

            risk_increasing = []
            risk_reducing = []

            for item in contributions:
                shap_val = item["shap_value"]
                fname = item["feature"]
                human_title = self._make_human_readable(fname, input_df)

                entry = {
                    "feature_key": fname,
                    "title": human_title,
                    "impact": round(float(shap_val), 4),
                    "direction": "INCREASES_CHURN_RISK" if shap_val > 0 else "REDUCES_CHURN_RISK"
                }

                if shap_val > 0 and len(risk_increasing) < top_k:
                    risk_increasing.append(entry)
                elif shap_val < 0 and len(risk_reducing) < top_k:
                    risk_reducing.append(entry)

            return {
                "top_risk_drivers": risk_increasing,
                "top_retention_drivers": risk_reducing,
                "disclaimer": (
                    "Explanations represent model-derived statistical signals associated with churn risk, "
                    "not causal guarantees. Retention actions should account for broader customer context."
                )
            }
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            raise InferenceError(f"Failed to compute model explanation: {e}")

    def _make_human_readable(self, feature_key: str, raw_df: pd.DataFrame) -> str:
        """Translate one-hot or engineered technical column names into business descriptions."""
        tenure_val = int(raw_df.get("tenure", [0]).iloc[0]) if "tenure" in raw_df else 0
        monthly_val = float(raw_df.get("MonthlyCharges", [0]).iloc[0]) if "MonthlyCharges" in raw_df else 0.0
        total_val = float(raw_df.get("TotalCharges", [0]).iloc[0]) if "TotalCharges" in raw_df else 0.0

        translations = {
            "Contract_Month-to-month": "Month-to-Month Contract",
            "Contract_Two year": "Two-Year Long-Term Contract",
            "Contract_One year": "One-Year Contract",
            "InternetService_Fiber optic": "Fiber Optic High-Speed Internet",
            "InternetService_DSL": "DSL Internet Service",
            "InternetService_No": "No Internet Service",
            "PaymentMethod_Electronic check": "Electronic Check Billing",
            "PaymentMethod_Credit card (automatic)": "Automatic Credit Card Billing",
            "PaymentMethod_Bank transfer (automatic)": "Automatic Bank Transfer",
            "PaymentMethod_Mailed check": "Mailed Paper Check",
            "OnlineSecurity_No": "No Online Security Service",
            "OnlineSecurity_Yes": "Active Online Security Protection",
            "TechSupport_No": "No Technical Support Plan",
            "TechSupport_Yes": "Active Premium Tech Support",
            "PaperlessBilling_Yes": "Paperless Billing Enrolled",
            "PaperlessBilling_No": "Paper Billing",
            "tenure": f"Customer Tenure ({tenure_val} months)",
            "MonthlyCharges": f"Monthly Charges (${monthly_val:.2f})",
            "TotalCharges": f"Total Spend (${total_val:.2f})",
            "fiber_without_tech_support_1.0": "Fiber Optic with No Tech Support",
            "high_risk_contract_payment_1.0": "Month-to-Month with Electronic Check",
            "charge_ratio": "Monthly-to-Total Spend Ratio",
            "total_active_services": "Total Active Add-on Services",
            "bill_shock_indicator": "Recent Monthly Bill Spike",
            "is_full_streamer_1.0": "Streaming TV & Movies Bundle",
            "tenure_cohort_0-12m": "Early Lifecycle Cohort (0-12m)",
            "tenure_cohort_49-72m": "Mature Loyalty Cohort (49-72m)"
        }

        if feature_key in translations:
            return translations[feature_key]

        # Generic formatting
        return feature_key.replace("_", " ").title()

    def get_global_feature_importance(self, top_n: int = 15) -> List[Dict[str, Any]]:
        """Extract global feature importance from the model."""
        importances = []
        if self.model_type in ["xgboost", "random_forest"]:
            score_dict = self.model.get_score(importance_type="gain")
            # Fill all features
            total_gain = sum(score_dict.values()) if score_dict else 1.0
            for name in self.feature_names:
                gain = score_dict.get(name, 0.0)
                norm_imp = gain / total_gain if total_gain > 0 else 0.0
                importances.append({
                    "feature": name,
                    "title": self._make_human_readable(name, pd.DataFrame()),
                    "importance": round(float(norm_imp), 4)
                })
        else:
            weights = getattr(self.model, "weights", np.zeros(len(self.feature_names)))
            abs_w = np.abs(weights)
            total_w = sum(abs_w) if sum(abs_w) > 0 else 1.0
            for name, w in zip(self.feature_names, abs_w):
                importances.append({
                    "feature": name,
                    "title": self._make_human_readable(name, pd.DataFrame()),
                    "importance": round(float(w / total_w), 4)
                })

        importances.sort(key=lambda x: x["importance"], reverse=True)
        return importances[:top_n]
