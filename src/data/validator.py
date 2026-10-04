from dataclasses import dataclass, field
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from src.logger import logger
from src.exceptions import DataValidationError

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
]

VALID_CATEGORIES = {
    "gender": {"Male", "Female"},
    "SeniorCitizen": {0, 1},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    },
    "Churn": {"Yes", "No"}
}

@dataclass
class ValidationReport:
    total_rows: int = 0
    total_cols: int = 0
    missing_columns: List[str] = field(default_factory=list)
    whitespace_fixed_count: int = 0
    duplicates_removed: int = 0
    invalid_categories: Dict[str, List[Any]] = field(default_factory=dict)
    missing_values: Dict[str, int] = field(default_factory=dict)
    target_distribution: Dict[str, int] = field(default_factory=dict)
    is_valid: bool = True

def validate_raw_dataset(df: pd.DataFrame, strict: bool = True) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Validate dataset against schema, data types, category sets, and handle corrupted values.
    Returns cleaned dataframe and validation report.
    """
    report = ValidationReport(total_rows=len(df), total_cols=len(df.columns))
    cleaned_df = df.copy()

    # 1. Column existence
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in cleaned_df.columns]
    if missing_cols:
        report.missing_columns = missing_cols
        report.is_valid = False
        msg = f"Missing required columns in dataset: {missing_cols}"
        logger.error(msg)
        if strict:
            raise DataValidationError(msg, details={"missing_columns": missing_cols})

    # 2. Check and remove duplicates
    initial_len = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(subset=["customerID"] if "customerID" in cleaned_df.columns else None)
    report.duplicates_removed = initial_len - len(cleaned_df)
    if report.duplicates_removed > 0:
        logger.warning(f"Removed {report.duplicates_removed} duplicate customer records")

    # 3. Handle TotalCharges whitespace / empty string issues (Telco standard quirk)
    if "TotalCharges" in cleaned_df.columns:
        if not pd.api.types.is_numeric_dtype(cleaned_df["TotalCharges"]):
            # Detect spaces
            whitespace_mask = cleaned_df["TotalCharges"].astype(str).str.strip().eq("")
            report.whitespace_fixed_count = int(whitespace_mask.sum())
            if report.whitespace_fixed_count > 0:
                logger.info(f"Detected {report.whitespace_fixed_count} records with empty TotalCharges (tenure=0). Setting to 0.0")
                cleaned_df.loc[whitespace_mask, "TotalCharges"] = "0.0"
            cleaned_df["TotalCharges"] = pd.to_numeric(cleaned_df["TotalCharges"], errors="coerce")
            # If any remaining NaNs, fill with MonthlyCharges * tenure or 0
            cleaned_df["TotalCharges"] = cleaned_df["TotalCharges"].fillna(cleaned_df["MonthlyCharges"] * cleaned_df["tenure"])

    # 4. Numerical range checks
    if "tenure" in cleaned_df.columns:
        cleaned_df["tenure"] = pd.to_numeric(cleaned_df["tenure"], errors="coerce").fillna(0).astype(int)
        if (cleaned_df["tenure"] < 0).any():
            raise DataValidationError("Found negative values in tenure column")

    if "MonthlyCharges" in cleaned_df.columns:
        cleaned_df["MonthlyCharges"] = pd.to_numeric(cleaned_df["MonthlyCharges"], errors="coerce")
        if (cleaned_df["MonthlyCharges"] < 0).any():
            raise DataValidationError("Found negative values in MonthlyCharges column")

    # 5. Check categorical validity
    for col, allowed_vals in VALID_CATEGORIES.items():
        if col in cleaned_df.columns:
            observed_vals = set(cleaned_df[col].dropna().unique())
            invalid = observed_vals - allowed_vals
            if invalid:
                report.invalid_categories[col] = list(invalid)
                report.is_valid = False
                logger.warning(f"Invalid category values found in column '{col}': {invalid}")
                if strict and col == "Churn":
                    raise DataValidationError(f"Invalid target labels in Churn column: {invalid}")

    # 6. Check target distribution
    if "Churn" in cleaned_df.columns:
        report.target_distribution = cleaned_df["Churn"].value_counts().to_dict()
        logger.info(f"Target 'Churn' distribution: {report.target_distribution}")

    # 7. Check remaining missing values
    report.missing_values = cleaned_df.isnull().sum()[cleaned_df.isnull().sum() > 0].to_dict()

    logger.info(f"Dataset validation completed successfully. Rows: {len(cleaned_df)}, Valid: {report.is_valid}")
    return cleaned_df, report
