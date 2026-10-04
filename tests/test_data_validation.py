import pandas as pd
import pytest
from src.data.validator import validate_raw_dataset, EXPECTED_COLUMNS
from src.exceptions import DataValidationError

def create_sample_raw_df():
    data = {col: ["No"] * 5 for col in EXPECTED_COLUMNS}
    data["customerID"] = [f"CUST-{i}" for i in range(5)]
    data["gender"] = ["Female", "Male", "Female", "Male", "Female"]
    data["SeniorCitizen"] = [0, 1, 0, 0, 1]
    data["tenure"] = [1, 24, 0, 60, 12]
    data["MonthlyCharges"] = [29.85, 59.90, 20.00, 105.50, 75.00]
    data["TotalCharges"] = ["29.85", "1437.60", " ", "6330.00", "900.00"]
    data["Contract"] = ["Month-to-month", "One year", "Two year", "Two year", "Month-to-month"]
    data["PaymentMethod"] = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)", "Electronic check"]
    data["Churn"] = ["Yes", "No", "No", "No", "Yes"]
    return pd.DataFrame(data)

def test_valid_dataset_passes():
    df = create_sample_raw_df()
    cleaned_df, report = validate_raw_dataset(df, strict=True)
    assert report.is_valid is True
    assert len(cleaned_df) == 5
    assert report.whitespace_fixed_count == 1
    # Check that whitespace TotalCharges was converted to numeric 0.0 for tenure 0
    assert cleaned_df.loc[cleaned_df["tenure"] == 0, "TotalCharges"].iloc[0] == 0.0

def test_missing_required_column_raises():
    df = create_sample_raw_df().drop(columns=["Contract"])
    with pytest.raises(DataValidationError) as exc_info:
        validate_raw_dataset(df, strict=True)
    assert "Contract" in str(exc_info.value)

def test_duplicate_removal():
    df = create_sample_raw_df()
    duplicate_row = df.iloc[[0]]
    df_with_dups = pd.concat([df, duplicate_row], ignore_index=True)
    assert len(df_with_dups) == 6

    cleaned_df, report = validate_raw_dataset(df_with_dups, strict=False)
    assert len(cleaned_df) == 5
    assert report.duplicates_removed == 1
