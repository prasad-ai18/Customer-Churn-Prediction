import pandas as pd
from src.features.engineer import TelcoFeatureEngineer, TelcoDataTransformer

def test_feature_engineer_adds_domain_features():
    df = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 6,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 510.0
    }])

    engineer = TelcoFeatureEngineer()
    eng_df = engineer.transform(df)

    assert "tenure_cohort" in eng_df.columns
    assert eng_df["tenure_cohort"].iloc[0] == "0-12m"
    assert "charge_ratio" in eng_df.columns
    assert "total_active_services" in eng_df.columns
    assert eng_df["total_active_services"].iloc[0] == 2  # StreamingTV + StreamingMovies
    assert eng_df["fiber_without_tech_support"].iloc[0] == 1
    assert eng_df["high_risk_contract_payment"].iloc[0] == 1
    assert eng_df["is_full_streamer"].iloc[0] == 1

def test_transformer_consistency():
    train_df = pd.DataFrame([{
        "tenure": 10,
        "MonthlyCharges": 50.0,
        "TotalCharges": 500.0,
        "charge_ratio": 0.1,
        "bill_shock_indicator": 0.0,
        "total_active_services": 3,
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "tenure_cohort": "0-12m",
        "fiber_without_tech_support": 0,
        "high_risk_contract_payment": 1,
        "is_full_streamer": 0
    }])

    transformer = TelcoDataTransformer()
    transformer.fit(train_df)

    transformed = transformer.transform(train_df)
    assert len(transformed.columns) == len(transformer.encoded_column_names)
    assert not transformed.isnull().any().any()
