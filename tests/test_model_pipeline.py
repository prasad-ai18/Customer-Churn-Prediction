import joblib
import pandas as pd
from config.settings import settings
from src.models.explainer import ChurnExplainer

def test_pipeline_artifact_loading():
    pipeline_path = settings.MODEL_ARTIFACTS_DIR / "best_churn_pipeline.joblib"
    assert pipeline_path.exists(), "Trained model pipeline joblib must exist"
    pipeline = joblib.load(pipeline_path)
    assert pipeline is not None
    assert hasattr(pipeline, "predict_proba")
    assert hasattr(pipeline, "predict")

def test_pipeline_prediction_ranges():
    pipeline_path = settings.MODEL_ARTIFACTS_DIR / "best_churn_pipeline.joblib"
    pipeline = joblib.load(pipeline_path)

    sample_df = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
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
        "MonthlyCharges": 95.0,
        "TotalCharges": 285.0
    }])

    proba = pipeline.predict_proba(sample_df)
    assert proba.shape == (1, 2)
    p0, p1 = proba[0, 0], proba[0, 1]
    assert 0.0 <= p1 <= 1.0
    assert abs((p0 + p1) - 1.0) < 1e-4

    decision = pipeline.predict(sample_df)
    assert decision[0] in [0, 1]

def test_explainer_drivers():
    pipeline_path = settings.MODEL_ARTIFACTS_DIR / "best_churn_pipeline.joblib"
    pipeline = joblib.load(pipeline_path)
    explainer = ChurnExplainer(pipeline)

    sample_df = pd.DataFrame([{
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 65.0,
        "TotalCharges": 3900.0
    }])

    explanation = explainer.explain_instance(sample_df, top_k=3)
    assert "top_risk_drivers" in explanation
    assert "top_retention_drivers" in explanation
    assert "disclaimer" in explanation
    assert len(explanation["top_retention_drivers"]) > 0
