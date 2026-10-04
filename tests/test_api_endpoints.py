import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "version" in data

def test_model_info_endpoint():
    res = client.get("/model-info")
    assert res.status_code == 200
    data = res.json()
    assert "model_name" in data
    assert "primary_metrics" in data
    assert "roc_auc" in data["primary_metrics"]
    assert "global_feature_importance" in data

def test_predict_endpoint_valid():
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
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
        "MonthlyCharges": 99.0,
        "TotalCharges": 99.0
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "churn_probability" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["Low", "Medium", "High"]
    assert "explanation" in data
    assert len(data["explanation"]["top_risk_drivers"]) > 0

def test_predict_endpoint_invalid_category_fails():
    payload = {
        "gender": "AlienGender",  # Invalid category
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 50.0
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422  # Pydantic validation error

def test_batch_prediction_endpoint():
    payload = {
        "customers": [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 75.0,
                "TotalCharges": 150.0
            }
        ]
    }
    res = client.post("/predict/batch", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_processed"] == 1
    assert len(data["predictions"]) == 1

def test_customers_sample_endpoint():
    res = client.get("/customers?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert "customers" in data
    assert len(data["customers"]) == 10
    assert "predicted_churn_prob" in data["customers"][0]

def test_cors_headers_present():
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST"
    }
    res = client.options("/predict", headers=headers)
    assert res.status_code == 200
    assert "access-control-allow-origin" in res.headers
    assert res.headers["access-control-allow-origin"] in ["http://localhost:3000", "*"]

def test_predict_endpoint_numeric_validation_error():
    # Negative tenure and negative monthly charges should fail validation with 422
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": -5,  # Invalid: ge=0 required
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": -20.0,  # Invalid: ge=0.0 required
        "TotalCharges": 50.0
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    errors = res.json()["detail"]
    fields_with_errors = [e["loc"][-1] for e in errors]
    assert "tenure" in fields_with_errors
    assert "MonthlyCharges" in fields_with_errors
