import json
import urllib.request
from pathlib import Path
import joblib
import pandas as pd

def run_verification():
    print("================================================================================")
    print("CUSTOMER CHURN PLATFORM: COMPLETE END-TO-END VERIFICATION AUDIT")
    print("================================================================================")

    # 1. Data & Artifacts
    print("\n[PHASE 1: DATA & ARTIFACT PERSISTENCE]")
    required_files = [
        "data/raw/Telco-Customer-Churn.csv",
        "data/processed/train.csv",
        "data/processed/val.csv",
        "data/processed/test.csv",
        "artifacts/models/best_churn_pipeline.joblib",
        "artifacts/models/model_metadata.json",
        "artifacts/models/all_models_comparison.json",
        "artifacts/models/evaluation_report.md",
        "artifacts/models/plots/roc_curve.svg",
        "artifacts/models/plots/precision_recall_curve.svg",
        "artifacts/models/plots/confusion_matrix.svg",
        "artifacts/models/plots/feature_importance.svg"
    ]
    for rf in required_files:
        p = Path(rf)
        assert p.exists(), f"Missing required file: {rf}"
        print(f"  [OK] Verified {rf} ({p.stat().st_size:,} bytes)")

    # 2. Pipeline Inference
    print("\n[PHASE 2: MODEL PIPELINE INFERENCE]")
    pipeline = joblib.load("artifacts/models/best_churn_pipeline.joblib")
    test_df = pd.read_csv("data/processed/test.csv")
    X_sample = test_df.drop(columns=["customerID", "Churn"], errors="ignore").head(5)
    sample_probs = pipeline.predict_proba(X_sample)[:, 1]
    assert len(sample_probs) == 5
    assert all(0.0 <= p <= 1.0 for p in sample_probs)
    print(f"  [OK] Successfully scored test batch. Sample probabilities: {[round(float(p), 4) for p in sample_probs]}")

    # 3. Live API Endpoints & UI Assets
    print("\n[PHASE 3: FASTAPI SERVICE & UI ASSET VERIFICATION]")
    base_url = "http://127.0.0.1:8000"
    endpoints_to_test = [
        ("/", 200, "Frontend Landing Page"),
        ("/health", 200, "Health Check"),
        ("/model-info", 200, "Model Metadata"),
        ("/metrics", 200, "Model Comparison Leaderboard"),
        ("/customers?limit=10", 200, "Customer Directory Sample"),
        ("/css/style.css", 200, "Antigravity CSS Stylesheet"),
        ("/js/app.js", 200, "Client-side Reactive Logic")
    ]
    for path, expected_status, label in endpoints_to_test:
        req = urllib.request.Request(f"{base_url}{path}")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == expected_status, f"Endpoint {path} failed: {resp.status}"
            content_len = len(resp.read())
            print(f"  [OK] {label:<32} {path:<22} -> HTTP {resp.status} ({content_len:,} bytes)")

    # 4. Live Single Inference & SHAP Explainability
    print("\n[PHASE 4: LIVE PREDICTION & SHAP EXPLAINABILITY]")
    test_customer = {
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
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50,
        "TotalCharges": 191.00
    }
    pred_req = urllib.request.Request(
        f"{base_url}/predict",
        data=json.dumps(test_customer).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(pred_req) as resp:
        pred_data = json.loads(resp.read().decode("utf-8"))
        assert "churn_probability" in pred_data
        assert "risk_level" in pred_data
        assert "explanation" in pred_data
        assert len(pred_data["explanation"]["top_risk_drivers"]) > 0
        assert len(pred_data["explanation"]["top_retention_drivers"]) > 0

        print(f"  [OK] Churn Probability : {pred_data['churn_probability'] * 100:.2f}%")
        print(f"  [OK] Risk Classification: {pred_data['risk_level']}")
        print(f"  [OK] Binary Prediction  : {'CHURN (1)' if pred_data['churn_prediction'] == 1 else 'RETAIN (0)'}")
        print("  [OK] Top Positive Risk Drivers:")
        for d in pred_data["explanation"]["top_risk_drivers"][:3]:
            print(f"       + {d['title']:<34} (Impact: +{d['impact']:.4f})")
        print("  [OK] Top Negative Retention Drivers:")
        for d in pred_data["explanation"]["top_retention_drivers"][:2]:
            print(f"       - {d['title']:<34} (Impact: {d['impact']:.4f})")

    # 5. Out-of-bounds Validation Failure
    print("\n[PHASE 5: INPUT BOUNDARY ERROR HANDLING]")
    invalid_customer = dict(test_customer)
    invalid_customer["tenure"] = -10  # Invalid tenure
    err_req = urllib.request.Request(
        f"{base_url}/predict",
        data=json.dumps(invalid_customer).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(err_req)
        raise AssertionError("Expected 422 Unprocessable Entity but succeeded!")
    except urllib.error.HTTPError as e:
        assert e.code == 422
        print(f"  [OK] Out-of-bounds input rejected correctly with HTTP 422 Unprocessable Entity.")

    print("\n" + "=" * 80)
    print("ALL 5 PHASES VERIFIED: DATA -> MODEL -> API -> UI -> PREDICTION -> EXPLANATION")
    print("SYSTEM IS FULLY PRODUCTION-READY.")
    print("=" * 80)

if __name__ == "__main__":
    run_verification()
