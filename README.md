# Customer Churn Intelligence & Prediction Platform

A production-ready machine learning platform that predicts customer churn, categorizes churn risk levels (`Low`, `Medium`, `High`), and provides actionable local SHAP feature attributions and counterfactual what-if simulations for every customer.

Built strictly according to **SOP Project 01 — Customer Churn Intelligence & Prediction Platform**.

---

## 1. End-to-End System Architecture

```text
       ┌────────────────────────────────────────────────────────┐
       │             Raw Dataset Ingestion                      │
       │    (IBM Telco Customer Churn, 7,043 rows, 21 cols)     │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │          Validation & Leakage-Free Splitting           │
       │   - Telco whitespace correction (tenure=0, charges=0)  │
       │   - Stratified Split: Train (65%), Val (15%), Test (20%)│
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │         Feature Engineering & Transformation           │
       │   - Tenure cohorts (0-12m, 13-24m, 25-48m, 49-72m)     │
       │   - Charge ratios & bill shock indicators              │
       │   - Service stickiness & support friction flags        │
       │   - Robust Z-score scaler & categorical encoding       │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │         Model Training & Benchmarking                  │
       │   - Imbalance mitigation (scale_pos_weight = 2.77)     │
       │   - Logistic Regression Baseline (Champion)            │
       │   - XGBoost Class-Weighted Classifier                  │
       │   - Gradient Boosting & Random Forest Ensembles        │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │         Evaluation & Model Selection                   │
       │   - Out-of-sample: ROC-AUC, PR-AUC, F1, Recall, Prec   │
       │   - 2x2 Confusion Matrix & Contract Cohort Slices      │
       │   - Pipeline serialization (best_churn_pipeline.joblib)│
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │             FastAPI Production REST Layer              │
       │   - Endpoints: /predict, /predict/batch, /model-info,  │
       │                /health, /metrics, /customers           │
       │   - Strict Pydantic v2 validation & error handling     │
       │   - Local SHAP explainability & plain-English signals  │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │             Modern Antigravity AI Frontend             │
       │   - Executive Dashboard with retention playbook        │
       │   - Customer Prediction Simulator & Radial Gauge       │
       │   - Customer Risk Analysis portfolio directory         │
       │   - Prediction Explanation & Counterfactual What-If    │
       │   - Model Performance & Interactive Canvas Curves      │
       └────────────────────────────────────────────────────────┘
```

---

## 2. Project Directory Structure

```text
Customer Churn Prediction/
├── .env                              # Active environment configuration
├── .env.example                      # Template environment configuration
├── requirements.txt                  # Production and development dependencies
├── README.md                         # Comprehensive documentation
├── train.py                          # Top-level training CLI entry point
├── evaluate.py                       # Top-level evaluation CLI entry point
│
├── config/
│   ├── __init__.py
│   └── settings.py                   # Pydantic-Settings environment loader
│
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv  # Verified IBM Telco dataset
│   └── processed/
│       ├── train.csv                 # 4,578 stratified training samples
│       ├── val.csv                   # 1,056 stratified validation samples
│       └── test.csv                  # 1,409 held-out test samples
│
├── src/
│   ├── __init__.py
│   ├── logger.py                     # Structured colorized logging engine
│   ├── exceptions.py                 # Domain-specific exception hierarchy
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessor.py           # Ingestion, validation & splitting
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineer.py               # Telco feature engineering & transforms
│   ├── models/
│   │   ├── __init__.py
│   │   ├── pipeline.py               # Serializable ChurnPipeline & models
│   │   ├── trainer.py                # Multi-model benchmarking engine
│   │   ├── evaluator.py              # Empirical evaluation metrics & curves
│   │   └── explainer.py              # SHAP & linear attribution explainer
│   └── api/
│       ├── __init__.py
│       ├── app.py                    # FastAPI application factory & middleware
│       ├── routes.py                 # REST routes (/predict, /health, etc.)
│       └── schemas.py                # Pydantic input/output schemas
│
├── scripts/
│   ├── __init__.py
│   ├── train.py                      # Standalone training CLI script
│   └── evaluate.py                   # Standalone evaluation & slice audit script
│
├── artifacts/
│   └── models/
│       ├── best_churn_pipeline.joblib# Serialized champion pipeline
│       ├── model_metadata.json       # Pipeline metadata & feature coefficients
│       ├── all_models_comparison.json# Complete multi-model leaderboard metrics
│       ├── evaluation_report.md      # Detailed evaluation markdown report
│       └── plots/                    # High-resolution vector visualizations
│           ├── roc_curve.svg
│           ├── precision_recall_curve.svg
│           ├── confusion_matrix.svg
│           └── feature_importance.svg
│
├── frontend/
│   ├── index.html                    # 5-section modern AI web interface
│   ├── css/
│   │   └── style.css                 # Antigravity design system & glassmorphism
│   └── js/
│       └── app.js                    # Reactive state, canvas charts & API client
│
└── tests/
    ├── __init__.py
    ├── test_data_validation.py       # Data cleaning & schema validation tests
    ├── test_feature_engineering.py   # Feature transform & encoding tests
    ├── test_model_pipeline.py        # Pipeline serialization & prediction tests
    └── test_api_endpoints.py         # FastAPI routes, CORS & validation tests
```

---

## 3. Real Held-Out Test Evaluation Benchmark

Evaluated on strictly held-out test data (`N = 1,409`, 374 churners, 1,035 retained). All metrics are empirically generated:

| Model Candidate | ROC-AUC | PR-AUC (Avg Prec) | F1-Score | Recall (Sensitivity) | Precision | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Champion)** | **0.8464** | **0.6753** | **0.6265** | **79.14%** | **51.84%** | **74.95%** |
| **XGBoost (Class Weighted)** | 0.8429 | 0.6717 | 0.6100 | 78.61% | 49.83% | 73.17% |
| **Gradient Boosting** | 0.8442 | 0.6705 | 0.5985 | 53.21% | 68.38% | 81.33% |
| **Random Forest** | 0.8405 | 0.6566 | 0.5957 | 55.35% | 64.49% | 80.70% |

### Champion Confusion Matrix (Test Set @ threshold = 0.50):
```text
                       Predicted Retained (0)    Predicted Churn (1)
Actual Retained (0) :         760 (TN)                  275 (FP)
Actual Churn (1)    :          78 (FN)                  296 (TP)
```
- **Churn Detection Recall**: 79.14% (captured 296 out of 374 actual churners).
- **False Negative Rate**: 20.86% (only 78 missed churners across the entire test set).

### Contract Cohort Slice Performance:
| Contract Slice | Total Records | Real Churn Rate | Recall (Detection Rate) | Avg Predicted Risk |
| :--- | :---: | :---: | :---: | :---: |
| **Month-to-month** | 789 | 42.21% | **87.39%** | 62.14% |
| **One year** | 301 | 9.30% | 17.86% | 24.81% |
| **Two year** | 319 | 4.08% | 0.00% | 10.32% |

---

## 4. Quickstart Setup & Installation

### Step 1: Clone and Enter Repository
```bash
git clone <repository_url>
cd "Customer Churn Prediction"
```

### Step 2: Install Python Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
```bash
copy .env.example .env
```

### Step 4: Run Reproducible Training Pipeline
Trains all 4 models, ranks by retention metric ($F_1 + \text{PR-AUC}$), saves the champion pipeline, and exports metadata and SVG plots:
```bash
python train.py
# Or with explicit arguments:
python scripts/train.py --data-path data/raw/Telco-Customer-Churn.csv
```

### Step 5: Run Standalone Evaluation & Slice Audit
```bash
python evaluate.py
# Or with custom decision cutoff:
python scripts/evaluate.py --threshold 0.50
```

### Step 6: Run Automated Test Suite
```bash
python -m pytest tests/ -v
```

### Step 7: Launch FastAPI Server & Web App
```bash
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```
- Web Application: **`http://127.0.0.1:8000/`**
- Interactive Swagger API Docs: **`http://127.0.0.1:8000/docs`**
- ReDoc API Documentation: **`http://127.0.0.1:8000/redoc`**

---

## 5. REST API Documentation & Examples

### `POST /predict`
Scores a single customer and returns probability, risk tier, decision, and local SHAP drivers.

**Request Payload:**
```json
{
  "customerID": "7590-VHVEG",
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
```

**Response Payload:**
```json
{
  "churn_probability": 0.9552,
  "risk_level": "High",
  "churn_prediction": 1,
  "explanation": {
    "top_risk_drivers": [
      {
        "feature_key": "bill_shock_indicator",
        "title": "Recent Monthly Bill Spike",
        "impact": 0.8198,
        "direction": "INCREASES_CHURN_RISK"
      },
      {
        "feature_key": "tenure",
        "title": "Customer Tenure (2 months)",
        "impact": 0.5796,
        "direction": "INCREASES_CHURN_RISK"
      },
      {
        "feature_key": "Contract_Month-to-month",
        "title": "Month-to-Month Contract",
        "impact": 0.4869,
        "direction": "INCREASES_CHURN_RISK"
      }
    ],
    "top_retention_drivers": [
      {
        "feature_key": "MultipleLines_No",
        "title": "Multiplelines No",
        "impact": -0.1870,
        "direction": "REDUCES_CHURN_RISK"
      },
      {
        "feature_key": "SeniorCitizen_0",
        "title": "Seniorcitizen 0",
        "impact": -0.1596,
        "direction": "REDUCES_CHURN_RISK"
      }
    ],
    "disclaimer": "Explanations represent model-derived statistical signals associated with churn risk, not causal guarantees. Retention actions should account for broader customer context."
  }
}
```

### `GET /health`
Returns operational service health, version, model load status, and timestamp.
```json
{
  "status": "healthy",
  "app_name": "Customer Churn Intelligence Platform",
  "version": "1.0.0",
  "model_loaded": true,
  "model_name": "logistic_regression_baseline",
  "timestamp": "2026-10-04T14:49:14.131355+00:00"
}
```

### `GET /model-info`
Returns active champion model name, training timestamp, primary metrics, and normalized global feature coefficients.

### `GET /metrics`
Returns benchmark scores, confusion matrices, and coordinates for ROC and Precision-Recall curves across all candidate models.

### `GET /customers?limit=60`
Returns real test population customer records pre-scored with real-time churn probability for portfolio exploration.

---

## 6. Frontend UI Sections

The web application is built with a premium **Antigravity AI product aesthetic**:

1. **Dashboard (`#dashboard`)**: Executive KPIs (total records, population churn rate, high-risk count, ROC-AUC), global feature importance chart, and actionable retention framework.
2. **Customer Prediction (`#prediction`)**: Interactive 19-parameter simulation form with preset profiles (`High Risk Fiber`, `Medium Risk Senior`, `Loyal Auto-Pay`), dynamic SVG radial risk gauge, and decision verdict.
3. **Customer Risk Analysis (`#risk-analysis`)**: Directory of customers with live search, risk filters (`All`, `High`, `Medium`, `Low`), and 1-click "Simulate" loading.
4. **Prediction Explanation (`#explanation`)**:
   - Active customer profile context banner.
   - Local SHAP Contribution Waterfall chart with crimson risk drivers and emerald retention drivers.
   - Interactive Counterfactual "What-If" Playbook (Upgrade Contract, Add Tech Support, Add Online Security, Switch to Auto-Pay) with instant recalculation.
5. **Model Performance (`#performance`)**: Candidate leaderboard comparison cards, interactive HTML5 Canvas ROC curve with diagonal baseline, Precision-Recall curve, and 2x2 confusion matrix.

---

## 7. Responsible AI & Explainability Notice

Model explanations provide statistical attribution signals based on historic training correlations. They are intended as decision-support insights for customer retention teams and do not constitute deterministic or causal guarantees.
