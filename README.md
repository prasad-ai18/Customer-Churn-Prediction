# Customer Churn Intelligence & Prediction Platform

A production-style machine learning platform that predicts whether a customer is likely to churn, determines their risk level (Low, Medium, High), and provides interpretable SHAP feature drivers behind every prediction.

Built strictly according to **SOP Project 01 — Customer Churn Intelligence & Prediction Platform**.

---

## 1. System Architecture & Flow

```
Customer Data (IBM Telco)
       │
       ▼
Data Ingestion & Schema Validation (Missing value handling, whitespace cleanup, category validation)
       │
       ▼
Feature Engineering & Transformation (Tenure cohorts, charge ratios, active services, friction profiles)
       │
       ▼
Class-Imbalance Handling (Balanced sample weighting & scale_pos_weight = 2.77)
       │
       ▼
Model Training & Benchmarking (Logistic Regression Baseline, Random Forest, XGBoost)
       │
       ▼
Model Evaluation & Selection (Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix, Slices)
       │
       ▼
Artifact Persistence (Unified ChurnPipeline, model_metadata.json, all_models_comparison.json)
       │
       ▼
FastAPI REST API (/predict, /health, /model-info, /predict/batch, /customers, /metrics)
       │
       ▼
Modern AI Web Dashboard (Interactive Simulator, SHAP Drivers, Risk Explorer, Canvas Curves)
```

---

## 2. Technology Stack

- **Machine Learning Core**: Python, NumPy, Pandas, XGBoost, Scikit-learn
- **Model Explainability**: Exact Tree SHAP attribution signals and linear feature drivers
- **REST API Serving**: FastAPI, Uvicorn, Pydantic v2, Pydantic-Settings
- **Frontend Application**: HTML5, Vanilla CSS3 (Antigravity Modern AI Dark Aesthetic, Glassmorphism), Modern Vanilla JavaScript, HTML5 Canvas charting
- **Artifact Persistence**: Joblib, JSON
- **Configuration & Environment**: `.env`, `.env.example`
- **Testing**: PyTest, FastApi TestClient, HTTPX

---

## 3. Dataset & Data Cleaning

- **Dataset**: IBM Telco Customer Churn (7,043 rows, 21 columns).
- **Whitespace / Zero-Tenure Correction**: Fixed Telco data quirk where `TotalCharges` is `" "` for 11 customers with `tenure == 0`, assigning them `0.0`.
- **Leakage-Free Splitting**: Stratified split on `Churn` into **Train (65%)**, **Validation (15%)**, and **Test (20%)**.
- **Domain Feature Engineering**:
  - `tenure_cohort`: Binned lifecycle stages (`0-12m`, `13-24m`, `25-48m`, `49-72m`).
  - `charge_ratio`: `MonthlyCharges / (TotalCharges + 1.0)` capturing disproportionate early financial commitment.
  - `bill_shock_indicator`: Difference between current monthly charge and historical average spend.
  - `total_active_services`: Count of active sticky services (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`).
  - `fiber_without_tech_support`: High-risk friction segment (Fiber optic with no tech support).
  - `high_risk_contract_payment`: Month-to-month contract paired with electronic check.
  - `is_full_streamer`: Dual subscription to Streaming TV & Movies.

---

## 4. Model Benchmarks & Evaluation

All metrics computed on the strictly held-out test set (1,409 customer records):

| Model Candidate | ROC-AUC | PR-AUC (Avg Precision) | F1-Score | Recall (Sensitivity) | Precision | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression Baseline** (★ Champion) | **0.8464** | **0.6753** | **0.6265** | **79.14%** | 51.84% | 74.95% |
| **XGBoost Boosted Ensemble** | 0.8429 | 0.6717 | 0.6100 | 78.61% | 49.83% | 73.67% |
| **Random Forest (Parallel Trees)** | 0.8405 | 0.6566 | 0.5957 | 55.35% | 64.49% | 78.78% |

### Champion Model Confusion Matrix (1,409 Test Samples)
- **True Positives (Churn Detected)**: 296
- **False Positives (False Alarms)**: 275
- **False Negatives (Missed Churn)**: 78
- **True Negatives (Retained Correctly)**: 760

> **Business Objective**: Capturing ~79.1% of all churning customers while maintaining 0.675 PR-AUC provides retention teams with early, reliable intervention triggers.

---

## 5. Model Explainability (SHAP)

Every individual prediction generates local SHAP feature attributions:
- **Positive Signals (+)**: Specific factors driving churn probability upward (e.g. Month-to-Month Contract, Low Tenure, High Bill Shock).
- **Negative Signals (-)**: Specific retention drivers pushing probability downward (e.g. Two-Year Contract, Active Tech Support, Automatic Credit Card Payment).
- **Business Translation**: Technical one-hot columns are translated into actionable, plain-English customer relationship insights.
- **Compliance Disclaimer**: Presented as model-derived statistical signals, not causal guarantees.

---

## 6. FastAPI Service Endpoints

The API is served at `http://127.0.0.1:8000`:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/predict` | Returns churn probability, risk level (`Low`, `Medium`, `High`), decision, and SHAP drivers |
| `POST` | `/predict/batch` | Batch scoring for multiple customer records |
| `GET` | `/health` | Service health status, app name, version, and model load status |
| `GET` | `/model-info` | Model metadata, training timestamp, primary metrics, and global feature importance |
| `GET` | `/metrics` | Out-of-sample benchmark metrics, confusion matrix, ROC & PR curve coordinates |
| `GET` | `/customers` | Sample customer records from test set with live risk rankings and search support |
| `GET` | `/docs` | Interactive Swagger API documentation |

---

## 7. Web Application Features

Access the web interface at `http://127.0.0.1:8000/`:
1. **Executive Dashboard**: Key metrics (total records, average predicted churn rate, high-risk count, ROC-AUC), global feature importance chart, and retention playbook.
2. **Prediction Simulator**: Customer input form with preset buttons (`High Risk Fiber`, `Medium Risk Senior`, `Loyal Auto-Pay`), real-time radial probability gauge, and SHAP waterfall driver list.
3. **Customer Risk Explorer**: Real-time searchable directory of test customers filterable by risk level (`High`, `Medium`, `Low`) with 1-click "Simulate" loading.
4. **Model Performance & Benchmarking**: Real out-of-sample candidate comparisons, interactive HTML5 Canvas ROC curve, Precision-Recall curve, and confusion matrix grid.

---

## 8. Installation & Quickstart

### Prerequisites
- Python 3.11+ / 3.14

### Setup Environment
```bash
# 1. Clone repository & navigate to directory
cd "Customer Churn Prediction"

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Copy environment configuration
copy .env.example .env
```

### Train Models & Generate Artifacts
```bash
python -m src.models.trainer
```

### Run Automated Test Suite
```bash
python -m pytest tests/ -v
```

### Launch FastAPI Server & Web App
```bash
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```
Open **`http://127.0.0.1:8000/`** in your browser.
Open **`http://127.0.0.1:8000/docs`** for interactive Swagger API documentation.
