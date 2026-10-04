import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status

from config.settings import settings
from src.logger import logger
from src.exceptions import ModelNotLoadedError, InferenceError
from src.api.schemas import (
    CustomerInput,
    PredictionResponse,
    PredictionExplanation,
    DriverExplanation,
    BatchCustomerInput,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse
)
from src.models.explainer import ChurnExplainer
from src.models.pipeline import ChurnPipeline

router = APIRouter()

# Global pipeline and explainer singletons
_pipeline: Optional[ChurnPipeline] = None
_explainer: Optional[ChurnExplainer] = None
_metadata: Optional[Dict[str, Any]] = None
_comparison: Optional[Dict[str, Any]] = None

def get_pipeline() -> ChurnPipeline:
    """Retrieve or load cached inference pipeline."""
    global _pipeline, _explainer, _metadata, _comparison
    if _pipeline is not None:
        return _pipeline

    pipeline_path = settings.MODEL_ARTIFACTS_DIR / "best_churn_pipeline.joblib"
    metadata_path = settings.MODEL_ARTIFACTS_DIR / "model_metadata.json"
    comparison_path = settings.MODEL_ARTIFACTS_DIR / "all_models_comparison.json"

    if not pipeline_path.exists():
        logger.warning("Pipeline artifact not found. Triggering automated model training...")
        from src.models.trainer import train_all_models
        train_all_models()

    try:
        _pipeline = joblib.load(pipeline_path)
        _explainer = ChurnExplainer(_pipeline)
        
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                _metadata = json.load(f)

        if comparison_path.exists():
            with open(comparison_path, "r", encoding="utf-8") as f:
                _comparison = json.load(f)

        logger.info("Successfully loaded ChurnPipeline and SHAP Explainer into memory.")
        return _pipeline
    except Exception as e:
        logger.error(f"Failed to load model pipeline: {e}")
        raise ModelNotLoadedError(f"Model artifact could not be loaded: {e}")

def get_explainer() -> ChurnExplainer:
    global _explainer
    if _explainer is None:
        get_pipeline()
    return _explainer

def get_metadata() -> Dict[str, Any]:
    global _metadata
    if _metadata is None:
        get_pipeline()
    return _metadata or {}

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Return operational service health status and model availability."""
    try:
        pipeline = get_pipeline()
        meta = get_metadata()
        return HealthResponse(
            status="healthy",
            app_name=settings.APP_NAME,
            version=settings.APP_VERSION,
            model_loaded=pipeline is not None,
            model_name=meta.get("model_name", "churn_pipeline"),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    except Exception as e:
        logger.warning(f"Health check warning: {e}")
        return HealthResponse(
            status="degraded",
            app_name=settings.APP_NAME,
            version=settings.APP_VERSION,
            model_loaded=False,
            model_name=None,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

@router.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """Return model version, training timestamp, primary metrics, and feature importance."""
    meta = get_metadata()
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is unavailable. Please ensure model training has been completed."
        )
    return ModelInfoResponse(**meta)

@router.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_churn(customer: CustomerInput):
    """
    Predict churn probability, risk level, and local SHAP feature explanations for a single customer.
    """
    try:
        pipeline = get_pipeline()
        explainer = get_explainer()

        # Convert pydantic model to dataframe
        data_dict = customer.model_dump()
        if data_dict.get("TotalCharges") is None:
            # Impute TotalCharges = MonthlyCharges * tenure if not provided
            data_dict["TotalCharges"] = round(data_dict["MonthlyCharges"] * max(data_dict["tenure"], 1), 2)

        input_df = pd.DataFrame([data_dict])

        # Compute probability and binary decision
        proba = pipeline.predict_proba(input_df)
        churn_prob = float(proba[0, 1])
        prediction = int(churn_prob >= 0.5)
        risk_level = settings.get_risk_level(churn_prob)

        # Generate SHAP explanations
        explanation_raw = explainer.explain_instance(input_df, top_k=5)

        explanation = PredictionExplanation(
            top_risk_drivers=[
                DriverExplanation(**d) for d in explanation_raw["top_risk_drivers"]
            ],
            top_retention_drivers=[
                DriverExplanation(**d) for d in explanation_raw["top_retention_drivers"]
            ],
            disclaimer=explanation_raw["disclaimer"]
        )

        return PredictionResponse(
            churn_probability=round(churn_prob, 4),
            risk_level=risk_level,
            churn_prediction=prediction,
            explanation=explanation
        )
    except Exception as e:
        logger.error(f"Inference error in POST /predict: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_churn_batch(batch: BatchCustomerInput):
    """Predict churn for a batch of customers."""
    try:
        pipeline = get_pipeline()
        explainer = get_explainer()

        results = []
        for cust in batch.customers:
            data_dict = cust.model_dump()
            if data_dict.get("TotalCharges") is None:
                data_dict["TotalCharges"] = round(data_dict["MonthlyCharges"] * max(data_dict["tenure"], 1), 2)
            input_df = pd.DataFrame([data_dict])

            churn_prob = float(pipeline.predict_proba(input_df)[0, 1])
            risk_level = settings.get_risk_level(churn_prob)
            explanation_raw = explainer.explain_instance(input_df, top_k=3)

            results.append(PredictionResponse(
                churn_probability=round(churn_prob, 4),
                risk_level=risk_level,
                churn_prediction=int(churn_prob >= 0.5),
                explanation=PredictionExplanation(
                    top_risk_drivers=[DriverExplanation(**d) for d in explanation_raw["top_risk_drivers"]],
                    top_retention_drivers=[DriverExplanation(**d) for d in explanation_raw["top_retention_drivers"]],
                    disclaimer=explanation_raw["disclaimer"]
                )
            ))

        return BatchPredictionResponse(total_processed=len(results), predictions=results)
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", tags=["Model"])
def get_evaluation_metrics():
    """Return complete benchmark comparison across all candidate models (ROC, PR, Confusion Matrix)."""
    global _comparison
    if _comparison is None:
        get_pipeline()
    return _comparison or {}

@router.get("/customers", tags=["Data"])
def get_sample_customers(limit: int = Query(50, ge=1, le=200), search: Optional[str] = None):
    """
    Return real customer records from the test split with actual labels and pre-calculated risk score.
    Used for customer table risk explorer and quick demonstration.
    """
    test_csv = settings.DATA_PROCESSED_DIR / "test.csv"
    if not test_csv.exists():
        raw_csv = settings.DATA_RAW_PATH
        if not raw_csv.exists():
            raise HTTPException(status_code=404, detail="Processed test data not found.")
        df = pd.read_csv(raw_csv)
    else:
        df = pd.read_csv(test_csv)

    pipeline = get_pipeline()
    # Score the sample
    sample_df = df.copy()
    if search:
        s = search.lower()
        mask = (
            sample_df["customerID"].astype(str).str.lower().str.contains(s) |
            sample_df["Contract"].astype(str).str.lower().str.contains(s) |
            sample_df["PaymentMethod"].astype(str).str.lower().str.contains(s)
        )
        sample_df = sample_df[mask]

    sample_df = sample_df.head(limit)
    probs = pipeline.predict_proba(sample_df.drop(columns=["customerID", "Churn"], errors="ignore"))[:, 1]

    records = []
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        p = float(probs[idx])
        rec = row.to_dict()
        rec["predicted_churn_prob"] = round(p, 4)
        rec["risk_level"] = settings.get_risk_level(p)
        rec["actual_churn"] = int(row.get("Churn", 0)) if str(row.get("Churn", 0)).isdigit() else (1 if row.get("Churn") == "Yes" else 0)
        records.append(rec)

    # Sort descending by risk
    records.sort(key=lambda r: r["predicted_churn_prob"], reverse=True)
    return {"total": len(records), "customers": records}
