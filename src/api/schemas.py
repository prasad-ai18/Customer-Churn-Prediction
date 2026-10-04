from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict

class CustomerInput(BaseModel):
    """Customer attributes schema matching IBM Telco benchmark with strict validation."""
    customerID: Optional[str] = Field(default="CUST-NEW", description="Optional customer ID")
    gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether customer is a senior citizen (1) or not (0)")
    Partner: Literal["Yes", "No"] = Field(..., description="Whether customer has a partner")
    Dependents: Literal["Yes", "No"] = Field(..., description="Whether customer has dependents")
    tenure: int = Field(..., ge=0, le=120, description="Months the customer has stayed with the company")
    PhoneService: Literal["Yes", "No"] = Field(..., description="Whether customer has phone service")
    MultipleLines: Literal["No phone service", "No", "Yes"] = Field(..., description="Whether customer has multiple lines")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., description="Customer internet service provider")
    OnlineSecurity: Literal["No internet service", "No", "Yes"] = Field(..., description="Whether customer has online security")
    OnlineBackup: Literal["No internet service", "No", "Yes"] = Field(..., description="Whether customer has online backup")
    DeviceProtection: Literal["No internet service", "No", "Yes"] = Field(..., description="Whether customer has device protection")
    TechSupport: Literal["No internet service", "No", "Yes"] = Field(..., description="Whether customer has tech support")
    StreamingTV: Literal["No internet service", "No", "Yes"] = Field(..., description="Whether customer has streaming TV")
    StreamingMovies: Literal["No internet service", "No", "Yes"] = Field(..., description="Whether customer has streaming movies")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., description="Contract term")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., description="Whether customer has paperless billing")
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ] = Field(..., description="Customer payment method")
    MonthlyCharges: float = Field(..., ge=0.0, le=500.0, description="Monthly charge amount")
    TotalCharges: Optional[float] = Field(default=None, ge=0.0, description="Total charges across entire tenure (calculated if omitted)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        }
    )


class DriverExplanation(BaseModel):
    feature_key: str
    title: str
    impact: float
    direction: str


class PredictionExplanation(BaseModel):
    top_risk_drivers: List[DriverExplanation]
    top_retention_drivers: List[DriverExplanation]
    disclaimer: str


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Probability of churn (0.0 to 1.0)")
    risk_level: str = Field(..., description="Risk category: Low, Medium, or High")
    churn_prediction: int = Field(..., description="Binary decision: 1 (Churn), 0 (Retain)")
    explanation: PredictionExplanation = Field(..., description="SHAP feature attribution signals")


class BatchCustomerInput(BaseModel):
    customers: List[CustomerInput]


class BatchPredictionResponse(BaseModel):
    total_processed: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    model_loaded: bool
    model_name: Optional[str] = None
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name: str
    trained_at: str
    app_version: str
    dataset: str
    train_samples: int
    val_samples: int
    test_samples: int
    class_imbalance_ratio: float
    primary_metrics: Dict[str, float]
    confusion_matrix: Dict[str, int]
    global_feature_importance: List[Dict[str, Any]]
    risk_thresholds: Dict[str, float]
