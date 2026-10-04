from pathlib import Path
from typing import List, Union
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # Application Config
    APP_NAME: str = "Customer Churn Intelligence Platform"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    APP_HOST: str = "127.0.0.1"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = True

    # Directories
    BASE_DIR: Path = BASE_DIR
    DATA_RAW_URL: str = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    DATA_RAW_PATH: Path = BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    MODEL_ARTIFACTS_DIR: Path = BASE_DIR / "artifacts" / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # Risk Score Thresholds
    LOW_RISK_MAX: float = 0.35
    HIGH_RISK_MIN: float = 0.65

    # CORS
    CORS_ORIGINS: Union[List[str], str] = ["*"]

    # Model Parameters
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    VAL_SIZE: float = 0.15

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def ensure_directories(self) -> None:
        """Create necessary project directories if they do not exist."""
        self.DATA_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    def get_risk_level(self, probability: float) -> str:
        """Categorize churn probability into business risk tier."""
        if probability < self.LOW_RISK_MAX:
            return "Low"
        elif probability < self.HIGH_RISK_MIN:
            return "Medium"
        else:
            return "High"

settings = Settings()
settings.ensure_directories()
