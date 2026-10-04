import httpx
import pandas as pd
from pathlib import Path
from config.settings import settings
from src.logger import logger
from src.exceptions import DataLoadError

def ensure_raw_data(destination: Path = None, url: str = None) -> Path:
    """Ensure the raw IBM Telco Customer Churn dataset exists locally; download if missing."""
    dest_path = destination or settings.DATA_RAW_PATH
    source_url = url or settings.DATA_RAW_URL

    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info(f"Raw dataset already exists at: {dest_path}")
        return dest_path

    logger.info(f"Downloading IBM Telco dataset from {source_url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(source_url)
            response.raise_for_status()
            dest_path.write_bytes(response.content)
            logger.info(f"Successfully downloaded dataset ({len(response.content):,} bytes) to {dest_path}")
            return dest_path
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise DataLoadError(f"Could not retrieve raw dataset from {source_url}: {e}")

def load_raw_data(file_path: Path = None) -> pd.DataFrame:
    """Load the raw Telco customer churn dataset into a pandas DataFrame."""
    path = file_path or ensure_raw_data()
    if not Path(path).exists():
        raise DataLoadError(f"Raw data file not found at {path}")

    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded raw dataset from {path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading raw CSV from {path}: {e}")
        raise DataLoadError(f"Failed to read CSV at {path}: {e}")
