from dataclasses import dataclass
from typing import Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from config.settings import settings
from src.logger import logger
from src.data.validator import validate_raw_dataset

@dataclass
class DataBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean raw dataframe."""
    cleaned_df, report = validate_raw_dataset(df)
    
    # Map target column to binary numeric if present
    if "Churn" in cleaned_df.columns:
        if cleaned_df["Churn"].isin(["Yes", "No"]).any():
            cleaned_df["Churn"] = cleaned_df["Churn"].map({"Yes": 1, "No": 0})
        cleaned_df["Churn"] = pd.to_numeric(cleaned_df["Churn"], errors="coerce").fillna(0).astype(int)
        
    return cleaned_df

def stratified_split(df: pd.DataFrame, target_col: str, test_size: float, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a clean, deterministic stratified split on target_col using numpy and pandas.
    Avoids external dependencies while guaranteeing reproducible target proportions.
    """
    rng = np.random.default_rng(random_state)
    train_indices = []
    test_indices = []

    for label, group in df.groupby(target_col):
        n_total = len(group)
        n_test = int(np.round(n_total * test_size))
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        
        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])

    train_df = df.loc[train_indices].sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test_df = df.loc[test_indices].sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return train_df, test_df

def split_data(
    df: pd.DataFrame,
    test_size: float = None,
    val_size: float = None,
    random_state: int = None,
    save_dir: Optional[Path] = None
) -> DataBundle:
    """
    Split dataset into train, validation, and test sets using stratified sampling on Churn.
    Guarantees zero data leakage across splits.
    """
    test_sz = test_size if test_size is not None else settings.TEST_SIZE
    val_sz = val_size if val_size is not None else settings.VAL_SIZE
    seed = random_state if random_state is not None else settings.RANDOM_STATE

    if "Churn" not in df.columns:
        raise ValueError("Cannot perform stratified split without 'Churn' target column")

    # Step 1: Split off test set
    train_val_df, test_df = stratified_split(df, target_col="Churn", test_size=test_sz, random_state=seed)

    # Step 2: Split remaining into train and validation
    val_ratio_of_remaining = val_sz / (1.0 - test_sz)
    train_df, val_df = stratified_split(train_val_df, target_col="Churn", test_size=val_ratio_of_remaining, random_state=seed)

    logger.info(
        f"Stratified data split: Train={len(train_df)} ({len(train_df)/len(df):.1%}), "
        f"Val={len(val_df)} ({len(val_df)/len(df):.1%}), "
        f"Test={len(test_df)} ({len(test_df)/len(df):.1%})"
    )

    # Optionally persist processed splits
    out_dir = save_dir or settings.DATA_PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    logger.info(f"Saved processed train, val, test CSVs to {out_dir}")

    return DataBundle(train_df=train_df, val_df=val_df, test_df=test_df)
