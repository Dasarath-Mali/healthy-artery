"""
Data ingestion and cleaning module for Healthy Artery.
Loads raw UCI Cleveland dataset, applies clinical validity filtering,
binarizes the target, and handles missing markers.
"""

import numpy as np
import pandas as pd
from src.config import COLUMNS, RAW_PATH, TARGET


def load_data(path=RAW_PATH):
    """
    Load and clean the Cleveland Heart Disease dataset.
    
    Cleaning rules:
    - Replace '?' with NaN
    - Target binarization: 0 = healthy (num == 0), 1 = heart disease (num > 0)
    - Set physiologically impossible values (chol == 0, trestbps == 0) to NaN
    - Remove duplicate rows
    
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with feature columns and binary TARGET column.
    """
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    df[TARGET] = (df["num"] > 0).astype(int)  # 0 = healthy, 1 = disease
    df = df.drop(columns=["num"])
    df.loc[df["chol"] == 0, "chol"] = np.nan  # zero cholesterol is physiologically impossible / missing
    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan
    return df.drop_duplicates().reset_index(drop=True)

