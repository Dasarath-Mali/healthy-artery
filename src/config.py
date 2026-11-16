"""
Configuration module for Healthy Artery.
Defines seeds, paths, column definitions, and feature subsets.
"""

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
N_REPEATS = 5

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

# Core experiment: continuous numeric features only
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Extension experiment: all numeric-coded columns (13 features)
ALL_NUMERIC_FEATURES = [c for c in COLUMNS if c != "num"]

TARGET = "target"
RAW_PATH = "data/raw/processed.cleveland.data"
PROCESSED_PATH = "data/processed/cleveland_cleaned.csv"

