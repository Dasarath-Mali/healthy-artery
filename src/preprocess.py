"""
Preprocessing module for Healthy Artery.
Provides train/test splitting and pipeline transformation steps
for scaled (linear) and unscaled (tree-based) models.
"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE, TEST_SIZE, TARGET


def split(df, features):
    """
    Split the dataset into stratified training and testing sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset containing features and TARGET.
    features : list
        List of feature column names to extract.
        
    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X, y = df[features], df[TARGET]
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )


def scaled_steps():
    """
    Preprocessing steps for scale-sensitive models (Logistic Regression):
    Median imputation followed by standard scaling.
    """
    return [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]


def unscaled_steps():
    """
    Preprocessing steps for tree-based models (Random Forest, XGBoost):
    Median imputation without scaling (tree splits are scale-invariant).
    """
    return [
        ("impute", SimpleImputer(strategy="median"))
    ]

