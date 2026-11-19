"""
Model architecture definitions for Healthy Artery.
Configures pipelines and hyperparameter search grids for:
- Logistic Regression (Baseline)
- Random Forest
- XGBoost
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_STATE
from src.preprocess import scaled_steps, unscaled_steps


def logistic():
    """Build Logistic Regression pipeline and parameter grid."""
    pipe = Pipeline(scaled_steps() + [
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    ])
    grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    return pipe, grid


def random_forest():
    """Build Random Forest pipeline and parameter grid."""
    pipe = Pipeline(unscaled_steps() + [
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE))
    ])
    grid = {
        "clf__n_estimators": [200, 500],
        "clf__max_depth": [3, 5, 8, None],
        "clf__min_samples_leaf": [1, 3, 5],
        "clf__max_features": ["sqrt", 0.5]
    }
    return pipe, grid


def xgboost():
    """Build XGBoost pipeline and parameter grid."""
    pipe = Pipeline(unscaled_steps() + [
        ("clf", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE))
    ])
    grid = {
        "clf__n_estimators": [100, 300],
        "clf__learning_rate": [0.03, 0.1],
        "clf__max_depth": [2, 3, 5],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
        "clf__reg_lambda": [1, 5]
    }
    return pipe, grid


def get_all_models():
    """Return dictionary of (pipeline, param_grid) for all models."""
    return {
        "logistic": logistic(),
        "random_forest": random_forest(),
        "xgboost": xgboost()
    }

