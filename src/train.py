"""
Model training and hyperparameter tuning for Healthy Artery.
Performs Stratified 5-Fold cross-validation grid search and persists
trained best estimators.
"""

import os
import argparse
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.config import RANDOM_STATE, N_SPLITS, NUMERIC_FEATURES, ALL_NUMERIC_FEATURES
from src.data import load_data
from src.preprocess import split
from src.models import logistic, random_forest, xgboost


def tune(pipe, grid, X_train, y_train, scoring="roc_auc"):
    """
    Tune model hyperparameters using Stratified K-Fold CV.
    
    Parameters
    ----------
    pipe : Pipeline
    grid : dict
    X_train : pd.DataFrame or np.ndarray
    y_train : pd.Series or np.ndarray
    scoring : str, default='roc_auc'
    
    Returns
    -------
    GridSearchCV
        Fitted grid search object with best_estimator_.
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipe, grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    return search


def save_model(model, name, folder="models"):
    """Persist trained model to disk."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"Saved: {path}")


def train_all(features=NUMERIC_FEATURES, prefix="", scoring="roc_auc"):
    """
    Load data, split, tune all 3 models, print results, and save estimators.
    """
    print(f"\n--- Training Healthy Artery Models [{prefix if prefix else 'core'}] ---")
    print(f"Features ({len(features)}): {features}")
    print(f"Scoring metric for CV: {scoring}")
    
    df = load_data()
    X_train, X_test, y_train, y_test = split(df, features)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train class balance: {dict(y_train.value_counts())}")
    
    models = {
        "logistic": logistic(),
        "random_forest": random_forest(),
        "xgboost": xgboost()
    }
    
    fitted_models = {}
    for name, (pipe, grid) in models.items():
        print(f"\nTuning {name}...")
        search = tune(pipe, grid, X_train, y_train, scoring=scoring)
        print(f"  Best CV {scoring}: {search.best_score_:.4f}")
        print(f"  Best params: {search.best_params_}")
        
        save_name = f"{prefix}_{name}" if prefix else name
        save_model(search.best_estimator_, save_name)
        fitted_models[name] = search.best_estimator_
        
    return fitted_models, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Healthy Artery prediction models.")
    parser.add_argument("--features", choices=["core", "all"], default="core",
                        help="Feature set to use: core (5 numeric) or all (13 features)")
    parser.add_argument("--scoring", default="roc_auc", help="CV scoring metric (roc_auc, recall, etc.)")
    args = parser.parse_args()
    
    feat = ALL_NUMERIC_FEATURES if args.features == "all" else NUMERIC_FEATURES
    pref = "ext" if args.features == "all" else ""
    train_all(features=feat, prefix=pref, scoring=args.scoring)

