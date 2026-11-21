"""
Evaluation module for Healthy Artery.
Computes comprehensive held-out test set metrics (accuracy, precision, recall,
specificity, F1, ROC-AUC, PR-AUC, Brier score, TN/FP/FN/TP counts) and
performs Repeated Stratified 5x5 Cross-Validation (mean +/- std).
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    brier_score_loss, confusion_matrix
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from src.config import (
    RANDOM_STATE, N_SPLITS, N_REPEATS, NUMERIC_FEATURES, ALL_NUMERIC_FEATURES
)
from src.data import load_data
from src.preprocess import split


def evaluate(model, X, y, threshold=0.5):
    """
    Evaluate single model on a given dataset (e.g. test set).
    """
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "brier": brier_score_loss(y, proba),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def compare(models, X, y, threshold=0.5):
    """
    Evaluate multiple models and return a comparison DataFrame.
    """
    return pd.DataFrame({n: evaluate(m, X, y, threshold) for n, m in models.items()}).T


def repeated_cv_evaluate(models, X, y, n_splits=N_SPLITS, n_repeats=N_REPEATS):
    """
    Run 5x5 Repeated Stratified Cross-Validation on each model.
    Reports mean and standard deviation for key metrics.
    """
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE
    )
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    
    results = {}
    for name, model in models.items():
        cv_res = cross_validate(model, X, y, cv=rskf, scoring=scoring, n_jobs=-1)
        row = {}
        for metric in scoring.keys():
            mean_val = np.mean(cv_res[f"test_{metric}"])
            std_val = np.std(cv_res[f"test_{metric}"])
            row[f"{metric}_mean"] = mean_val
            row[f"{metric}_std"] = std_val
            row[f"{metric}"] = f"{mean_val:.3f} +/- {std_val:.3f}"
        results[name] = row
        
    return pd.DataFrame(results).T


def run_evaluation(features=NUMERIC_FEATURES, prefix=""):
    df = load_data()
    X_train, X_test, y_train, y_test = split(df, features)
    
    models = {}
    for name in ["logistic", "random_forest", "xgboost"]:
        model_name = f"{prefix}_{name}" if prefix else name
        path = f"models/{model_name}.joblib"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}. Run train.py first.")
        models[name] = joblib.load(path)
        
    print(f"\n==================================================")
    print(f"  HEALTHY ARTERY EVALUATION REPORT [{prefix if prefix else 'Core'}]")
    print(f"==================================================")
    
    # 1. Held-out test set evaluation
    print("\n--- Held-Out Test Set Performance ---")
    test_df = compare(models, X_test, y_test)
    print(test_df.round(4).to_string())
    
    # 2. Repeated Stratified 5x5 Cross-Validation
    print(f"\n--- Repeated Stratified {N_SPLITS}x{N_REPEATS} Cross-Validation (Full Dataset) ---")
    X_full, y_full = df[features], df["target"]
    cv_df = repeated_cv_evaluate(models, X_full, y_full)
    summary_cols = ["roc_auc", "recall", "precision", "f1", "accuracy", "pr_auc"]
    print(cv_df[summary_cols].to_string())
    
    return test_df, cv_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Healthy Artery prediction models.")
    parser.add_argument("--features", choices=["core", "all"], default="core")
    args = parser.parse_args()
    
    feat = ALL_NUMERIC_FEATURES if args.features == "all" else NUMERIC_FEATURES
    pref = "ext" if args.features == "all" else ""
    run_evaluation(features=feat, prefix=pref)

