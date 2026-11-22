"""
False prediction analysis module for Healthy Artery.
Performs in-depth analysis of errors:
1. Confusion matrix plotting
2. Error identification (False Negatives & False Positives)
3. Model agreement and error overlap
4. Threshold sweeping (trading precision for recall)
5. Clinical feature profiling of errors vs true disease cases
6. Model interpretability (Logistic coefficients & SHAP values)
7. Calibration curve analysis
8. Subgroup performance (by sex and age categories)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.calibration import calibration_curve

from src.config import (
    NUMERIC_FEATURES, ALL_NUMERIC_FEATURES, TARGET
)
from src.data import load_data
from src.preprocess import split


def plot_confusion_matrices(models, X, y, output_dir="reports/figures"):
    """Plot confusion matrices for all models and save to disk."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.2))
    if len(models) == 1:
        axes = [axes]
        
    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Healthy", "Disease"],
                    yticklabels=["Healthy", "Disease"])
        ax.set_title(f"Confusion Matrix: {name.replace('_', ' ').title()}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def collect_errors(models, df, features, X_test, y_test):
    """
    Construct dataframe of all test rows with model predictions, probabilities,
    and error labels (FN or FP).
    """
    error_records = []
    indices = X_test.index
    
    for idx in indices:
        row_full = df.loc[idx]
        actual = int(y_test.loc[idx])
        record = {
            "index": idx,
            "actual": actual,
            **{f: row_full[f] for f in features if f in row_full}
        }
        
        # Add demographic info if present in df
        for col in ["age", "sex"]:
            if col in row_full and col not in record:
                record[col] = row_full[col]
                
        for name, model in models.items():
            prob = float(model.predict_proba(X_test.loc[[idx]])[:, 1][0])
            pred = int(prob >= 0.5)
            record[f"{name}_prob"] = prob
            record[f"{name}_pred"] = pred
            
            if actual == 1 and pred == 0:
                record[f"{name}_err"] = "FN"
            elif actual == 0 and pred == 1:
                record[f"{name}_err"] = "FP"
            else:
                record[f"{name}_err"] = "Correct"
                
        error_records.append(record)
        
    return pd.DataFrame(error_records)


def analyze_overlap(err_df, model_names):
    """
    Analyze overlap of errors across models.
    """
    err_cols = [f"{m}_err" for m in model_names]
    
    def get_misclassified_count(row):
        return sum(1 for col in err_cols if row[col] in ["FN", "FP"])
    
    err_df["error_count"] = err_df.apply(get_misclassified_count, axis=1)
    
    all_three = err_df[err_df["error_count"] == len(model_names)]
    two_models = err_df[err_df["error_count"] == 2]
    one_model = err_df[err_df["error_count"] == 1]
    all_correct = err_df[err_df["error_count"] == 0]
    
    summary = {
        "Total test patients": len(err_df),
        "Correct by all models": len(all_correct),
        "Misclassified by all 3": len(all_three),
        "Misclassified by exactly 2": len(two_models),
        "Misclassified by exactly 1": len(one_model),
    }
    return summary, err_df


def threshold_sweep(models, X, y, thresholds=np.arange(0.1, 0.95, 0.05),
                    output_dir="reports/figures"):
    """
    Evaluate precision, recall, FN, FP counts across thresholds for all models.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for name, model in models.items():
        proba = model.predict_proba(X)[:, 1]
        for t in thresholds:
            pred = (proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
            rec = recall_score(y, pred, zero_division=0)
            prec = precision_score(y, pred, zero_division=0)
            results.append({
                "model": name,
                "threshold": round(t, 2),
                "recall": rec,
                "precision": prec,
                "FN": fn,
                "FP": fp,
                "TP": tp,
                "TN": tn
            })
            
    df_sweep = pd.DataFrame(results)
    
    # Plot curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Recall vs Threshold
    for name in models.keys():
        m_df = df_sweep[df_sweep["model"] == name]
        axes[0].plot(m_df["threshold"], m_df["recall"], marker="o", label=f"{name} (Recall)")
        axes[0].plot(m_df["threshold"], m_df["precision"], linestyle="--", label=f"{name} (Precision)")
    axes[0].set_title("Precision & Recall vs. Decision Threshold")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Score")
    axes[0].axhline(0.90, color="red", linestyle=":", label="90% Recall Target")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    
    # FN and FP trade-off
    for name in models.keys():
        m_df = df_sweep[df_sweep["model"] == name]
        axes[1].plot(m_df["threshold"], m_df["FN"], marker="s", label=f"{name} FN (Missed Disease)")
        axes[1].plot(m_df["threshold"], m_df["FP"], linestyle="--", label=f"{name} FP (False Alarms)")
    axes[1].set_title("False Negatives vs False Positives by Threshold")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "threshold_sweep.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")
    
    return df_sweep


def profile_errors(err_df, model_name, features=NUMERIC_FEATURES):
    """
    Compare physiological feature means for FNs vs correctly identified Disease cases (TPs).
    """
    tp_mask = (err_df["actual"] == 1) & (err_df[f"{model_name}_pred"] == 1)
    fn_mask = (err_df["actual"] == 1) & (err_df[f"{model_name}_pred"] == 0)
    
    tps = err_df[tp_mask]
    fns = err_df[fn_mask]
    
    profile = {}
    for feat in features:
        if feat in err_df.columns:
            profile[feat] = {
                "Detected Disease (TP) Mean": tps[feat].mean() if len(tps) > 0 else np.nan,
                "Missed Disease (FN) Mean": fns[feat].mean() if len(fns) > 0 else np.nan,
                "Delta (FN - TP)": (fns[feat].mean() - tps[feat].mean()) if len(tps) > 0 and len(fns) > 0 else np.nan
            }
    return pd.DataFrame(profile).T


def plot_calibration(models, X, y, output_dir="reports/figures"):
    """Plot probability calibration curves."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    
    for name, model in models.items():
        proba = model.predict_proba(X)[:, 1]
        frac_pos, mean_pred = calibration_curve(y, proba, n_bins=5)
        plt.plot(mean_pred, frac_pos, "s-", label=name.replace('_', ' ').title())
        
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability / Calibration Curves (Healthy Artery)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    path = os.path.join(output_dir, "calibration_curves.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def analyze_subgroups(err_df, model_name):
    """
    Evaluate subgroup error rates across sex and age bands.
    """
    results = {}
    
    # Subgroup: Sex (0 = female, 1 = male in Cleveland dataset)
    if "sex" in err_df.columns:
        sex_groups = err_df.groupby("sex")
        for sex_val, group in sex_groups:
            label = "Female" if sex_val == 0 else "Male"
            total = len(group)
            fn = len(group[(group["actual"] == 1) & (group[f"{model_name}_pred"] == 0)])
            fp = len(group[(group["actual"] == 0) & (group[f"{model_name}_pred"] == 1)])
            err_rate = (fn + fp) / total if total > 0 else 0
            results[f"Sex: {label}"] = {"Total": total, "FN": fn, "FP": fp, "Error Rate": err_rate}
            
    # Subgroup: Age band (<55, >=55)
    if "age" in err_df.columns:
        err_df["age_band"] = pd.cut(
            err_df["age"], bins=[0, 54, 120], labels=["< 55 years", ">= 55 years"]
        )
        age_groups = err_df.groupby("age_band", observed=False)
        for band, group in age_groups:
            total = len(group)
            fn = len(group[(group["actual"] == 1) & (group[f"{model_name}_pred"] == 0)])
            fp = len(group[(group["actual"] == 0) & (group[f"{model_name}_pred"] == 1)])
            err_rate = (fn + fp) / total if total > 0 else 0
            results[f"Age: {band}"] = {"Total": total, "FN": fn, "FP": fp, "Error Rate": err_rate}
            
    return pd.DataFrame(results).T

