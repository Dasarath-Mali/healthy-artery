# BUILDME: Heart Disease Prediction System

Step-by-step instructions to build the project from scratch. Follow the steps in order. Each step ends with a **Done when** check.

## 0. Prerequisites

- Python 3.10 or newer
- pip and venv
- Jupyter (optional, for notebooks)

## 1. Set up the project

```bash
mkdir heart-disease-prediction && cd heart-disease-prediction
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

mkdir -p data/raw data/processed notebooks src reports/figures models
touch src/__init__.py
```

Create `requirements.txt`:

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
shap
joblib
jupyter
```

```bash
pip install -r requirements.txt
```

**Done when:** `python -c "import sklearn, xgboost, pandas"` runs without errors.

## 2. Get the data

Download the processed Cleveland file from the UCI Machine Learning Repository (Heart Disease dataset) and save it as `data/raw/processed.cleveland.data`.

Column names, in order:

```
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num
```

**Done when:** the file exists in `data/raw/` and has about 303 rows.

## 3. Write `src/config.py`

```python
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
N_REPEATS = 5

COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

# Core experiment: continuous numeric features only
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Extension experiment: all numeric-coded columns
ALL_NUMERIC_FEATURES = [c for c in COLUMNS if c != "num"]

TARGET = "target"
RAW_PATH = "data/raw/processed.cleveland.data"
```

## 4. Write `src/data.py`

Responsibilities: load, clean, binarize the target.

```python
import numpy as np
import pandas as pd
from src.config import COLUMNS, RAW_PATH, TARGET

def load_data(path=RAW_PATH):
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    df[TARGET] = (df["num"] > 0).astype(int)   # 0 = healthy, 1 = disease
    df = df.drop(columns=["num"])
    df.loc[df["chol"] == 0, "chol"] = np.nan   # zero cholesterol means missing
    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan
    return df.drop_duplicates().reset_index(drop=True)
```

**Done when:** `load_data().shape` returns roughly (302, 14) and `df[TARGET].value_counts()` shows both classes.

## 5. Explore the data (notebook `01_eda.ipynb`)

- `df.describe()` and `df.isna().sum()`
- Class balance bar chart
- Histograms of each numeric feature, split by target
- Correlation heatmap
- Boxplots to spot outliers

Save key figures to `reports/figures/`.

**Done when:** you can state the class balance and which features differ most between classes.

## 6. Write `src/preprocess.py`

Use pipelines so the imputer and scaler are fitted on training data only.

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE, TEST_SIZE, TARGET

def split(df, features):
    X, y = df[features], df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE,
                            stratify=y, random_state=RANDOM_STATE)

def scaled_steps():
    return [("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())]

def unscaled_steps():
    return [("impute", SimpleImputer(strategy="median"))]
```

**Done when:** train and test shapes look right and class ratios match in both.

## 7. Write `src/models.py`

One function per model, each returning a `(pipeline, param_grid)` pair.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from src.config import RANDOM_STATE
from src.preprocess import scaled_steps, unscaled_steps

def logistic():
    pipe = Pipeline(scaled_steps() + [("clf", LogisticRegression(
        max_iter=2000, random_state=RANDOM_STATE))])
    grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    return pipe, grid

def random_forest():
    pipe = Pipeline(unscaled_steps() + [("clf", RandomForestClassifier(
        random_state=RANDOM_STATE))])
    grid = {"clf__n_estimators": [200, 500],
            "clf__max_depth": [3, 5, 8, None],
            "clf__min_samples_leaf": [1, 3, 5],
            "clf__max_features": ["sqrt", 0.5]}
    return pipe, grid

def xgboost():
    pipe = Pipeline(unscaled_steps() + [("clf", XGBClassifier(
        eval_metric="logloss", random_state=RANDOM_STATE))])
    grid = {"clf__n_estimators": [100, 300],
            "clf__learning_rate": [0.03, 0.1],
            "clf__max_depth": [2, 3, 5],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
            "clf__reg_lambda": [1, 5]}
    return pipe, grid
```

**Done when:** all three functions return a pipeline that fits on the training set without errors.

## 8. Write `src/train.py`

Tune each model with stratified cross-validation, scored on **recall** (or ROC-AUC if you prefer), then save the best estimator.

```python
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.config import RANDOM_STATE, N_SPLITS

def tune(pipe, grid, X_train, y_train, scoring="roc_auc"):
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipe, grid, cv=cv, scoring=scoring, n_jobs=-1)
    search.fit(X_train, y_train)
    return search

def save(model, name):
    joblib.dump(model, f"models/{name}.joblib")
```

Run all three and save them as `logistic`, `random_forest`, `xgboost`.

**Done when:** three `.joblib` files exist in `models/` and best CV scores are printed.

## 9. Write `src/evaluate.py`

For each model compute on the **test set**:

- Accuracy, precision, recall, specificity, F1
- ROC-AUC, PR-AUC
- Brier score
- Confusion matrix (TN, FP, FN, TP)

```python
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             brier_score_loss, confusion_matrix)

def evaluate(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "brier": brier_score_loss(y, proba),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }

def compare(models, X, y):
    return pd.DataFrame({n: evaluate(m, X, y) for n, m in models.items()}).T
```

Because the test set is tiny, also run **repeated stratified 5x5 cross-validation** with `cross_validate` on all three models and report mean and standard deviation.

**Done when:** you have one comparison table with all three models and mean plus std from repeated CV.

## 10. False prediction analysis (`src/errors.py` and notebook `03_errors.ipynb`)

1. **Confusion matrices:** plot one per model.
2. **Collect errors:** for each model, build a DataFrame of test rows where prediction differs from truth, labelled FN or FP, with predicted probability attached.
3. **Overlap:** find which patients are misclassified by all three models, by two, and by only one.
4. **Threshold sweep:** for thresholds 0.1 to 0.9, record recall, precision and FN/FP counts. Plot them. Pick the threshold that reaches your target recall (for example 0.90) and report the extra false positives it costs.
5. **Profile the errors:** compare mean age, blood pressure, cholesterol, max heart rate and oldpeak for FNs vs correctly detected disease cases.
6. **Explain a few cases:** use SHAP on the tree models and coefficients on logistic regression for 3 to 5 FN and FP examples.
7. **Calibration:** plot calibration curves and check whether wrong predictions were made with high confidence.
8. **Subgroups:** compute error rates by sex and age band.

**Done when:** you can write one paragraph per model describing what kind of patients it gets wrong and why.

## 11. Run the extension experiment

Change `NUMERIC_FEATURES` to `ALL_NUMERIC_FEATURES`, rerun steps 6 to 10, and compare results with the 5-feature version. This shows how much the numeric-only constraint costs.

## 12. Produce the report

Write `reports/report.md` containing:

1. Problem and dataset summary
2. Method (preprocessing, models, tuning, evaluation)
3. Results table (all three models, with std from repeated CV)
4. ROC and PR curves
5. False prediction analysis
6. Recommendation: which model, which threshold
7. Limitations (small data, single source, not for clinical use)

## 13. Add a README

Include: project purpose, setup commands, how to run each step, folder layout, and a link to the report.

## 14. Desktop app

Launch the CustomTkinter desktop graphical user interface for interactive predictions and model comparison:

```bash
python app.py
```

**Done when:** the desktop window opens with the title "Heart disease risk predictor", lets you enter patient parameters, select between models, compute risk probability with colored badges, and browse model comparison benchmarks and saved charts.

## 15. Final checklist

- [ ] Imputer and scaler fitted only on training data
- [ ] Same split and CV folds for all three models
- [ ] Random seed fixed everywhere
- [ ] Metrics reported with variance, not a single number
- [ ] FN and FP counts reported for every model
- [ ] Threshold trade-off documented
- [ ] Figures saved in `reports/figures/`
- [ ] Disclaimer that this is not a clinical tool
- [ ] Desktop app running and validated

## Suggested run order

```bash
python -m src.train        # tune and save models
python -m src.evaluate     # build comparison table
python app.py              # launch desktop application
jupyter notebook           # open 03_errors.ipynb for error analysis
```
