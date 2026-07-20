"""
Baseline classifiers for comparison with the Geometry-Aware Fisher Kernel.

Matches the thesis notebook protocol:
- 5-fold stratified CV (random_state=42)
- Logistic Regression with StandardScaler
- Random Forest and XGBoost on encoded mixed-type features
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency
    xgb = None


@dataclass
class BaselineFoldResult:
    model: str
    fold: int
    accuracy: float
    macro_f1: float
    auc: float


@dataclass
class BaselineSummary:
    model: str
    mean_accuracy: float
    std_accuracy: float
    mean_macro_f1: float
    std_macro_f1: float
    mean_auc: float
    std_auc: float


@dataclass
class BaselineCVResult:
    fold_results: List[BaselineFoldResult]
    summaries: List[BaselineSummary]


def _default_models() -> Dict[str, object]:
    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=1000, random_state=42),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
    }

    if xgb is not None:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )

    return models


def run_baseline_cv(
    X: np.ndarray,
    y: np.ndarray,
    variable_names: Optional[List[str]] = None,
    outer_splits: int = 5,
    random_state: int = 42,
    models: Optional[Dict[str, object]] = None,
    verbose: bool = True,
) -> BaselineCVResult:
    """
    Evaluate standard baselines with the same CV protocol as the thesis.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)

    if variable_names is None:
        variable_names = [f"x{i}" for i in range(X.shape[1])]

    X_df = pd.DataFrame(X, columns=variable_names)
    y_series = pd.Series(y, name="target")

    cv = StratifiedKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=random_state,
    )

    models = models or _default_models()
    fold_results: List[BaselineFoldResult] = []
    summaries: List[BaselineSummary] = []

    for model_name, model in models.items():
        if verbose:
            print(f"\n=== Evaluating {model_name} ===")

        accs: List[float] = []
        f1s: List[float] = []
        aucs: List[float] = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_df, y_series)):
            X_train = X_df.iloc[train_idx]
            X_test = X_df.iloc[test_idx]
            y_train = y_series.iloc[train_idx]
            y_test = y_series.iloc[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba")
                else None
            )

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

            accs.append(acc)
            f1s.append(f1)
            if not np.isnan(auc):
                aucs.append(auc)

            fold_results.append(
                BaselineFoldResult(
                    model=model_name,
                    fold=fold_idx,
                    accuracy=acc,
                    macro_f1=f1,
                    auc=auc,
                )
            )

            if verbose:
                print(
                    f"Fold {fold_idx}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
                )

        summary = BaselineSummary(
            model=model_name,
            mean_accuracy=float(np.mean(accs)),
            std_accuracy=float(np.std(accs)),
            mean_macro_f1=float(np.mean(f1s)),
            std_macro_f1=float(np.std(f1s)),
            mean_auc=float(np.mean(aucs)) if aucs else np.nan,
            std_auc=float(np.std(aucs)) if aucs else np.nan,
        )
        summaries.append(summary)

        if verbose:
            print(
                f"{model_name} Mean: "
                f"Acc={summary.mean_accuracy:.4f}±{summary.std_accuracy:.4f} | "
                f"F1={summary.mean_macro_f1:.4f}±{summary.std_macro_f1:.4f} | "
                f"AUC={summary.mean_auc:.4f}±{summary.std_auc:.4f}"
            )

    if verbose:
        print("\n" + "=" * 50)
        print("BASELINE CV SUMMARY")
        print("=" * 50)
        for summary in summaries:
            print(
                f"{summary.model:22s}  "
                f"Acc={summary.mean_accuracy:.3f}±{summary.std_accuracy:.3f}  "
                f"F1={summary.mean_macro_f1:.3f}±{summary.std_macro_f1:.3f}  "
                f"AUC={summary.mean_auc:.3f}±{summary.std_auc:.3f}"
            )
        print("=" * 50)

    return BaselineCVResult(fold_results=fold_results, summaries=summaries)


def summaries_to_dataframe(result: BaselineCVResult) -> pd.DataFrame:
    rows = []
    for summary in result.summaries:
        rows.append(
            {
                "Model": summary.model,
                "Accuracy_mean": summary.mean_accuracy,
                "Accuracy_std": summary.std_accuracy,
                "F1_mean": summary.mean_macro_f1,
                "F1_std": summary.std_macro_f1,
                "ROC-AUC_mean": summary.mean_auc,
                "ROC-AUC_std": summary.std_auc,
            }
        )
    return pd.DataFrame(rows)
