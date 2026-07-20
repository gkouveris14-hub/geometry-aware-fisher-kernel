"""
Nested cross-validation for the Geometry-Aware Fisher Kernel.
Follows the revised experimental protocol (no leakage).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .pipeline import GeometryFisherClassifier
from .structure import StructuralMask


@dataclass
class FoldResult:
    accuracy: float
    macro_f1: float
    auc: float
    n_params: int


@dataclass
class NestedCVResult:
    fold_results: List[FoldResult]
    mean_accuracy: float
    std_accuracy: float
    mean_macro_f1: float
    std_macro_f1: float
    mean_auc: float
    std_auc: float


class NestedCVExperiment:
    """
    Nested cross-validation experiment.

    Parameters
    ----------
    mask : str or StructuralMask
        "hand", "data_driven", or a StructuralMask instance.
    mask_params : dict
        Parameters for data-driven mask (alpha, tau_stab, B, exogenous...).
    feature_type : str
        "linear", "quadratic", or "full".
    outer_splits : int
        Number of outer folds (default 5).
    random_state : int
    """

    def __init__(
        self,
        mask: Union[str, StructuralMask] = "hand",
        mask_object: Optional[StructuralMask] = None,
        mask_params: Optional[Dict[str, Any]] = None,
        feature_type: str = "linear",
        lambda_reg: float = 0.01,
        ridge_gamma: float = 1e-4,
        C: float = 1.0,
        outer_splits: int = 5,
        random_state: int = 42,
    ):
        self.mask = mask
        self.mask_object = mask_object
        self.mask_params = mask_params or {}
        self.feature_type = feature_type
        self.lambda_reg = lambda_reg
        self.ridge_gamma = ridge_gamma
        self.C = C
        self.outer_splits = outer_splits
        self.random_state = random_state

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        continuous_idx: np.ndarray,
        ordinal_idx: np.ndarray,
        variable_names: Optional[List[str]] = None,
    ) -> NestedCVResult:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)

        outer_cv = StratifiedKFold(
            n_splits=self.outer_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            print(f"\n=== Outer Fold {fold_idx + 1}/{self.outer_splits} ===")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Create a fresh classifier for this fold
            clf = GeometryFisherClassifier(
                mask=self.mask,
                mask_object=self.mask_object,
                mask_params=self.mask_params,
                lambda_reg=self.lambda_reg,
                ridge_gamma=self.ridge_gamma,
                feature_type=self.feature_type,
                C=self.C,
            )

            # Fit only on the training fold
            clf.fit(
                X_train, y_train,
                continuous_idx=continuous_idx,
                ordinal_idx=ordinal_idx,
                variable_names=variable_names,
            )

            # Predict on the held-out test fold
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            try:
                auc = roc_auc_score(y_test, y_proba)
            except ValueError:
                auc = np.nan

            n_params = clf.mask_.n_params if clf.mask_ is not None else -1

            print(f"Fold {fold_idx + 1} — Acc: {acc:.3f} | Macro-F1: {f1:.3f} | AUC: {auc:.3f} | n_params: {n_params}")

            fold_results.append(FoldResult(
                accuracy=acc,
                macro_f1=f1,
                auc=auc,
                n_params=n_params,
            ))

        # Aggregate
        accs = [r.accuracy for r in fold_results]
        f1s = [r.macro_f1 for r in fold_results]
        aucs = [r.auc for r in fold_results if not np.isnan(r.auc)]

        result = NestedCVResult(
            fold_results=fold_results,
            mean_accuracy=float(np.mean(accs)),
            std_accuracy=float(np.std(accs)),
            mean_macro_f1=float(np.mean(f1s)),
            std_macro_f1=float(np.std(f1s)),
            mean_auc=float(np.mean(aucs)) if aucs else np.nan,
            std_auc=float(np.std(aucs)) if aucs else np.nan,
        )

        print("\n" + "=" * 50)
        print("NESTED CV SUMMARY")
        print("=" * 50)
        print(f"Accuracy:  {result.mean_accuracy:.3f} ± {result.std_accuracy:.3f}")
        print(f"Macro-F1:  {result.mean_macro_f1:.3f} ± {result.std_macro_f1:.3f}")
        print(f"AUC:       {result.mean_auc:.3f} ± {result.std_auc:.3f}")
        print("=" * 50)

        return result