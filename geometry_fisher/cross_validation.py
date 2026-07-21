"""
K-fold cross-validation for the geometry-aware classifier.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Optional, Union, Literal
from dataclasses import dataclass
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .pipeline import GeometryFisherClassifier
from .structure import StructuralMask, discover_data_driven_mask


@dataclass
class FoldResult:
    accuracy: float
    macro_f1: float
    auc: float
    n_params: int


@dataclass
class CrossValidationResult:
    fold_results: List[FoldResult]
    mean_accuracy: float
    std_accuracy: float
    mean_macro_f1: float
    std_macro_f1: float
    mean_auc: float
    std_auc: float
    fixed_mask: Optional[StructuralMask] = None


class CrossValidationExperiment:
    """
    Stratified k-fold cross-validation for GeometryFisherClassifier.

    Parameters
    ----------
    mask : str or StructuralMask
        ``hand``, ``pc``, ``stability``, or a ``StructuralMask`` instance.
    mask_params : dict
        For ``pc``: ``alpha``, ``exogenous``, ``forbidden_edges``.
        For ``stability``: ``alpha``, ``tau_stab``, ``B``, ``exogenous``,
        ``forbidden_edges``, ``random_state``.

        ``exogenous`` blocks all incoming edges to the listed variables.
        ``forbidden_edges`` is a list of ``(target, source)`` pairs removed
        after structure discovery (domain-knowledge curation).
    discover_mask_on : {"full_data", "train_fold"}
        For ``pc`` and ``stability``, where to learn the structural mask.
        ``full_data`` discovers the mask once on the full dataset and reuses it
        in every fold. ``train_fold`` re-estimates the mask on each training split.
    feature_type : str
        ``godambe`` (thesis method) or ``raw`` (unwhitened gradients).
    outer_splits : int
        Number of CV folds (default 5).
    random_state : int
    """

    def __init__(
        self,
        mask: Union[str, StructuralMask] = "hand",
        mask_object: Optional[StructuralMask] = None,
        mask_params: Optional[Dict[str, Any]] = None,
        discover_mask_on: Literal["full_data", "train_fold"] = "full_data",
        feature_type: str = "godambe",
        lambda_reg: float = 0.01,
        ridge_gamma: float = 1e-3,
        C: float = 1.0,
        scale_phi: bool = True,
        outer_splits: int = 5,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.mask = mask
        self.mask_object = mask_object
        self.mask_params = mask_params or {}
        self.discover_mask_on = discover_mask_on
        self.feature_type = feature_type
        self.lambda_reg = lambda_reg
        self.ridge_gamma = ridge_gamma
        self.C = C
        self.scale_phi = scale_phi
        self.outer_splits = outer_splits
        self.random_state = random_state
        self.verbose = verbose

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        continuous_idx: np.ndarray,
        ordinal_idx: np.ndarray,
        variable_names: Optional[List[str]] = None,
    ) -> CrossValidationResult:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)

        mask = self.mask
        mask_object = self.mask_object
        mask_params = self.mask_params
        fixed_mask: Optional[StructuralMask] = None

        if mask_object is None and mask in ("pc", "stability", "data_driven"):
            if self.discover_mask_on == "full_data":
                if variable_names is None:
                    raise ValueError(
                        "variable_names are required to discover a data-driven mask."
                    )
                fixed_mask = discover_data_driven_mask(
                    X,
                    variable_names,
                    continuous_idx,
                    str(mask),
                    mask_params,
                )
                mask = "hand"
                mask_object = fixed_mask
                mask_params = {}
                if self.verbose:
                    print(
                        "Fixed structural mask discovered on full data: "
                        f"{fixed_mask}"
                    )

        cv = StratifiedKFold(
            n_splits=self.outer_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            if self.verbose:
                print(f"\n=== Fold {fold_idx + 1}/{self.outer_splits} ===")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            fold_mask = mask
            fold_mask_object = mask_object
            fold_mask_params = mask_params

            if (
                mask_object is None
                and mask in ("pc", "stability", "data_driven")
                and self.discover_mask_on == "train_fold"
            ):
                fold_mask_params = mask_params

            clf = GeometryFisherClassifier(
                mask=fold_mask,
                mask_object=fold_mask_object,
                mask_params=fold_mask_params,
                lambda_reg=self.lambda_reg,
                ridge_gamma=self.ridge_gamma,
                feature_type=self.feature_type,
                C=self.C,
                scale_phi=self.scale_phi,
                verbose=self.verbose,
            )

            clf.fit(
                X_train,
                y_train,
                continuous_idx=continuous_idx,
                ordinal_idx=ordinal_idx,
                variable_names=variable_names,
            )

            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            try:
                auc = roc_auc_score(y_test, y_proba)
            except ValueError:
                auc = np.nan

            n_params = clf.mask_.n_params if clf.mask_ is not None else -1

            if self.verbose:
                print(
                    f"Fold {fold_idx + 1} — Acc: {acc:.3f} | Macro-F1: {f1:.3f} | "
                    f"AUC: {auc:.3f} | n_params: {n_params}"
                )

            fold_results.append(
                FoldResult(
                    accuracy=acc,
                    macro_f1=f1,
                    auc=auc,
                    n_params=n_params,
                )
            )

        accs = [r.accuracy for r in fold_results]
        f1s = [r.macro_f1 for r in fold_results]
        aucs = [r.auc for r in fold_results if not np.isnan(r.auc)]

        result = CrossValidationResult(
            fold_results=fold_results,
            mean_accuracy=float(np.mean(accs)),
            std_accuracy=float(np.std(accs)),
            mean_macro_f1=float(np.mean(f1s)),
            std_macro_f1=float(np.std(f1s)),
            mean_auc=float(np.mean(aucs)) if aucs else np.nan,
            std_auc=float(np.std(aucs)) if aucs else np.nan,
            fixed_mask=fixed_mask,
        )

        if self.verbose:
            print("\n" + "=" * 50)
            print("CROSS-VALIDATION SUMMARY")
            print("=" * 50)
            print(f"Accuracy:  {result.mean_accuracy:.3f} ± {result.std_accuracy:.3f}")
            print(f"Macro-F1:  {result.mean_macro_f1:.3f} ± {result.std_macro_f1:.3f}")
            print(f"AUC:       {result.mean_auc:.3f} ± {result.std_auc:.3f}")
            print("=" * 50)

        return result
