"""
Post-fit diagnostics for choosing raw vs Godambe-whitened gradient features.

After fitting class-conditional composite models, these checks estimate whether
inverse-Godambe whitening is likely to help on a tabular dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from numpy.linalg import cond, inv

from .composite import CompositeLikelihoodModel
from .cross_validation import CrossValidationExperiment
from .geometry import stable_symmetrize
from .simulation import hj_discrepancy
from .structure import StructuralMask, discover_data_driven_mask


FeatureChoice = Literal["raw", "godambe", "either"]


@dataclass
class ClassGeometryDiagnostics:
    label: int
    n_samples: int
    n_params: int
    hj_discrepancy: float
    cond_H: float
    cond_Ginv: float


@dataclass
class FeatureTypeReport:
    mask: StructuralMask
    n_total: int
    n_min_class: int
    class_diagnostics: List[ClassGeometryDiagnostics]
    raw_accuracy: float
    raw_accuracy_std: float
    raw_auc: float
    raw_auc_std: float
    godambe_accuracy: float
    godambe_accuracy_std: float
    godambe_auc: float
    godambe_auc_std: float
    delta_auc: float
    delta_accuracy: float
    sample_ratio: float
    mean_hj_discrepancy: float
    outer_splits: int
    recommendation: FeatureChoice
    reasons: List[str]


def _fit_class_model(
    Xk: np.ndarray,
    X_pooled: np.ndarray,
    mask: StructuralMask,
    continuous_idx: np.ndarray,
    ordinal_idx: np.ndarray,
    variable_names: List[str],
    lambda_reg: float,
) -> CompositeLikelihoodModel:
    temp = CompositeLikelihoodModel(mask=mask, lambda_reg=lambda_reg)
    temp.continuous_idx_ = continuous_idx
    temp.ordinal_idx_ = ordinal_idx
    shared_thresholds, shared_categories = temp._estimate_thresholds_and_cats(X_pooled)
    model = CompositeLikelihoodModel(mask=mask, lambda_reg=lambda_reg)
    model.fit(
        Xk,
        continuous_idx,
        ordinal_idx,
        variable_names,
        shared_thresholds=shared_thresholds,
        shared_categories=shared_categories,
    )
    return model


def estimate_class_geometry(
    X: np.ndarray,
    y: np.ndarray,
    mask: StructuralMask,
    continuous_idx: np.ndarray,
    ordinal_idx: np.ndarray,
    variable_names: List[str],
    *,
    lambda_reg: float = 0.01,
    ridge_gamma: float = 1e-3,
) -> List[ClassGeometryDiagnostics]:
    """Per-class H, J, and inverse-Godambe diagnostics at the fitted solution."""
    rows: List[ClassGeometryDiagnostics] = []
    for label in (0, 1):
        Xk = X[y == label]
        if len(Xk) < 5:
            continue
        model = _fit_class_model(
            Xk,
            X,
            mask,
            continuous_idx,
            ordinal_idx,
            variable_names,
            lambda_reg,
        )
        grads = model.per_observation_gradient(Xk)
        H = model.objective_hessian(Xk)
        J = stable_symmetrize((grads.T @ grads) / len(Xk))
        H_reg = stable_symmetrize(H) + ridge_gamma * np.eye(H.shape[0])
        H_inv = inv(H_reg)
        Ginv = H_inv @ J @ H_inv
        rows.append(
            ClassGeometryDiagnostics(
                label=label,
                n_samples=len(Xk),
                n_params=mask.n_params,
                hj_discrepancy=hj_discrepancy(H, J),
                cond_H=float(cond(H_reg)),
                cond_Ginv=float(cond(Ginv)),
            )
        )
    return rows


def _resolve_mask(
    mask_kind: str,
    X: np.ndarray,
    variable_names: List[str],
    continuous_idx: np.ndarray,
    mask_object: Optional[StructuralMask],
    mask_params: Optional[dict],
) -> Tuple[StructuralMask, str, Optional[StructuralMask], dict]:
    if mask_object is not None:
        return mask_object, "hand", mask_object, mask_params or {}
    if mask_kind in ("pc", "stability", "data_driven"):
        params = mask_params or {}
        fixed = discover_data_driven_mask(
            X,
            variable_names,
            continuous_idx,
            mask_kind,
            params,
        )
        return fixed, "hand", fixed, {}
    raise ValueError(f"Unknown mask kind: {mask_kind}")


def recommend_feature_type(
    *,
    delta_auc: float,
    delta_accuracy: float,
    n_min_class: int,
    n_params: int,
    mean_hj: float,
    cv_threshold: float = 0.02,
    hj_threshold: float = 0.5,
    sample_ratio_threshold: float = 2.0,
) -> Tuple[FeatureChoice, List[str]]:
    """
    Combine CV comparison with geometry/sample-size heuristics.

    Priority:
    1. Clear CV winner on AUC (primary) or accuracy (secondary)
    2. If tied, prefer godambe when sample is small vs d or H differs strongly from J
    3. Otherwise raw (simpler default when performance matches)
    """
    reasons: List[str] = []
    sample_ratio = n_min_class / max(n_params, 1)

    if delta_auc >= cv_threshold:
        reasons.append(
            f"Cross-validated AUC is {delta_auc:+.3f} higher with Godambe whitening."
        )
        return "godambe", reasons
    if delta_auc <= -cv_threshold:
        reasons.append(
            f"Cross-validated AUC is {-delta_auc:+.3f} higher with raw gradients."
        )
        return "raw", reasons

    if delta_accuracy >= cv_threshold:
        reasons.append(
            f"Cross-validated accuracy is {delta_accuracy:+.3f} higher with Godambe."
        )
        return "godambe", reasons
    if delta_accuracy <= -cv_threshold:
        reasons.append(
            f"Cross-validated accuracy is {-delta_accuracy:+.3f} higher with raw gradients."
        )
        return "raw", reasons

    reasons.append(
        f"Cross-validated AUC and accuracy differ by less than {cv_threshold:.2f} "
        "(performance tie)."
    )

    if sample_ratio < sample_ratio_threshold:
        reasons.append(
            f"Smallest class size ({n_min_class}) is small relative to mask "
            f"dimension d={n_params} (n_min/d={sample_ratio:.2f}); "
            "Godambe is safer when geometry must be estimated from limited data."
        )
        return "godambe", reasons

    if mean_hj >= hj_threshold:
        reasons.append(
            f"Mean ||H-J||/||H|| = {mean_hj:.3f} (composite likelihood regime); "
            "Godambe is the principled metric even when CV is tied."
        )
        return "godambe", reasons

    reasons.append(
        f"Plenty of samples per class (n_min/d={sample_ratio:.2f}) and moderate "
        "H-J discrepancy; raw gradients are sufficient."
    )
    return "raw", reasons


def evaluate_feature_type_choice(
    X: np.ndarray,
    y: np.ndarray,
    continuous_idx: np.ndarray,
    ordinal_idx: np.ndarray,
    variable_names: List[str],
    *,
    mask: str = "hand",
    mask_object: Optional[StructuralMask] = None,
    mask_params: Optional[dict] = None,
    outer_splits: int = 5,
    random_state: int = 42,
    lambda_reg: float = 0.01,
    ridge_gamma: float = 1e-3,
    cv_threshold: float = 0.02,
) -> FeatureTypeReport:
    """
    Run geometry diagnostics and paired CV for raw vs Godambe features.

    Use the returned ``recommendation`` to set ``feature_type`` in
    ``GeometryFisherClassifier`` / ``CrossValidationExperiment``.
    """
    resolved_mask, cv_mask, cv_mask_object, cv_mask_params = _resolve_mask(
        mask,
        X,
        variable_names,
        continuous_idx,
        mask_object,
        mask_params,
    )

    class_diag = estimate_class_geometry(
        X,
        y,
        resolved_mask,
        continuous_idx,
        ordinal_idx,
        variable_names,
        lambda_reg=lambda_reg,
        ridge_gamma=ridge_gamma,
    )
    mean_hj = float(np.mean([row.hj_discrepancy for row in class_diag])) if class_diag else float("nan")
    n_min = min(int(np.sum(y == 0)), int(np.sum(y == 1)))
    d = resolved_mask.n_params

    cv_results: Dict[str, Tuple[float, float, float, float]] = {}
    for feature_type in ("raw", "godambe"):
        experiment = CrossValidationExperiment(
            mask=cv_mask,
            mask_object=cv_mask_object,
            mask_params=cv_mask_params,
            discover_mask_on="full_data",
            feature_type=feature_type,
            lambda_reg=lambda_reg,
            ridge_gamma=ridge_gamma,
            outer_splits=outer_splits,
            random_state=random_state,
            verbose=False,
        )
        result = experiment.run(
            X,
            y,
            continuous_idx=continuous_idx,
            ordinal_idx=ordinal_idx,
            variable_names=variable_names,
        )
        cv_results[feature_type] = (
            result.mean_accuracy,
            result.std_accuracy,
            result.mean_auc,
            result.std_auc,
        )

    raw_acc, raw_acc_s, raw_auc, raw_auc_s = cv_results["raw"]
    god_acc, god_acc_s, god_auc, god_auc_s = cv_results["godambe"]
    delta_auc = god_auc - raw_auc
    delta_acc = god_acc - raw_acc

    recommendation, reasons = recommend_feature_type(
        delta_auc=delta_auc,
        delta_accuracy=delta_acc,
        n_min_class=n_min,
        n_params=d,
        mean_hj=mean_hj,
        cv_threshold=cv_threshold,
    )

    for row in class_diag:
        if row.cond_Ginv > 1e8 or row.cond_H > 1e8:
            reasons.append(
                f"Class {row.label}: large condition number "
                f"(cond(H)={row.cond_H:.2e}, cond(G_inv)={row.cond_Ginv:.2e}) — "
                "check ridge_gamma or use a sparser mask."
            )

    return FeatureTypeReport(
        mask=resolved_mask,
        n_total=len(y),
        n_min_class=n_min,
        class_diagnostics=class_diag,
        raw_accuracy=raw_acc,
        raw_accuracy_std=raw_acc_s,
        raw_auc=raw_auc,
        raw_auc_std=raw_auc_s,
        godambe_accuracy=god_acc,
        godambe_accuracy_std=god_acc_s,
        godambe_auc=god_auc,
        godambe_auc_std=god_auc_s,
        delta_auc=delta_auc,
        delta_accuracy=delta_acc,
        sample_ratio=n_min / max(d, 1),
        mean_hj_discrepancy=mean_hj,
        outer_splits=outer_splits,
        recommendation=recommendation,
        reasons=reasons,
    )


def format_feature_type_report(report: FeatureTypeReport, mask_label: str) -> str:
    lines = [
        "Feature-type selection report",
        "=" * 40,
        f"Samples: {report.n_total}   min class: {report.n_min_class}   "
        f"mask: {mask_label}   d={report.mask.n_params}",
        "",
        "Estimated geometry (after composite fit on full data)",
        "-" * 40,
    ]
    for row in report.class_diagnostics:
        lines.append(
            f"  class {row.label}: n={row.n_samples}  "
            f"||H-J||/||H||={row.hj_discrepancy:.3f}  "
            f"cond(H)={row.cond_H:.2e}  cond(G_inv)={row.cond_Ginv:.2e}"
        )
    lines.extend(
        [
            f"  mean ||H-J||/||H||: {report.mean_hj_discrepancy:.3f}",
            f"  n_min / d: {report.sample_ratio:.2f}",
            "",
            f"Paired {report.outer_splits}-fold CV (same mask, scale_phi=True)",
            "-" * 40,
            f"  raw     accuracy {report.raw_accuracy:.3f} ± {report.raw_accuracy_std:.3f}   "
            f"AUC {report.raw_auc:.3f} ± {report.raw_auc_std:.3f}",
            f"  godambe accuracy {report.godambe_accuracy:.3f} ± {report.godambe_accuracy_std:.3f}   "
            f"AUC {report.godambe_auc:.3f} ± {report.godambe_auc_std:.3f}",
            f"  delta(Godambe - raw): accuracy {report.delta_accuracy:+.3f}   AUC {report.delta_auc:+.3f}",
            "",
            f"Recommendation: feature_type='{report.recommendation}'",
            "-" * 40,
        ]
    )
    for reason in report.reasons:
        lines.append(f"  - {reason}")
    return "\n".join(lines)
