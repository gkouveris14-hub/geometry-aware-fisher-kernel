"""
Quick comparison of raw vs Godambe features on a dataset.

Use this before committing to the full pipeline: it runs the same CV
protocol with both feature types and reports whether whitening helps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from geometry_fisher.composite import CompositeLikelihoodModel
from geometry_fisher.cross_validation import CrossValidationExperiment
from geometry_fisher.data import load_heart_disease
from geometry_fisher.simulation import _hj_discrepancy
from geometry_fisher.structure import StructuralMask


def _mean_hj_discrepancy(
    X: np.ndarray,
    y: np.ndarray,
    mask: StructuralMask,
    continuous_idx: np.ndarray,
    ordinal_idx: np.ndarray,
    variable_names: list[str],
    lambda_reg: float = 0.01,
) -> float:
    """Average class-wise ||H - J|| / ||H|| at the fitted composite solution."""
    values = []
    for label in (0, 1):
        Xk = X[y == label]
        if len(Xk) < 5:
            continue
        temp = CompositeLikelihoodModel(mask=mask, lambda_reg=lambda_reg)
        temp.continuous_idx_ = continuous_idx
        temp.ordinal_idx_ = ordinal_idx
        shared_thresholds, shared_categories = temp._estimate_thresholds_and_cats(X)
        model = CompositeLikelihoodModel(mask=mask, lambda_reg=lambda_reg)
        model.fit(
            Xk,
            continuous_idx,
            ordinal_idx,
            variable_names,
            shared_thresholds=shared_thresholds,
            shared_categories=shared_categories,
        )
        grads = model.per_observation_gradient(Xk)
        H = model.objective_hessian(Xk)
        J = (grads.T @ grads) / len(Xk)
        values.append(_hj_discrepancy(H, J))
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare raw vs Godambe features under the same CV protocol."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="CSV path (default: bundled Heart Disease benchmark)",
    )
    parser.add_argument("--mask", choices=("hand", "pc", "stability"), default="hand")
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()

    if args.data is None:
        repo_root = Path(__file__).resolve().parents[1]
        data_path = repo_root / "data" / "heart_disease_uci.csv"
    else:
        data_path = args.data

    X, y, variable_names, continuous_idx, ordinal_idx = load_heart_disease(
        path=str(data_path),
        binary_target=True,
        verbose=False,
    )

    mask_object = None
    mask_params: dict = {}
    if args.mask == "hand":
        mask_object = StructuralMask.from_domain_knowledge(
            variable_names,
            exogenous=["age", "sex"],
        )
        mask = "hand"
    else:
        mask = args.mask
        mask_params = {
            "alpha": 0.05,
            "exogenous": ["age", "sex"],
            "tau_stab": 0.6,
            "B": 50,
            "random_state": 42,
        }

    n_min = min(int(np.sum(y == 0)), int(np.sum(y == 1)))
    d = mask_object.n_params if mask_object is not None else -1
    if mask_object is not None:
        hj = _mean_hj_discrepancy(
            X,
            y,
            mask_object,
            continuous_idx,
            ordinal_idx,
            variable_names,
        )
    else:
        hj = float("nan")

    print(f"Samples: {len(y)}  min class count: {n_min}  mask type: {args.mask}")
    if d > 0:
        print(f"Hand-mask parameters (reference): d={d}")
    print(f"Mean class-wise ||H - J|| / ||H||: {hj:.3f}")
    print()

    rows = []
    for feature_type in ("raw", "godambe"):
        experiment = CrossValidationExperiment(
            mask=mask,
            mask_object=mask_object,
            mask_params=mask_params,
            discover_mask_on="full_data",
            feature_type=feature_type,
            outer_splits=args.splits,
            random_state=42,
            verbose=False,
        )
        result = experiment.run(
            X,
            y,
            continuous_idx=continuous_idx,
            ordinal_idx=ordinal_idx,
            variable_names=variable_names,
        )
        rows.append(
            (
                feature_type,
                result.mean_accuracy,
                result.std_accuracy,
                result.mean_auc,
                result.std_auc,
            )
        )

    print(f"{'Feature type':<10} {'Accuracy':>18} {'ROC-AUC':>18}")
    for feature_type, acc_m, acc_s, auc_m, auc_s in rows:
        print(
            f"{feature_type:<10} {acc_m:6.3f} ± {acc_s:.3f}      "
            f"{auc_m:6.3f} ± {auc_s:.3f}"
        )

    raw_auc = rows[0][3]
    god_auc = rows[1][3]
    delta_auc = god_auc - raw_auc
    print()
    print(f"AUC gain (Godambe - raw): {delta_auc:+.3f}")
    print()
    print("Guidance:")
    if delta_auc > 0.02:
        print("  Whitening improves CV AUC on this dataset — prefer feature_type='godambe'.")
    elif delta_auc < -0.02:
        print("  Raw gradients match or beat whitening here — godambe is optional.")
    else:
        print("  Raw and Godambe are similar — either is fine; mask structure matters more.")
    if n_min < 2 * d if d > 0 else False:
        print("  Small sample per class relative to mask size — consider godambe or a sparser mask.")
    if hj > 0.5:
        print("  Large H-J discrepancy — composite geometry differs from Fisher; godambe is theoretically preferred.")


if __name__ == "__main__":
    main()
