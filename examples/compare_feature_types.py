"""
Choose raw vs Godambe gradient features after fitting the composite geometry.

Runs estimated-geometry checks (H vs J, condition numbers, sample size vs mask
dimension) and paired cross-validation on the same mask, then prints a
recommended feature_type for GeometryFisherClassifier.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from geometry_fisher.data import load_heart_disease
from geometry_fisher.diagnostics import evaluate_feature_type_choice, format_feature_type_report
from geometry_fisher.structure import StructuralMask


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-fit tests to choose raw vs Godambe-whitened gradient features."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="CSV path (default: bundled Heart Disease benchmark)",
    )
    parser.add_argument("--mask", choices=("hand", "pc", "stability"), default="hand")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument(
        "--cv-threshold",
        type=float,
        default=0.02,
        help="Minimum AUC/accuracy gain to prefer Godambe over raw (default 0.02).",
    )
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

    if args.mask in ("pc", "stability"):
        mask_params = {
            "alpha": 0.05,
            "exogenous": ["age", "sex"],
            "tau_stab": 0.6,
            "B": 50,
            "random_state": 42,
        }

    report = evaluate_feature_type_choice(
        X,
        y,
        continuous_idx,
        ordinal_idx,
        variable_names,
        mask=args.mask,
        mask_object=mask_object,
        mask_params=mask_params,
        outer_splits=args.splits,
        cv_threshold=args.cv_threshold,
    )

    print(format_feature_type_report(report, mask_label=args.mask))
    print()
    print(
        "Next step: set feature_type in GeometryFisherClassifier to the "
        f"recommended value ('{report.recommendation}')."
    )


if __name__ == "__main__":
    main()
