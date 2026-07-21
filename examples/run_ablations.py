"""
Optional exploratory ablation (variants A–E). Not used in the paper table.
See run_paper_experiments.py for the published comparison (raw vs Godambe).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from geometry_fisher.data import load_heart_disease
from geometry_fisher.nested_cv import NestedCVExperiment
from geometry_fisher.structure import StructuralMask

from config import DATA_PATH, OUTPUT_DIR

VARIANTS = [
    ("A", "raw", "Concatenated raw gradients"),
    ("B", "fisher_only", "Whitened with J^{-1/2} only"),
    ("C", "linear", "Full Godambe linear whitening"),
    ("D", "quadratic", "Mahalanobis quadratic scores"),
    ("E", "full", "Linear Godambe + quadratic scores"),
]


def main() -> pd.DataFrame:
    X, y, variable_names, continuous_idx, ordinal_idx = load_heart_disease(
        path=str(DATA_PATH),
        binary_target=True,
        only_cleveland=False,
    )

    mask = StructuralMask.from_domain_knowledge(
        variable_names=variable_names,
        exogenous=["age", "sex"],
    )

    print("=" * 78)
    print("ABLATION STUDY – Geometry-Aware Fisher Kernel (531 samples, 5-fold CV)")
    print("=" * 78)
    print(f"Hand-specified mask: {mask}")

    rows = []

    for letter, variant, description in VARIANTS:
        print(f"\n\n>>>>> Variant {letter}: {variant.upper()} — {description} <<<<<\n")

        experiment = NestedCVExperiment(
            mask="hand",
            mask_object=mask,
            feature_type=variant,
            lambda_reg=0.01,
            ridge_gamma=1e-3,
            shrink_j=False,
            outer_splits=5,
            random_state=42,
            verbose=True,
        )

        result = experiment.run(
            X,
            y,
            continuous_idx=continuous_idx,
            ordinal_idx=ordinal_idx,
            variable_names=variable_names,
        )

        rows.append(
            {
                "variant": variant,
                "letter": letter,
                "description": description,
                "accuracy_mean": result.mean_accuracy,
                "accuracy_std": result.std_accuracy,
                "macro_f1_mean": result.mean_macro_f1,
                "macro_f1_std": result.std_macro_f1,
                "auc_mean": result.mean_auc,
                "auc_std": result.std_auc,
            }
        )

    summary = pd.DataFrame(rows)

    print("\n\n" + "=" * 78)
    print("ABLATION SUMMARY")
    print("=" * 78)
    print(
        f"{'Var':<5} {'Feature':<13} {'Accuracy':>14} {'Macro-F1':>14} {'AUC':>14}"
    )
    print("-" * 78)

    for row in rows:
        print(
            f"{row['letter']:<5} {row['variant']:<13} "
            f"{row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f}   "
            f"{row['macro_f1_mean']:.3f}±{row['macro_f1_std']:.3f}   "
            f"{row['auc_mean']:.3f}±{row['auc_std']:.3f}"
        )

    print("=" * 78)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "ablation_summary.csv"
    json_path = OUTPUT_DIR / "ablation_summary.json"

    summary.to_csv(csv_path, index=False)
    summary.to_json(json_path, orient="records", indent=2)

    print(f"\nSaved summary to:\n  {csv_path}\n  {json_path}")
    return summary


if __name__ == "__main__":
    main()
