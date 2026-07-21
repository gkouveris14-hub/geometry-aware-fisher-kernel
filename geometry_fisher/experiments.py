"""
Heart Disease experiment table: baselines and geometry-aware classifiers.

Protocol: 531 samples, 5-fold stratified CV, random_state=42, hand-specified mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .baselines import run_baseline_cv
from .cross_validation import CrossValidationExperiment
from .data import load_heart_disease
from .structure import StructuralMask


@dataclass(frozen=True)
class MethodSpec:
    name: str
    group: str
    feature_type: Optional[str] = None


METHODS: List[MethodSpec] = [
    MethodSpec("Logistic Regression", "baseline"),
    MethodSpec("Random Forest", "baseline"),
    MethodSpec("XGBoost", "baseline"),
    MethodSpec("Raw gradient features", "comparison", "raw"),
    MethodSpec("Godambe whitening", "method", "godambe"),
]


def _summary_row(
    name: str,
    group: str,
    mean_accuracy: float,
    std_accuracy: float,
    mean_macro_f1: float,
    std_macro_f1: float,
    mean_auc: float,
    std_auc: float,
) -> dict:
    return {
        "Method": name,
        "Group": group,
        "Accuracy": f"{mean_accuracy:.3f} ± {std_accuracy:.3f}",
        "Macro-F1": f"{mean_macro_f1:.3f} ± {std_macro_f1:.3f}",
        "ROC-AUC": f"{mean_auc:.3f} ± {std_auc:.3f}",
        "Accuracy_mean": mean_accuracy,
        "Accuracy_std": std_accuracy,
        "Macro-F1_mean": mean_macro_f1,
        "Macro-F1_std": std_macro_f1,
        "ROC-AUC_mean": mean_auc,
        "ROC-AUC_std": std_auc,
    }


def run_experiments(
    data_path: str,
    *,
    outer_splits: int = 5,
    random_state: int = 42,
    lambda_reg: float = 0.01,
    ridge_gamma: float = 1e-3,
    verbose: bool = True,
) -> pd.DataFrame:
    X, y, variable_names, continuous_idx, ordinal_idx = load_heart_disease(
        path=data_path,
        binary_target=True,
        verbose=verbose,
    )
    n_samples = X.shape[0]

    mask = StructuralMask.from_domain_knowledge(
        variable_names=variable_names,
        exogenous=["age", "sex"],
    )

    rows: List[dict] = []

    if verbose:
        print("=" * 78)
        print(f"Heart Disease experiments ({n_samples} samples, 5-fold CV)")
        print("=" * 78)
        print(f"Hand-specified mask: {mask}\n")
        print("Baselines\n")

    baseline_result = run_baseline_cv(
        X,
        y,
        variable_names=variable_names,
        outer_splits=outer_splits,
        random_state=random_state,
        verbose=verbose,
    )

    baseline_by_name = {s.model: s for s in baseline_result.summaries}
    for method in METHODS:
        if method.group != "baseline":
            continue
        if method.name not in baseline_by_name:
            if method.name == "XGBoost":
                continue
            raise KeyError(f"Missing baseline result for {method.name}")

        summary = baseline_by_name[method.name]
        rows.append(
            _summary_row(
                method.name,
                method.group,
                summary.mean_accuracy,
                summary.std_accuracy,
                summary.mean_macro_f1,
                summary.std_macro_f1,
                summary.mean_auc,
                summary.std_auc,
            )
        )

    for method in METHODS:
        if method.feature_type is None:
            continue

        if verbose:
            print(f"\n{method.name}\n")

        experiment = CrossValidationExperiment(
            mask="hand",
            mask_object=mask,
            feature_type=method.feature_type,
            lambda_reg=lambda_reg,
            ridge_gamma=ridge_gamma,
            outer_splits=outer_splits,
            random_state=random_state,
            verbose=verbose,
        )

        result = experiment.run(
            X,
            y,
            continuous_idx=continuous_idx,
            ordinal_idx=ordinal_idx,
            variable_names=variable_names,
        )

        rows.append(
            _summary_row(
                method.name,
                method.group,
                result.mean_accuracy,
                result.std_accuracy,
                result.mean_macro_f1,
                result.std_macro_f1,
                result.mean_auc,
                result.std_auc,
            )
        )

    table = pd.DataFrame(rows)

    if verbose:
        display_cols = ["Method", "Group", "Accuracy", "Macro-F1", "ROC-AUC"]
        print("\n" + "=" * 78)
        print("Results")
        print("=" * 78)
        print(table[display_cols].to_string(index=False))
        print("=" * 78)

    return table


def save_results_table(
    table: pd.DataFrame,
    output_dir: Path,
    stem: str = "results",
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"

    table.to_csv(csv_path, index=False)
    table.to_json(json_path, orient="records", indent=2)
    return csv_path, json_path
