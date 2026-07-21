"""
Run Experiment 2: data-driven structural masks (PC and stability selection).

Outputs are written to examples/outputs/experiment2_results.csv.
"""

from __future__ import annotations

import pandas as pd

from geometry_fisher.cross_validation import CrossValidationExperiment
from geometry_fisher.data import load_heart_disease
from geometry_fisher.experiments import save_results_table

from config import DATA_PATH, OUTPUT_DIR

MASK_SPECS = [
    {
        "name": "PC algorithm (single run)",
        "mask": "pc",
        "mask_params": {"alpha": 0.05, "exogenous": ["age", "sex"]},
    },
    {
        "name": "PC stability selection",
        "mask": "stability",
        "mask_params": {
            "alpha": 0.05,
            "tau_stab": 0.6,
            "B": 50,
            "exogenous": ["age", "sex"],
            "random_state": 42,
        },
    },
]


def main() -> pd.DataFrame:
    X, y, variable_names, continuous_idx, ordinal_idx = load_heart_disease(
        path=str(DATA_PATH),
        binary_target=True,
        verbose=True,
    )

    rows = []

    print("=" * 78)
    print(f"Experiment 2 — data-driven masks ({X.shape[0]} samples, 5-fold CV)")
    print("=" * 78)

    for spec in MASK_SPECS:
        print(f"\n{spec['name']}\n")

        experiment = CrossValidationExperiment(
            mask=spec["mask"],
            mask_params=spec["mask_params"],
            feature_type="godambe",
            lambda_reg=0.01,
            ridge_gamma=1e-3,
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

        mean_n_params = sum(r.n_params for r in result.fold_results) / len(
            result.fold_results
        )

        rows.append(
            {
                "Method": spec["name"],
                "Mask": spec["mask"],
                "Accuracy": f"{result.mean_accuracy:.3f} ± {result.std_accuracy:.3f}",
                "Macro-F1": f"{result.mean_macro_f1:.3f} ± {result.std_macro_f1:.3f}",
                "ROC-AUC": f"{result.mean_auc:.3f} ± {result.std_auc:.3f}",
                "Accuracy_mean": result.mean_accuracy,
                "Accuracy_std": result.std_accuracy,
                "Macro-F1_mean": result.mean_macro_f1,
                "Macro-F1_std": result.std_macro_f1,
                "ROC-AUC_mean": result.mean_auc,
                "ROC-AUC_std": result.std_auc,
                "Mean_n_params": mean_n_params,
            }
        )

    table = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print("Experiment 2 results")
    print("=" * 78)
    print(
        table[["Method", "Accuracy", "Macro-F1", "ROC-AUC", "Mean_n_params"]].to_string(
            index=False
        )
    )
    print("=" * 78)

    return table


if __name__ == "__main__":
    table = main()
    csv_path, json_path = save_results_table(
        table,
        OUTPUT_DIR,
        stem="experiment2_results",
    )
    print(f"\nSaved:\n  {csv_path}\n  {json_path}")
