"""
Run Experiment 2: data-driven structural masks (PC and stability selection).

The PC row uses the fixed 16-edge thesis reference mask discovered on the
full pooled sample. Stability selection is discovered once on all 531 samples.

Outputs are written to docs/results/experiment2_results.csv and
examples/outputs/experiment2_results.csv.
"""

from __future__ import annotations

import pandas as pd

from geometry_fisher.cross_validation import CrossValidationExperiment
from geometry_fisher.data import load_heart_disease
from geometry_fisher.experiments import save_results_table
from geometry_fisher.structure import StructuralMask

from config import DATA_PATH, OUTPUT_DIR, RESULTS_DIR

MASK_SPECS = [
    {
        "name": "PC algorithm (single run)",
        "mask": "hand",
        "mask_object": None,
        "use_thesis_reference": True,
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
        "use_thesis_reference": False,
    },
]


def main() -> pd.DataFrame:
    X, y, variable_names, continuous_idx, ordinal_idx = load_heart_disease(
        path=str(DATA_PATH),
        binary_target=True,
        verbose=True,
    )

    thesis_pc_mask = StructuralMask.from_thesis_pc_reference(variable_names)
    print(f"Thesis PC reference mask: {thesis_pc_mask}")

    rows = []

    print("=" * 78)
    print(f"Experiment 2 — data-driven masks ({X.shape[0]} samples, 5-fold CV)")
    print("=" * 78)

    for spec in MASK_SPECS:
        print(f"\n{spec['name']}\n")

        mask_object = thesis_pc_mask if spec.get("use_thesis_reference") else None
        mask = spec["mask"]
        mask_params = spec.get("mask_params", {})

        experiment = CrossValidationExperiment(
            mask=mask,
            mask_object=mask_object,
            mask_params=mask_params,
            discover_mask_on="full_data",
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

        fixed_mask = mask_object if mask_object is not None else result.fixed_mask
        n_params = fixed_mask.n_params if fixed_mask is not None else -1

        rows.append(
            {
                "Method": spec["name"],
                "Mask": "thesis_pc" if spec.get("use_thesis_reference") else spec["mask"],
                "Accuracy": f"{result.mean_accuracy:.3f} ± {result.std_accuracy:.3f}",
                "Macro-F1": f"{result.mean_macro_f1:.3f} ± {result.std_macro_f1:.3f}",
                "ROC-AUC": f"{result.mean_auc:.3f} ± {result.std_auc:.3f}",
                "Accuracy_mean": result.mean_accuracy,
                "Accuracy_std": result.std_accuracy,
                "Macro-F1_mean": result.mean_macro_f1,
                "Macro-F1_std": result.std_macro_f1,
                "ROC-AUC_mean": result.mean_auc,
                "ROC-AUC_std": result.std_auc,
                "n_params": n_params,
            }
        )

    table = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print("Experiment 2 results")
    print("=" * 78)
    print(
        table[["Method", "Accuracy", "Macro-F1", "ROC-AUC", "n_params"]].to_string(
            index=False
        )
    )
    print("=" * 78)

    return table


if __name__ == "__main__":
    table = main()
    saved_paths = []
    for output_dir in (RESULTS_DIR, OUTPUT_DIR):
        csv_path, json_path = save_results_table(
            table,
            output_dir,
            stem="experiment2_results",
        )
        saved_paths.extend([csv_path, json_path])

    print("\nSaved:")
    for path in saved_paths:
        print(f"  {path}")
