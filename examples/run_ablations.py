"""
Run the main ablation study (Variants A–E) on Heart Disease data.
"""

import numpy as np
from geometry_fisher.data import load_heart_disease
from geometry_fisher.structure import StructuralMask
from geometry_fisher.nested_cv import NestedCVExperiment

DATA_PATH = r"C:\ΑΡΧΕΙΑ\UNIC\Thesis\heart_disease_uci.csv"

X, y, variable_names, continuous_idx, ordinal_idx = load_heart_disease(
    path=DATA_PATH,
    binary_target=True,
    only_cleveland=True,
)

mask = StructuralMask.from_domain_knowledge(
    variable_names=variable_names,
    exogenous=["age", "sex"]
)

variants = ["raw", "fisher_only", "linear", "quadratic", "full"]

print("=" * 70)
print("ABLATION STUDY – Geometry-Aware Fisher Kernel")
print("=" * 70)

results = {}

for variant in variants:
    print(f"\n\n>>>>> Running variant: {variant.upper()} <<<<<\n")

    experiment = NestedCVExperiment(
        mask="hand",
        mask_object=mask,
        feature_type=variant,
        lambda_reg=0.01,
        ridge_gamma=1e-3,
        outer_splits=5,
        random_state=42,
    )

    result = experiment.run(
        X, y,
        continuous_idx=continuous_idx,
        ordinal_idx=ordinal_idx,
        variable_names=variable_names,
    )

    results[variant] = result

print("\n\n" + "=" * 70)
print("ABLATION SUMMARY")
print("=" * 70)
print(f"{'Variant':<15} {'Accuracy':>12} {'Macro-F1':>12} {'AUC':>12}")
print("-" * 70)

for variant in variants:
    r = results[variant]
    print(f"{variant:<15} {r.mean_accuracy:.3f}±{r.std_accuracy:.3f}  "
          f"{r.mean_macro_f1:.3f}±{r.std_macro_f1:.3f}  "
          f"{r.mean_auc:.3f}±{r.std_auc:.3f}")

print("=" * 70)