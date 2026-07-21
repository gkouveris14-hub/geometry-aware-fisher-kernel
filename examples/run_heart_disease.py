"""
Run Nested CV on the real UCI Heart Disease dataset.
"""

import numpy as np
from geometry_fisher.data import load_heart_disease
from geometry_fisher.structure import StructuralMask
from geometry_fisher.nested_cv import NestedCVExperiment

from config import DATA_PATH

X, y, variable_names, continuous_idx, ordinal_idx = load_heart_disease(
    path=str(DATA_PATH),
    binary_target=True,
)

print("\nVariable names:", variable_names)

# -------------------------------------------------
# 2. Hand-specified mask (domain knowledge)
# -------------------------------------------------
mask = StructuralMask.from_domain_knowledge(
    variable_names=variable_names,
    exogenous=["age", "sex"]
)

print(f"Hand-specified mask: {mask}")

# -------------------------------------------------
# 3. Run Nested CV
# -------------------------------------------------
print("\nStarting Nested Cross-Validation...")

experiment = NestedCVExperiment(
    mask="hand",
    mask_object=mask,
    feature_type="godambe",
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

print("\nExperiment finished.")