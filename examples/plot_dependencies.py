"""
Fit a model on Heart Disease data and plot the dependency structures.
"""

import numpy as np
import matplotlib.pyplot as plt

from geometry_fisher.data import load_heart_disease
from geometry_fisher.structure import StructuralMask
from geometry_fisher.pipeline import GeometryFisherClassifier
from geometry_fisher.visualization import (
    plot_class_dependencies,
    plot_difference_heatmap,
    plot_mask,
)

from config import DATA_PATH

X, y, variable_names, continuous_idx, ordinal_idx = load_heart_disease(
    path=str(DATA_PATH),
    binary_target=True,
    only_cleveland=False,
)

# -------------------------------------------------
# 2. Create a hand-specified mask
# -------------------------------------------------
mask = StructuralMask.from_domain_knowledge(
    variable_names=variable_names,
    exogenous=["age", "sex"]
)

print(f"Mask: {mask}")

# -------------------------------------------------
# 3. Fit the classifier (on the full data for visualization)
# -------------------------------------------------
print("Fitting model...")

clf = GeometryFisherClassifier(
    mask="hand",
    mask_object=mask,
    feature_type="linear",
    lambda_reg=0.01,
    ridge_gamma=1e-3,
)

clf.fit(
    X, y,
    continuous_idx=continuous_idx,
    ordinal_idx=ordinal_idx,
    variable_names=variable_names,
)

print("Fitting done.")

# -------------------------------------------------
# 4. Create the plots
# -------------------------------------------------
print("Generating plots...")

# Plot the mask
fig_mask = plot_mask(mask)
fig_mask.savefig("mask.png", dpi=150, bbox_inches="tight")
print("Saved: mask.png")

# Plot class-specific dependency matrices
fig_deps = plot_class_dependencies(clf.model_0_, clf.model_1_, variable_names)
fig_deps.savefig("class_dependencies.png", dpi=150, bbox_inches="tight")
print("Saved: class_dependencies.png")

# Plot the difference
fig_diff = plot_difference_heatmap(clf.model_0_, clf.model_1_, variable_names)
fig_diff.savefig("difference_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved: difference_heatmap.png")

print("\nAll plots saved successfully.")
plt.show()