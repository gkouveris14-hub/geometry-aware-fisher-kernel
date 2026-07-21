"""
Fit a model on Heart Disease data and plot the dependency structures.

Saves figures to docs/figures/ for README preview and local inspection.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geometry_fisher.data import load_heart_disease
from geometry_fisher.structure import StructuralMask
from geometry_fisher.pipeline import GeometryFisherClassifier
from geometry_fisher.visualization import (
    plot_class_dependencies,
    plot_difference_heatmap,
    plot_mask,
)

from config import DATA_PATH, FIGURES_DIR


def main(*, show: bool = False) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X, y, variable_names, continuous_idx, ordinal_idx = load_heart_disease(
        path=str(DATA_PATH),
        binary_target=True,
    )

    mask = StructuralMask.from_domain_knowledge(
        variable_names=variable_names,
        exogenous=["age", "sex"],
    )

    print(f"Mask: {mask}")
    print("Fitting model...")

    clf = GeometryFisherClassifier(
        mask="hand",
        mask_object=mask,
        feature_type="linear",
        lambda_reg=0.01,
        ridge_gamma=1e-3,
        scale_phi=True,
        verbose=False,
    )
    clf.fit(
        X,
        y,
        continuous_idx=continuous_idx,
        ordinal_idx=ordinal_idx,
        variable_names=variable_names,
    )

    print("Generating plots...")

    figures = {
        "mask.png": plot_mask(mask),
        "class_dependencies.png": plot_class_dependencies(
            clf.model_0_, clf.model_1_, variable_names
        ),
        "difference_heatmap.png": plot_difference_heatmap(
            clf.model_0_, clf.model_1_, variable_names
        ),
    }

    for name, fig in figures.items():
        path = FIGURES_DIR / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")

    if show:
        plt.show()
    else:
        plt.close("all")

    print("\nAll plots saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures after saving (requires a GUI backend).",
    )
    args = parser.parse_args()
    main(show=args.show)
