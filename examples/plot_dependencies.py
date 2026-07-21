"""
Fit a model on Heart Disease data and plot dependency structures.

Saves figures to `docs/figures/`:
  - Hand-specified mask and fitted class dependencies (Experiment 1)
  - PC and stability-selection masks on the full dataset (Experiment 2)
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from geometry_fisher.data import load_heart_disease
from geometry_fisher.structure import StructuralMask, prepare_matrix_for_pc
from geometry_fisher.pipeline import GeometryFisherClassifier
from geometry_fisher.visualization import (
    plot_class_dependencies,
    plot_difference_heatmap,
    plot_mask,
    plot_structure_curation,
    plot_structure_graph,
)

from config import DATA_PATH, FIGURES_DIR

MASK_PARAMS = {
    "alpha": 0.05,
    "exogenous": ["age", "sex"],
    "tau_stab": 0.6,
    "B": 50,
    "random_state": 42,
}


def _scale_continuous(X: np.ndarray, continuous_idx: np.ndarray) -> np.ndarray:
    X_scaled = X.copy()
    scaler = StandardScaler()
    X_scaled[:, continuous_idx] = scaler.fit_transform(X[:, continuous_idx])
    return X_scaled


def main(*, show: bool = False) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X, y, variable_names, continuous_idx, ordinal_idx = load_heart_disease(
        path=str(DATA_PATH),
        binary_target=True,
    )
    X_scaled = _scale_continuous(X, continuous_idx)
    X_pc = prepare_matrix_for_pc(X)

    hand_mask = StructuralMask.from_domain_knowledge(
        variable_names=variable_names,
        exogenous=MASK_PARAMS["exogenous"],
    )

    print(f"Hand mask: {hand_mask}")
    print("Fitting Experiment 1 model...")

    clf = GeometryFisherClassifier(
        mask="hand",
        mask_object=hand_mask,
        feature_type="godambe",
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

    print("Building Experiment 2 data-driven masks on the full dataset...")
    pc_discovered = StructuralMask.from_pc_algorithm(
        X_pc,
        variable_names,
        alpha=MASK_PARAMS["alpha"],
    )
    pc_mask = StructuralMask.from_thesis_pc_reference(variable_names)
    print(f"PC discovered (raw): {pc_discovered}")
    print(f"Thesis PC reference: {pc_mask}")

    stability_discovered = StructuralMask.from_stability_selection(
        X_pc,
        variable_names,
        alpha=MASK_PARAMS["alpha"],
        tau_stab=MASK_PARAMS["tau_stab"],
        B=MASK_PARAMS["B"],
        random_state=MASK_PARAMS["random_state"],
    )
    stability_mask = stability_discovered.enforce_exogeneity(MASK_PARAMS["exogenous"])
    print(f"Stability discovered: {stability_discovered}")
    print(f"Stability curated: {stability_mask}")

    print("Generating plots...")

    figures = {
        "mask_hand.png": plot_mask(
            hand_mask,
            title=f"Hand-Specified Mask ({hand_mask.n_params} free parameters)",
        ),
        "mask_pc.png": plot_mask(
            pc_mask,
            title=f"Thesis PC Mask ({pc_mask.n_params} free parameters, 19.8% density)",
        ),
        "mask_stability.png": plot_mask(
            stability_mask,
            title=(
                "PC Stability Selection Mask "
                f"({stability_mask.n_params} free parameters, "
                f"B={MASK_PARAMS['B']}, tau={MASK_PARAMS['tau_stab']})"
            ),
        ),
        "class_dependencies.png": plot_class_dependencies(
            clf.model_0_, clf.model_1_, variable_names
        ),
        "difference_heatmap.png": plot_difference_heatmap(
            clf.model_0_, clf.model_1_, variable_names
        ),
        "structure_hand_graph.png": plot_structure_graph(
            hand_mask,
            title=f"Hand-Specified Structure ({hand_mask.n_params} edges)",
            exogenous=MASK_PARAMS["exogenous"],
        ),
        "structure_pc_graph.png": plot_structure_graph(
            pc_mask,
            title=f"PC Structure After Curation ({pc_mask.n_params} edges)",
            exogenous=MASK_PARAMS["exogenous"],
        ),
        "structure_pc_curation.png": plot_structure_curation(
            pc_discovered,
            pc_mask,
            title="PC Discovery and Thesis Domain Curation (age/sex exogenous)",
            exogenous=MASK_PARAMS["exogenous"],
        ),
        "structure_stability_graph.png": plot_structure_graph(
            stability_mask,
            title=(
                "Stability Structure After Curation "
                f"({stability_mask.n_params} edges)"
            ),
            exogenous=MASK_PARAMS["exogenous"],
        ),
        "structure_stability_curation.png": plot_structure_curation(
            stability_discovered,
            stability_mask,
            title="Stability Selection and Domain Curation (age/sex exogenous)",
            exogenous=MASK_PARAMS["exogenous"],
        ),
    }

    for name, fig in figures.items():
        path = FIGURES_DIR / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")

    legacy_mask = FIGURES_DIR / "mask.png"
    figures["mask_hand.png"].savefig(legacy_mask, dpi=150, bbox_inches="tight")
    print(f"Saved: {legacy_mask} (legacy alias)")

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
