"""Feature construction for the Geometry-Aware Fisher Kernel."""

from __future__ import annotations

import numpy as np

from .geometry import GodambeGeometry

FEATURE_TYPES = ("raw", "godambe")


def normalize_feature_type(feature_type: str) -> str:
    """Return the canonical feature type."""
    if feature_type not in FEATURE_TYPES:
        raise ValueError(
            f"Unknown feature_type: '{feature_type}'. "
            f"Choose from: {', '.join(FEATURE_TYPES)}"
        )
    return feature_type


def requires_geometry(feature_type: str) -> bool:
    return normalize_feature_type(feature_type) != "raw"


def build_feature_matrix(
    feature_type: str,
    gradients_0: np.ndarray,
    gradients_1: np.ndarray,
    geometry_0: GodambeGeometry | None = None,
    geometry_1: GodambeGeometry | None = None,
) -> np.ndarray:
    """Build Phi(x) from class-specific composite-likelihood gradients."""
    canonical = normalize_feature_type(feature_type)

    if canonical == "raw":
        return np.hstack([gradients_0, gradients_1])

    if geometry_0 is None or geometry_1 is None:
        raise ValueError(
            "Godambe geometry is required for feature_type='godambe'."
        )

    phi_godambe = np.hstack([
        geometry_0.transform(gradients_0),
        geometry_1.transform(gradients_1),
    ])
    return phi_godambe
