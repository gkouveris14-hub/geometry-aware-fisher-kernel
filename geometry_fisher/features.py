"""Feature construction for the Geometry-Aware Fisher Kernel."""

from __future__ import annotations

import warnings

import numpy as np
from dataclasses import dataclass

from .geometry import GodambeGeometry

FEATURE_TYPES = ("raw", "fisher_only", "godambe", "quadratic", "full")
FEATURE_TYPE_ALIASES = {"linear": "godambe"}


def normalize_feature_type(feature_type: str) -> str:
    """Return the canonical feature type, accepting legacy aliases."""
    canonical = FEATURE_TYPE_ALIASES.get(feature_type, feature_type)
    if feature_type in FEATURE_TYPE_ALIASES:
        warnings.warn(
            f"feature_type='{feature_type}' is deprecated; use 'godambe' instead.",
            FutureWarning,
            stacklevel=3,
        )
    if canonical not in FEATURE_TYPES:
        accepted = ", ".join(FEATURE_TYPES + tuple(FEATURE_TYPE_ALIASES))
        raise ValueError(
            f"Unknown feature_type: '{feature_type}'. Choose from: {accepted}"
        )
    return canonical


def requires_geometry(feature_type: str) -> bool:
    return normalize_feature_type(feature_type) != "raw"


def _safe_whiten(gradients: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 1e-10, None)
    A = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return gradients @ A.T


@dataclass
class FisherFeatures:
    raw: np.ndarray
    fisher_only: np.ndarray
    godambe: np.ndarray
    quadratic: np.ndarray
    full: np.ndarray


def build_features(
    gradients_0: np.ndarray,
    gradients_1: np.ndarray,
    geometry_0: GodambeGeometry,
    geometry_1: GodambeGeometry,
) -> FisherFeatures:
    """Build all feature variants (used by exploratory ablation scripts)."""
    phi_raw = np.hstack([gradients_0, gradients_1])

    J0 = geometry_0.J_shrunk_ if geometry_0.J_shrunk_ is not None else geometry_0.J_
    J1 = geometry_1.J_shrunk_ if geometry_1.J_shrunk_ is not None else geometry_1.J_
    phi_fisher_only = np.hstack([
        _safe_whiten(gradients_0, J0),
        _safe_whiten(gradients_1, J1),
    ])

    g_tilde_0 = geometry_0.transform(gradients_0)
    g_tilde_1 = geometry_1.transform(gradients_1)
    phi_godambe = np.hstack([g_tilde_0, g_tilde_1])

    q0 = geometry_0.quadratic_form(gradients_0)
    q1 = geometry_1.quadratic_form(gradients_1)
    phi_quadratic = np.column_stack([q0, q1, q1 - q0])
    phi_full = np.hstack([phi_godambe, phi_quadratic])

    return FisherFeatures(
        raw=phi_raw,
        fisher_only=phi_fisher_only,
        godambe=phi_godambe,
        quadratic=phi_quadratic,
        full=phi_full,
    )


def build_feature_matrix(
    feature_type: str,
    gradients_0: np.ndarray,
    gradients_1: np.ndarray,
    geometry_0: GodambeGeometry | None = None,
    geometry_1: GodambeGeometry | None = None,
) -> np.ndarray:
    """Build the feature matrix for a single requested feature type."""
    canonical = normalize_feature_type(feature_type)

    if canonical == "raw":
        return np.hstack([gradients_0, gradients_1])

    if geometry_0 is None or geometry_1 is None:
        raise ValueError(
            f"Godambe geometry is required for feature_type='{canonical}'."
        )

    features = build_features(gradients_0, gradients_1, geometry_0, geometry_1)
    return getattr(features, canonical)
