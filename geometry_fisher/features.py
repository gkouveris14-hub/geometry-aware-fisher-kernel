"""
Feature construction for the Geometry-Aware Generalized Fisher Kernel.
Supports the full ablation suite (A–E).
"""

from __future__ import annotations

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from scipy.linalg import sqrtm, inv, pinvh

from .geometry import GodambeGeometry


@dataclass
class FisherFeatures:
    raw: np.ndarray            # Variant A: concat(g0, g1)
    fisher_only: np.ndarray    # Variant B: concat(J^{-1/2} g0, J^{-1/2} g1)
    linear: np.ndarray         # Variant C: concat(A0 g0, A1 g1)  (thesis)
    quadratic: np.ndarray      # Variant D: [q0, q1, q_diff]
    full: np.ndarray           # Variant E: linear + quadratic


def _safe_whiten(gradients: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply whitening with a matrix (handles numerical issues)."""
    try:
        # Symmetric square root of the inverse
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.clip(eigvals, 1e-10, None)
        A = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        return gradients @ A.T
    except Exception:
        return gradients


def build_features(
    gradients_0: np.ndarray,
    gradients_1: np.ndarray,
    geometry_0: GodambeGeometry,
    geometry_1: GodambeGeometry,
) -> FisherFeatures:
    """
    Build all ablation feature sets.
    """
    # --- Variant A: raw gradients ---
    phi_raw = np.hstack([gradients_0, gradients_1])

    # --- Variant B: Fisher-only (using J only) ---
    # We use the shrunk J from the geometry objects
    J0 = geometry_0.J_shrunk_ if geometry_0.J_shrunk_ is not None else geometry_0.J_
    J1 = geometry_1.J_shrunk_ if geometry_1.J_shrunk_ is not None else geometry_1.J_

    g0_fisher = _safe_whiten(gradients_0, J0)
    g1_fisher = _safe_whiten(gradients_1, J1)
    phi_fisher_only = np.hstack([g0_fisher, g1_fisher])

    # --- Variant C: full Godambe linear (thesis original) ---
    g_tilde_0 = geometry_0.transform(gradients_0)
    g_tilde_1 = geometry_1.transform(gradients_1)
    phi_linear = np.hstack([g_tilde_0, g_tilde_1])

    # --- Variant D: quadratic forms ---
    q0 = geometry_0.quadratic_form(gradients_0)
    q1 = geometry_1.quadratic_form(gradients_1)
    q_diff = q1 - q0
    phi_quadratic = np.column_stack([q0, q1, q_diff])

    # --- Variant E: full augmented ---
    phi_full = np.hstack([phi_linear, phi_quadratic])

    return FisherFeatures(
        raw=phi_raw,
        fisher_only=phi_fisher_only,
        linear=phi_linear,
        quadratic=phi_quadratic,
        full=phi_full,
    )