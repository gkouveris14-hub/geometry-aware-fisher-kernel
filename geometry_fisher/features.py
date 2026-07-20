"""
Feature construction for the Geometry-Aware Generalized Fisher Kernel.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass

from .geometry import GodambeGeometry


@dataclass
class FisherFeatures:
    """
    Container for the different feature variants used in the ablations.
    """
    linear: np.ndarray          # shape (n_samples, 2*d)   – original thesis features
    quadratic: np.ndarray       # shape (n_samples, 3)     – [q0, q1, q_diff]
    full: np.ndarray            # shape (n_samples, 2*d+3) – linear + quadratic


def build_features(
    gradients_0: np.ndarray,
    gradients_1: np.ndarray,
    geometry_0: GodambeGeometry,
    geometry_1: GodambeGeometry,
) -> FisherFeatures:
    """
    Build the three feature sets from class-conditional gradients and geometries.

    Parameters
    ----------
    gradients_0, gradients_1 : arrays of shape (n_samples, n_params)
        Per-observation gradients under the two class models.
    geometry_0, geometry_1 : fitted GodambeGeometry objects

    Returns
    -------
    FisherFeatures
    """
    # Whitened gradients
    g_tilde_0 = geometry_0.transform(gradients_0)
    g_tilde_1 = geometry_1.transform(gradients_1)

    # Linear features (original thesis)
    phi_linear = np.hstack([g_tilde_0, g_tilde_1])

    # Quadratic / Mahalanobis features
    q0 = geometry_0.quadratic_form(gradients_0)
    q1 = geometry_1.quadratic_form(gradients_1)
    q_diff = q1 - q0
    phi_quadratic = np.column_stack([q0, q1, q_diff])

    # Full augmented features
    phi_full = np.hstack([phi_linear, phi_quadratic])

    return FisherFeatures(
        linear=phi_linear,
        quadratic=phi_quadratic,
        full=phi_full,
    )