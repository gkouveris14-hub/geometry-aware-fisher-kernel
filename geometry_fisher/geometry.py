"""
Godambe geometry: sensitivity, variability, shrinkage, and whitening.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, field
from scipy.linalg import sqrtm, inv, pinvh


def ledoit_wolf_shrinkage(J: np.ndarray, n_samples: int) -> Tuple[np.ndarray, float]:
    """
    Simple Ledoit-Wolf style shrinkage of a covariance-like matrix toward
    a scaled identity matrix.

    Returns
    -------
    J_shrunk : np.ndarray
    delta : float
        Shrinkage intensity (0 = no shrinkage, 1 = full shrinkage).
    """
    d = J.shape[0]
    # Target: scaled identity
    mu = np.trace(J) / d
    target = mu * np.eye(d)

    # Frobenius norms
    frobenius_J = np.sum(J ** 2)
    frobenius_diff = np.sum((J - target) ** 2)

    # Simple estimate of shrinkage intensity (can be refined)
    # This is a practical approximation suitable for our sample sizes
    delta = min(1.0, max(0.0, (frobenius_J / n_samples) / (frobenius_diff + 1e-12)))

    J_shrunk = (1 - delta) * J + delta * target
    return J_shrunk, delta


@dataclass
class GodambeGeometry:
    """
    Computes and stores the Godambe geometry for a fitted CompositeLikelihoodModel.
    """

    # Inputs
    gradients: np.ndarray          # shape (n_samples, n_params)
    ridge_gamma: float = 1e-4      # ridge for H

    # Learned / computed
    H_: Optional[np.ndarray] = field(default=None, init=False)
    J_: Optional[np.ndarray] = field(default=None, init=False)
    J_shrunk_: Optional[np.ndarray] = field(default=None, init=False)
    delta_: Optional[float] = field(default=None, init=False)
    G_inv_: Optional[np.ndarray] = field(default=None, init=False)
    A_: Optional[np.ndarray] = field(default=None, init=False)  # whitening matrix
    is_fitted_: bool = field(default=False, init=False)

    def fit(self) -> "GodambeGeometry":
        """
        Estimate H, J (with shrinkage), G^{-1} and the whitening matrix A.
        """
        n_samples, d = self.gradients.shape

        # Variability matrix J = empirical second-moment of gradients
        self.J_ = (self.gradients.T @ self.gradients) / n_samples

        # Shrink J
        self.J_shrunk_, self.delta_ = ledoit_wolf_shrinkage(self.J_, n_samples)

        # Sensitivity matrix H
        # We use a simple approximation: H ≈ average outer product of gradients
        # (more accurate analytic Hessian can be added later)
        # For stability we also add a ridge.
        self.H_ = self.J_.copy() + self.ridge_gamma * np.eye(d)

        # Inverse Godambe: G^{-1} = H^{-1} J H^{-1}
        try:
            H_inv = inv(self.H_)
        except np.linalg.LinAlgError:
            H_inv = pinvh(self.H_)

        self.G_inv_ = H_inv @ self.J_shrunk_ @ H_inv

        # Whitening matrix A such that A.T @ A ≈ G^{-1}
        # We compute the symmetric square root of G_inv
        # with a small eigenvalue floor for numerical safety
        eigvals, eigvecs = np.linalg.eigh(self.G_inv_)
        eigvals = np.clip(eigvals, 1e-10, None)          # floor
        self.A_ = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        self.is_fitted_ = True
        return self

    def transform(self, gradients: np.ndarray) -> np.ndarray:
        """
        Apply the whitening transformation: g_tilde = A @ g
        """
        if not self.is_fitted_:
            raise RuntimeError("GodambeGeometry must be fitted first.")
        return gradients @ self.A_.T

    def quadratic_form(self, gradients: np.ndarray) -> np.ndarray:
        """
        Compute the quadratic form q(x) = ||A g(x)||^2
        for each observation.
        """
        g_tilde = self.transform(gradients)
        return np.sum(g_tilde ** 2, axis=1)

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "not fitted"
        delta_str = f"{self.delta_:.3f}" if self.delta_ is not None else "None"
        return f"GodambeGeometry(delta={delta_str}, {status})"