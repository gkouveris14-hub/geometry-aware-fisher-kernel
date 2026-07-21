"""
Godambe geometry: sensitivity (H), variability (J), and whitening.
Matches the thesis sandwich transform: G = H^{-1} J H^{-1}, A = psd_sqrt(G).
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from numpy.linalg import inv


def stable_symmetrize(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    return 0.5 * (M + M.T)


def psd_sqrt(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    M = stable_symmetrize(M)
    evals, evecs = np.linalg.eigh(M)
    evals = np.clip(evals, eps, None)
    return evecs @ np.diag(np.sqrt(evals)) @ evecs.T


@dataclass
class GodambeGeometry:
    """
    Godambe sandwich geometry for a fitted class-conditional model.
    """

    gradients: np.ndarray
    H: np.ndarray
    ridge_gamma: float = 1e-3

    H_: Optional[np.ndarray] = field(default=None, init=False)
    J_: Optional[np.ndarray] = field(default=None, init=False)
    G_inv_: Optional[np.ndarray] = field(default=None, init=False)
    A_: Optional[np.ndarray] = field(default=None, init=False)
    is_fitted_: bool = field(default=False, init=False)

    def fit(self) -> "GodambeGeometry":
        n_samples, d = self.gradients.shape

        self.H_ = stable_symmetrize(self.H)
        self.J_ = stable_symmetrize((self.gradients.T @ self.gradients) / n_samples)

        H_reg = self.H_ + self.ridge_gamma * np.eye(d)
        H_inv = inv(H_reg)

        self.G_inv_ = H_inv @ self.J_ @ H_inv
        self.A_ = psd_sqrt(self.G_inv_)

        self.is_fitted_ = True
        return self

    def transform(self, gradients: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("GodambeGeometry must be fitted first.")
        return gradients @ self.A_.T

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "not fitted"
        return f"GodambeGeometry({status})"
