"""
Godambe geometry: sensitivity (H), variability (J), and whitening.

Literature Godambe information: G = H J^{-1} H.
This module whitens using the inverse sandwich G^{-1} = H^{-1} J H^{-1}
(with ridge on H), then A = psd_sqrt(G^{-1}).
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
    Whitening geometry from the inverse Godambe sandwich G^{-1} = H^{-1} J H^{-1}.
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
