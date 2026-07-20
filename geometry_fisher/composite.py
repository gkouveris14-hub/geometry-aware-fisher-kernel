"""
Class-specific Composite Likelihood Model for mixed continuous + ordinal data.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
from scipy.optimize import minimize
from scipy.stats import norm

from .structure import StructuralMask


@dataclass
class CompositeLikelihoodModel:
    """
    Class-conditional composite likelihood model under a structural mask.

    Parameters
    ----------
    mask : StructuralMask
        The structural mask defining allowed dependencies.
    lambda_reg : float
        L2 regularization strength.
    """

    mask: StructuralMask
    lambda_reg: float = 0.01

    # Learned attributes (set after fit)
    theta_hat_: Optional[np.ndarray] = field(default=None, init=False)
    thresholds_: Optional[Dict[int, np.ndarray]] = field(default=None, init=False)
    continuous_idx_: Optional[np.ndarray] = field(default=None, init=False)
    ordinal_idx_: Optional[np.ndarray] = field(default=None, init=False)
    variable_names_: Optional[List[str]] = field(default=None, init=False)
    n_features_: Optional[int] = field(default=None, init=False)
    is_fitted_: bool = field(default=False, init=False)

    def fit(
        self,
        X: np.ndarray,
        continuous_idx: np.ndarray,
        ordinal_idx: np.ndarray,
        variable_names: Optional[List[str]] = None,
        max_iter: int = 500,
    ) -> "CompositeLikelihoodModel":
        """
        Fit the composite likelihood model on data from one class.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        continuous_idx : indices of continuous variables
        ordinal_idx : indices of ordinal variables
        variable_names : optional list of variable names
        """
        X = np.asarray(X, dtype=float)
        self.n_features_ = X.shape[1]
        self.continuous_idx_ = np.asarray(continuous_idx)
        self.ordinal_idx_ = np.asarray(ordinal_idx)
        self.variable_names_ = variable_names

        # 1. Estimate ordinal thresholds from this class data only
        self.thresholds_ = self._estimate_thresholds(X)

        # 2. Initialize theta (active parameters only)
        d = self.mask.n_params
        theta0 = np.zeros(d)

        # 3. Optimize
        result = minimize(
            fun=self._objective,
            x0=theta0,
            args=(X,),
            method="L-BFGS-B",
            options={"maxiter": max_iter, "disp": False},
        )

        self.theta_hat_ = result.x
        self.is_fitted_ = True
        return self

    def _estimate_thresholds(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """Estimate ordinal thresholds from empirical CDF (with continuity correction)."""
        thresholds = {}
        n = X.shape[0]
        eps = 1.0 / (2 * n)

        for j in self.ordinal_idx_:
            values = X[:, j]
            # Assume ordinal variables are coded as 1, 2, ..., M
            categories = np.sort(np.unique(values))
            M = len(categories)
            # Empirical CDF
            ecdf = np.array([np.mean(values <= c) for c in categories[:-1]])
            ecdf = np.clip(ecdf, eps, 1 - eps)
            thresholds[j] = norm.ppf(ecdf)

        return thresholds

    def _theta_to_W(self, theta: np.ndarray) -> np.ndarray:
        """Map the active parameter vector back to a full p×p matrix."""
        W = np.zeros((self.n_features_, self.n_features_))
        active = np.argwhere(self.mask.matrix == 1)
        for idx, (i, j) in enumerate(active):
            W[i, j] = theta[idx]
        return W

    def _negative_log_likelihood(self, theta: np.ndarray, X: np.ndarray) -> float:
        """Compute the composite negative log-likelihood + regularization."""
        W = self._theta_to_W(theta)
        nll = 0.0

        # Continuous part (Gaussian, variance fixed to 1)
        for j in self.continuous_idx_:
            mu = X @ W[j, :]
            resid = X[:, j] - mu
            nll += 0.5 * np.sum(resid ** 2)

        # Ordinal part (probit)
        for j in self.ordinal_idx_:
            mu = X @ W[j, :]
            thresholds = self.thresholds_[j]
            # For simplicity we handle the observed category via the thresholds
            # (full implementation can be refined later)
            for i in range(X.shape[0]):
                # This is a simplified placeholder – we will improve it
                nll += 0.5 * (X[i, j] - mu[i]) ** 2  # temporary

        # L2 regularization
        nll += self.lambda_reg * np.sum(theta ** 2)
        return nll

    def _objective(self, theta: np.ndarray, X: np.ndarray) -> float:
        return self._negative_log_likelihood(theta, X)

    def per_observation_gradient(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-observation gradients at theta_hat.

        Returns
        -------
        gradients : array of shape (n_samples, n_params)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted first.")

        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        d = self.mask.n_params
        grads = np.zeros((n_samples, d))

        # Numerical gradient for now (stable and simple).
        # We can replace with analytic later if needed.
        eps = 1e-5
        theta = self.theta_hat_.copy()

        for i in range(n_samples):
            x_i = X[i : i + 1, :]
            base = self._negative_log_likelihood(theta, x_i)

            for k in range(d):
                theta[k] += eps
                plus = self._negative_log_likelihood(theta, x_i)
                theta[k] -= eps
                grads[i, k] = (plus - base) / eps

        return grads

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "not fitted"
        return f"CompositeLikelihoodModel(n_params={self.mask.n_params}, {status})"