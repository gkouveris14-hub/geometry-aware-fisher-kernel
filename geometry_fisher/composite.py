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
        n_samples = X.shape[0]

        # --------------------------
        # Continuous part (Gaussian, variance = 1)
        # --------------------------
        for j in self.continuous_idx_:
            mu = X @ W[j, :]
            resid = X[:, j] - mu
            nll += 0.5 * np.sum(resid ** 2)

        # --------------------------
        # Ordinal part (ordered probit)
        # --------------------------
        for j in self.ordinal_idx_:
            mu = X @ W[j, :]
            thresholds = self.thresholds_[j]   # shape (M-1,)

            # Observed ordinal values (assumed to be integers 0, 1, 2, ..., M-1 or 1, 2, ..., M)
            y_ord = X[:, j].astype(int)

            # Handle both 0-based and 1-based coding
            min_cat = y_ord.min()
            if min_cat == 1:
                y_ord = y_ord - 1   # convert to 0-based

            M = len(thresholds) + 1

            for i in range(n_samples):
                m = y_ord[i]
                # Lower and upper thresholds for category m
                lower = -np.inf if m == 0 else thresholds[m - 1]
                upper = np.inf if m == M - 1 else thresholds[m]

                # Probability of falling into category m
                # P(lower < Z <= upper) where Z ~ N(mu, 1)
                p = norm.cdf(upper - mu[i]) - norm.cdf(lower - mu[i])
                p = np.clip(p, 1e-10, 1.0)   # numerical safety
                nll -= np.log(p)

        # L2 regularization
        nll += self.lambda_reg * np.sum(theta ** 2)
        return nll

    def _objective(self, theta: np.ndarray, X: np.ndarray) -> float:
        return self._negative_log_likelihood(theta, X)
   
    def per_observation_gradient(self, X: np.ndarray) -> np.ndarray:
        """
        Compute analytic per-observation gradients at theta_hat.

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

        W = self._theta_to_W(self.theta_hat_)
        active = np.argwhere(self.mask.matrix == 1)  # list of (i, j) pairs

        # Pre-compute linear predictors for all variables
        mu = X @ W.T   # shape (n_samples, p)

        for obs in range(n_samples):
            g_obs = np.zeros(d)

            # ----- Continuous variables -----
            for j in self.continuous_idx_:
                resid = X[obs, j] - mu[obs, j]
                # dNLL/dW[j, k] = -resid * X[obs, k]
                for idx, (row, col) in enumerate(active):
                    if row == j:
                        g_obs[idx] += -resid * X[obs, col]

            # ----- Ordinal variables (ordered probit) -----
            for j in self.ordinal_idx_:
                thresholds = self.thresholds_[j]
                y_ord = int(X[obs, j])
                if y_ord >= 1 and y_ord == X[:, j].min():  # handle 1-based
                    y_ord = y_ord - 1
                # safer conversion
                y_val = int(X[obs, j])
                min_cat = int(X[:, j].min())
                if min_cat == 1:
                    y_val -= 1

                M = len(thresholds) + 1
                m = y_val

                lower = -np.inf if m == 0 else thresholds[m - 1]
                upper = np.inf if m == M - 1 else thresholds[m]

                # Score for mu: d/dmu log(Φ(u-μ) - Φ(l-μ)) = - (φ(u-μ) - φ(l-μ)) / p
                phi_upper = norm.pdf(upper - mu[obs, j]) if np.isfinite(upper) else 0.0
                phi_lower = norm.pdf(lower - mu[obs, j]) if np.isfinite(lower) else 0.0
                p = norm.cdf(upper - mu[obs, j]) - norm.cdf(lower - mu[obs, j])
                p = np.clip(p, 1e-12, 1.0)

                d_nll_dmu = (phi_upper - phi_lower) / p   # note: NLL = -log p

                # Chain rule: dNLL/dW[j, k] = dNLL/dmu * X[obs, k]
                for idx, (row, col) in enumerate(active):
                    if row == j:
                        g_obs[idx] += d_nll_dmu * X[obs, col]

            # Regularization gradient (2 * λ * θ) is the same for every observation
            # We distribute it evenly or add it only once later. For per-observation
            # features it is common to omit the regularizer from the score.
            # Here we omit it so that the features reflect pure data contribution.

            grads[obs] = g_obs

        return grads 
   
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "not fitted"
        return f"CompositeLikelihoodModel(n_params={self.mask.n_params}, {status})"