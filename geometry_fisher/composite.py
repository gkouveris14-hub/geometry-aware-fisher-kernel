"""
Class-specific Composite Likelihood Model using JAX + automatic differentiation.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import value_and_grad, jit, grad
import optax

from .structure import StructuralMask

jax.config.update("jax_enable_x64", True)


@dataclass
class CompositeLikelihoodModel:
    mask: StructuralMask
    lambda_reg: float = 0.01
    learning_rate: float = 0.05
    max_iter: int = 600

    theta_hat_: Optional[np.ndarray] = field(default=None, init=False)
    thresholds_: Optional[Dict[int, np.ndarray]] = field(default=None, init=False)
    continuous_idx_: Optional[np.ndarray] = field(default=None, init=False)
    ordinal_idx_: Optional[np.ndarray] = field(default=None, init=False)
    variable_names_: Optional[List[str]] = field(default=None, init=False)
    n_features_: Optional[int] = field(default=None, init=False)
    is_fitted_: bool = field(default=False, init=False)

    _active: Optional[jnp.ndarray] = field(default=None, init=False, repr=False)

    def fit(
        self,
        X: np.ndarray,
        continuous_idx: np.ndarray,
        ordinal_idx: np.ndarray,
        variable_names: Optional[List[str]] = None,
    ) -> "CompositeLikelihoodModel":
        X = np.asarray(X, dtype=np.float64)
        self.n_features_ = X.shape[1]
        self.continuous_idx_ = np.asarray(continuous_idx)
        self.ordinal_idx_ = np.asarray(ordinal_idx)
        self.variable_names_ = variable_names

        # Thresholds (kept for later use / diagnostics)
        self.thresholds_ = self._estimate_thresholds(X)

        active = np.argwhere(self.mask.matrix == 1)
        self._active = jnp.array(active)

        d = self.mask.n_params
        theta0 = jnp.zeros(d)

        X_jax = jnp.array(X)
        cont_idx = jnp.array(self.continuous_idx_)
        ord_idx = jnp.array(self.ordinal_idx_)

        def loss_fn(theta):
            return _nll_jax(theta, X_jax, cont_idx, ord_idx, self._active, self.lambda_reg, self.n_features_)

        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(theta0)

        @jit
        def step(theta, opt_state):
            loss, grads = value_and_grad(loss_fn)(theta)
            updates, opt_state = optimizer.update(grads, opt_state, theta)
            theta = optax.apply_updates(theta, updates)
            return theta, opt_state, loss

        theta = theta0
        for i in range(self.max_iter):
            theta, opt_state, loss = step(theta, opt_state)

        self.theta_hat_ = np.array(theta)
        self.is_fitted_ = True
        return self

    def _estimate_thresholds(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        from scipy.stats import norm
        thresholds = {}
        n = X.shape[0]
        eps = 1.0 / (2 * n)
        for j in self.ordinal_idx_:
            values = X[:, j]
            categories = np.sort(np.unique(values))
            if len(categories) <= 1:
                thresholds[j] = np.array([])
                continue
            ecdf = np.array([np.mean(values <= c) for c in categories[:-1]])
            ecdf = np.clip(ecdf, eps, 1 - eps)
            thresholds[j] = norm.ppf(ecdf)
        return thresholds

    def per_observation_gradient(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted first.")

        X_jax = jnp.array(X, dtype=jnp.float64)
        theta = jnp.array(self.theta_hat_)
        cont_idx = jnp.array(self.continuous_idx_)
        ord_idx = jnp.array(self.ordinal_idx_)
        active = self._active
        n_features = self.n_features_
        lam = self.lambda_reg

        def single_loss(theta, x_i):
            return _nll_jax(theta, x_i, cont_idx, ord_idx, active, lam, n_features)

        grad_fn = jit(grad(single_loss, argnums=0))

        grads = []
        for i in range(X.shape[0]):
            g = grad_fn(theta, X_jax[i:i+1])
            grads.append(np.array(g))
        return np.stack(grads, axis=0)

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "not fitted"
        return f"CompositeLikelihoodModel(n_params={self.mask.n_params}, {status})"


def _nll_jax(theta, X, cont_idx, ord_idx, active, lambda_reg, n_features):
    """Pure JAX negative log-likelihood (no Python side effects)."""
    W = jnp.zeros((n_features, n_features))
    W = W.at[active[:, 0], active[:, 1]].set(theta)

    nll = 0.0

    # Continuous variables
    def cont_body(j, nll):
        mu = X @ W[j]
        resid = X[:, j] - mu
        return nll + 0.5 * jnp.sum(resid ** 2)

    nll = jax.lax.fori_loop(0, cont_idx.shape[0], lambda i, nll: cont_body(cont_idx[i], nll), nll)

    # Ordinal variables – currently using a Gaussian approximation for speed & stability
    # (can be replaced with proper probit later)
    def ord_body(j, nll):
        mu = X @ W[j]
        resid = X[:, j] - mu
        return nll + 0.5 * jnp.sum(resid ** 2)

    nll = jax.lax.fori_loop(0, ord_idx.shape[0], lambda i, nll: ord_body(ord_idx[i], nll), nll)

    # Regularization
    nll = nll + lambda_reg * jnp.sum(theta ** 2)
    return nll