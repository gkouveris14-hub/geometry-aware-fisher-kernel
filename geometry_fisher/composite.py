"""
Composite Likelihood Model – faithful to the original thesis notebook.
- Shared thresholds
- Mask applied in mean structure mu = (W * mask) @ x
- Proper ordered probit with jax.lax.cond
- theta parameterization with Optax + early stopping
- Analytic Hessian for Godambe sensitivity matrix H
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import value_and_grad, jit, vmap, grad, hessian
from jax.scipy.stats import norm
import optax
from scipy.stats import norm as scipy_norm

from .structure import StructuralMask

jax.config.update("jax_enable_x64", True)


def _compute_mu(x, W, mask):
    return (W * mask) @ x


def _continuous_loss_one(x, W, mask, cont_idx):
    mu = _compute_mu(x, W, mask)
    x_cont = x[cont_idx]
    mu_cont = mu[cont_idx]
    return 0.5 * jnp.sum((x_cont - mu_cont) ** 2)


def _ordinal_neglogprob_one(x_j, mu_j, thresholds, categories, eps=1e-8):
    x_j_int = x_j.astype(jnp.int32)
    m = jnp.argmax(categories == x_j_int)
    M = categories.shape[0]

    def first_cat(_):
        return norm.cdf(thresholds[0] - mu_j)

    def last_cat(_):
        return 1.0 - norm.cdf(thresholds[-1] - mu_j)

    def middle_cat(_):
        return norm.cdf(thresholds[m] - mu_j) - norm.cdf(thresholds[m - 1] - mu_j)

    prob = jax.lax.cond(
        m == 0,
        first_cat,
        lambda _: jax.lax.cond(m == M - 1, last_cat, middle_cat, operand=None),
        operand=None,
    )
    prob = jnp.clip(prob, eps, 1.0)
    return -jnp.log(prob)


def _ordinal_loss_one(x, W, mask, ord_idx, thresholds_list, categories_list):
    mu = _compute_mu(x, W, mask)
    total = 0.0
    for i, j in enumerate(ord_idx):
        total = total + _ordinal_neglogprob_one(
            x[j], mu[j], thresholds_list[i], categories_list[i]
        )
    return total


def _total_loss_one(x, W, mask, cont_idx, ord_idx, thresholds_list, categories_list):
    return (
        _continuous_loss_one(x, W, mask, cont_idx)
        + _ordinal_loss_one(x, W, mask, ord_idx, thresholds_list, categories_list)
    )


@dataclass
class CompositeLikelihoodModel:
    mask: StructuralMask
    lambda_reg: float = 0.01
    learning_rate: float = 0.01
    max_iter: int = 800
    early_stop_tol: float = 1e-4
    early_stop_patience: int = 10

    theta_hat_: Optional[np.ndarray] = field(default=None, init=False)
    thresholds_: Optional[Dict[int, np.ndarray]] = field(default=None, init=False)
    continuous_idx_: Optional[np.ndarray] = field(default=None, init=False)
    ordinal_idx_: Optional[np.ndarray] = field(default=None, init=False)
    variable_names_: Optional[List[str]] = field(default=None, init=False)
    n_features_: Optional[int] = field(default=None, init=False)
    is_fitted_: bool = field(default=False, init=False)

    _active_rows: Optional[jnp.ndarray] = field(default=None, init=False, repr=False)
    _active_cols: Optional[jnp.ndarray] = field(default=None, init=False, repr=False)
    _mask_jax: Optional[jnp.ndarray] = field(default=None, init=False, repr=False)
    _thresholds_list: Optional[list] = field(default=None, init=False, repr=False)
    _categories_list: Optional[list] = field(default=None, init=False, repr=False)

    def fit(
        self,
        X: np.ndarray,
        continuous_idx: np.ndarray,
        ordinal_idx: np.ndarray,
        variable_names: Optional[List[str]] = None,
        shared_thresholds: Optional[Dict[int, np.ndarray]] = None,
        shared_categories: Optional[Dict[int, np.ndarray]] = None,
    ) -> "CompositeLikelihoodModel":
        X = np.asarray(X, dtype=np.float64)
        n_samples, self.n_features_ = X.shape
        self.continuous_idx_ = np.asarray(continuous_idx, dtype=np.int32)
        self.ordinal_idx_ = np.asarray(ordinal_idx, dtype=np.int32)
        self.variable_names_ = variable_names
        self._mask_jax = jnp.array(self.mask.matrix, dtype=jnp.float64)

        active = np.argwhere(self.mask.matrix == 1)
        self._active_rows = jnp.array(active[:, 0], dtype=jnp.int32)
        self._active_cols = jnp.array(active[:, 1], dtype=jnp.int32)
        d = active.shape[0]

        if shared_thresholds is None or shared_categories is None:
            shared_thresholds, shared_categories = self._estimate_thresholds_and_cats(X)

        self.thresholds_ = shared_thresholds

        self._thresholds_list = []
        self._categories_list = []
        for j in self.ordinal_idx_:
            self._thresholds_list.append(
                jnp.array(shared_thresholds[int(j)], dtype=jnp.float64)
            )
            self._categories_list.append(
                jnp.array(shared_categories[int(j)], dtype=jnp.int32)
            )

        X_jax = jnp.array(X, dtype=jnp.float64)
        cont_idx = jnp.array(self.continuous_idx_)
        ord_idx = jnp.array(self.ordinal_idx_)
        mask_jax = self._mask_jax
        thresholds_list = self._thresholds_list
        categories_list = self._categories_list
        active_rows = self._active_rows
        active_cols = self._active_cols
        n_features = self.n_features_
        lam = self.lambda_reg

        def class_objective_theta(theta):
            W = jnp.zeros((n_features, n_features), dtype=jnp.float64)
            W = W.at[active_rows, active_cols].set(theta)

            def loss_one(x):
                return _total_loss_one(
                    x, W, mask_jax, cont_idx, ord_idx, thresholds_list, categories_list
                )

            losses = vmap(loss_one)(X_jax)
            reg = lam * jnp.sum(theta ** 2)
            return jnp.sum(losses) + reg

        value_and_grad_fn = jit(value_and_grad(class_objective_theta))
        optimizer = optax.adam(self.learning_rate)
        theta = jnp.zeros(d, dtype=jnp.float64)
        opt_state = optimizer.init(theta)

        best_loss = float("inf")
        best_theta = theta
        wait = 0

        for step in range(self.max_iter):
            loss_val, grads = value_and_grad_fn(theta)
            updates, opt_state = optimizer.update(grads, opt_state, theta)
            theta = optax.apply_updates(theta, updates)

            loss_float = float(loss_val)
            if best_loss - loss_float > self.early_stop_tol:
                best_loss = loss_float
                best_theta = theta
                wait = 0
            else:
                wait += 1
                if wait >= self.early_stop_patience:
                    break

        self.theta_hat_ = np.asarray(best_theta)
        self.is_fitted_ = True
        return self

    def _estimate_thresholds_and_cats(self, X: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        thresholds: Dict[int, np.ndarray] = {}
        categories: Dict[int, np.ndarray] = {}
        n = X.shape[0]
        eps = 1.0 / (2 * n)

        for j in self.ordinal_idx_:
            values = X[:, j].astype(int)
            cats = np.sort(np.unique(values))
            categories[int(j)] = cats

            if len(cats) <= 1:
                thresholds[int(j)] = np.array([])
                continue

            ecdf = np.array([np.mean(values <= c) for c in cats[:-1]])
            ecdf = np.clip(ecdf, eps, 1 - eps)
            thresholds[int(j)] = scipy_norm.ppf(ecdf)

        return thresholds, categories

    def _loss_one_theta(self, theta, x):
        W = jnp.zeros((self.n_features_, self.n_features_), dtype=jnp.float64)
        W = W.at[self._active_rows, self._active_cols].set(theta)
        return _total_loss_one(
            x,
            W,
            self._mask_jax,
            jnp.array(self.continuous_idx_),
            jnp.array(self.ordinal_idx_),
            self._thresholds_list,
            self._categories_list,
        )

    def per_observation_gradient(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted first.")

        X_jax = jnp.array(X, dtype=jnp.float64)
        theta = jnp.array(self.theta_hat_)
        grad_fn = jit(grad(self._loss_one_theta, argnums=0))
        grad_one = lambda x: grad_fn(theta, x)
        return np.asarray(vmap(grad_one)(X_jax))

    def objective_hessian(self, X: np.ndarray) -> np.ndarray:
        """Hessian of the regularized class objective at the fitted theta."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted first.")

        X_jax = jnp.array(X, dtype=jnp.float64)
        cont_idx = jnp.array(self.continuous_idx_)
        ord_idx = jnp.array(self.ordinal_idx_)
        mask_jax = self._mask_jax
        thresholds_list = self._thresholds_list
        categories_list = self._categories_list
        active_rows = self._active_rows
        active_cols = self._active_cols
        n_features = self.n_features_
        lam = self.lambda_reg
        theta_hat = jnp.array(self.theta_hat_)

        def class_objective_theta(theta):
            W = jnp.zeros((n_features, n_features), dtype=jnp.float64)
            W = W.at[active_rows, active_cols].set(theta)

            def loss_one(x):
                return _total_loss_one(
                    x, W, mask_jax, cont_idx, ord_idx, thresholds_list, categories_list
                )

            losses = vmap(loss_one)(X_jax)
            reg = lam * jnp.sum(theta ** 2)
            return jnp.sum(losses) + reg

        H = hessian(class_objective_theta)(theta_hat)
        return np.asarray(H)

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "not fitted"
        return f"CompositeLikelihoodModel(n_params={self.mask.n_params}, {status})"
