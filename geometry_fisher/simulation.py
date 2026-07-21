"""
Synthetic studies showing when inverse-Godambe whitening matters.

Tier 1 (score_geometry)
    Direct control of H, J, and score vectors. Labels are a sparse linear
    functional of Godambe-whitened scores. Because the separator is sparse in
    whitened space but dense in raw score space, regularized linear classifiers
    need more data when fed raw gradients.

Tier 2 (composite_hetero)
    Full composite-likelihood pipeline with heteroscedastic generation and a
    sparser fitted mask (H != J in practice). Harder setting; useful as a
    secondary, realistic stress test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import inv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .composite import CompositeLikelihoodModel
from .geometry import GodambeGeometry, psd_sqrt, stable_symmetrize
from .structure import StructuralMask


@dataclass
class ScoreGeometryConfig:
    n_samples: int = 120
    d: int = 40
    sparsity: int = 4
    test_size: float = 0.35
    lr_C: float = 0.005
    random_state: int = 0


@dataclass
class CompositeSimulationConfig:
    n_per_class: int = 250
    p: int = 8
    test_size: float = 0.35
    lambda_reg: float = 0.01
    ridge_gamma: float = 1e-3
    lr_C: float = 0.05
    random_state: int = 0
    hetero_strength: float = 2.5
    outlier_frac: float = 0.08
    outlier_scale: float = 6.0


@dataclass
class FeatureBundle:
    phi_raw: np.ndarray
    phi_godambe: np.ndarray
    phi_hessian: np.ndarray
    delta_hj_0: float
    delta_hj_1: float


def _random_spd(d: int, rng: np.random.Generator, floor: float = 0.5) -> np.ndarray:
    M = rng.normal(size=(d, d))
    return stable_symmetrize(M @ M.T) + floor * np.eye(d)


def _hj_discrepancy(H: np.ndarray, J: np.ndarray) -> float:
    Hs = stable_symmetrize(H)
    Js = stable_symmetrize(J)
    denom = np.linalg.norm(Hs, ord="fro")
    if denom < 1e-12:
        return 0.0
    return float(np.linalg.norm(Hs - Js, ord="fro") / denom)


def _evaluate_linear_classifier(
    phi_train: np.ndarray,
    y_train: np.ndarray,
    phi_test: np.ndarray,
    y_test: np.ndarray,
    *,
    C: float = 1.0,
) -> Dict[str, float]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=8000, solver="lbfgs", random_state=42),
    )
    clf.fit(phi_train, y_train)
    proba = clf.predict_proba(phi_test)[:, 1]
    pred = clf.predict(phi_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "auc": float(roc_auc_score(y_test, proba)),
    }


def run_score_geometry_study(config: ScoreGeometryConfig) -> Dict[str, float]:
    """
    Tier 1: known H, J, sparse separator in Godambe-whitened score space.

    Each replicate draws scores g ~ N(0, J), builds A = (H^{-1} J H^{-1})^{1/2},
    and sets labels from a sparse vector w acting on Ag. Raw and Godambe features
    encode the same linear separator, but the raw-space coefficient vector is
    dense; with strong L2 regularization Godambe wins at small sample size.
    """
    rng = np.random.default_rng(config.random_state)
    H = _random_spd(config.d, rng)
    J = _random_spd(config.d, rng)
    Ginv = inv(H) @ J @ inv(H)
    A = psd_sqrt(Ginv)

    w = np.zeros(config.d, dtype=float)
    active = rng.choice(config.d, size=config.sparsity, replace=False)
    w[active] = rng.normal(0.0, 1.0, size=config.sparsity)

    g = rng.multivariate_normal(np.zeros(config.d), J, size=config.n_samples)
    phi_godambe = g @ A.T
    logits = phi_godambe @ w
    y = (logits > np.median(logits)).astype(int)

    g_train, g_test, phi_train, phi_test, y_train, y_test = train_test_split(
        g,
        phi_godambe,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    A_hess = psd_sqrt(inv(H + 1e-3 * np.eye(config.d)))
    phi_hessian = g @ A_hess.T
    phi_hess_train = g_train @ A_hess.T
    phi_hess_test = g_test @ A_hess.T

    results = {"delta_hj": _hj_discrepancy(H, J)}
    for name, ptr, pte in [
        ("raw", g_train, g_test),
        ("godambe", phi_train, phi_test),
        ("hessian", phi_hess_train, phi_hess_test),
    ]:
        metrics = _evaluate_linear_classifier(
            ptr,
            y_train,
            pte,
            y_test,
            C=config.lr_C,
        )
        results[f"{name}_accuracy"] = metrics["accuracy"]
        results[f"{name}_auc"] = metrics["auc"]
    return results


def run_score_geometry_sweep(
    sample_sizes: Optional[List[int]] = None,
    n_replicates: int = 30,
    base_seed: int = 0,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Learning-curve summary for Tier 1."""
    sample_sizes = sample_sizes or [60, 100, 160, 240, 360]
    out: Dict[int, Dict[str, Dict[str, float]]] = {}
    for n in sample_sizes:
        rows = []
        for rep in range(n_replicates):
            cfg = ScoreGeometryConfig(n_samples=n, random_state=base_seed + rep)
            rows.append(run_score_geometry_study(cfg))
        out[n] = _summarize_rows(rows)
    return out


def chain_mask(p: int, extra_edges: Optional[List[Tuple[int, int]]] = None) -> StructuralMask:
    matrix = np.zeros((p, p), dtype=int)
    names = [f"x{i}" for i in range(p)]
    for i in range(1, p):
        matrix[i, i - 1] = 1
    if extra_edges:
        for target, source in extra_edges:
            matrix[target, source] = 1
    return StructuralMask(matrix=matrix, variable_names=names)


def generative_mask(p: int, rng: np.random.Generator) -> StructuralMask:
    extra = []
    for _ in range(max(2, p // 2)):
        source = int(rng.integers(0, p - 2))
        target = int(rng.integers(source + 2, p))
        extra.append((target, source))
    return chain_mask(p, extra_edges=extra)


def _topological_order(mask: StructuralMask) -> List[int]:
    p = mask.p
    remaining = set(range(p))
    order = []
    while remaining:
        ready = [
            i
            for i in remaining
            if all(mask.matrix[i, j] == 0 for j in remaining if j != i)
        ]
        i = min(ready or [min(remaining)])
        order.append(i)
        remaining.remove(i)
    return order


def _sample_from_weight_matrix(
    n: int,
    W: np.ndarray,
    mask: StructuralMask,
    rng: np.random.Generator,
    *,
    heteroscedastic: bool = False,
    hetero_strength: float = 2.0,
    outlier_frac: float = 0.0,
    outlier_scale: float = 5.0,
) -> np.ndarray:
    p = mask.p
    X = np.zeros((n, p), dtype=float)
    order = _topological_order(mask)
    for row in range(n):
        x = np.zeros(p, dtype=float)
        for i in order:
            parents = np.where(mask.matrix[i])[0]
            mu = float(np.sum(W[i, parents] * x[parents])) if len(parents) else 0.0
            if i == 0:
                mu += rng.normal(0.0, 0.5)
            var = 1.0
            if heteroscedastic and len(parents):
                var = 1.0 + hetero_strength * float(np.mean(x[parents] ** 2))
            x[i] = mu + rng.normal(0.0, np.sqrt(var))
        if outlier_frac > 0 and rng.random() < outlier_frac:
            j = int(rng.integers(0, p))
            x[j] += rng.normal(0.0, outlier_scale)
        X[row] = x
    return X


def _random_weight_matrix(mask: StructuralMask, rng: np.random.Generator, scale: float = 0.8) -> np.ndarray:
    p = mask.p
    W = np.zeros((p, p), dtype=float)
    for i, j in np.argwhere(mask.matrix == 1):
        W[i, j] = rng.normal(0.0, scale)
    return W


def generate_class_conditional_dataset(
    config: CompositeSimulationConfig,
    gen_mask: StructuralMask,
    fit_mask: Optional[StructuralMask] = None,
    *,
    heteroscedastic: bool = False,
    outlier_frac: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, StructuralMask, np.ndarray]:
    rng = np.random.default_rng(config.random_state)
    fit_mask = fit_mask or gen_mask
    p = config.p
    continuous_idx = np.arange(p, dtype=int)
    ordinal_idx = np.array([], dtype=int)

    W0 = _random_weight_matrix(gen_mask, rng, scale=0.7)
    W1 = _random_weight_matrix(gen_mask, rng, scale=0.7)
    X0 = _sample_from_weight_matrix(
        config.n_per_class,
        W0,
        gen_mask,
        rng,
        heteroscedastic=heteroscedastic,
        hetero_strength=config.hetero_strength,
        outlier_frac=outlier_frac,
        outlier_scale=config.outlier_scale,
    )
    X1 = _sample_from_weight_matrix(
        config.n_per_class,
        W1,
        gen_mask,
        rng,
        heteroscedastic=heteroscedastic,
        hetero_strength=config.hetero_strength,
        outlier_frac=outlier_frac,
        outlier_scale=config.outlier_scale,
    )
    X = np.vstack([X0, X1])
    y = np.array([0] * config.n_per_class + [1] * config.n_per_class, dtype=int)
    perm = rng.permutation(len(y))
    return X[perm], y[perm], fit_mask, continuous_idx


def fit_feature_bundle(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    mask: StructuralMask,
    continuous_idx: np.ndarray,
    ordinal_idx: np.ndarray,
    *,
    lambda_reg: float = 0.01,
    ridge_gamma: float = 1e-3,
) -> FeatureBundle:
    variable_names = mask.variable_names
    X0 = X_train[y_train == 0]
    X1 = X_train[y_train == 1]

    temp = CompositeLikelihoodModel(mask=mask, lambda_reg=lambda_reg)
    temp.continuous_idx_ = continuous_idx
    temp.ordinal_idx_ = ordinal_idx
    shared_thresholds, shared_categories = temp._estimate_thresholds_and_cats(X_train)

    model_0 = CompositeLikelihoodModel(mask=mask, lambda_reg=lambda_reg, max_iter=600)
    model_1 = CompositeLikelihoodModel(mask=mask, lambda_reg=lambda_reg, max_iter=800)
    model_0.fit(
        X0,
        continuous_idx,
        ordinal_idx,
        variable_names,
        shared_thresholds=shared_thresholds,
        shared_categories=shared_categories,
    )
    model_1.fit(
        X1,
        continuous_idx,
        ordinal_idx,
        variable_names,
        shared_thresholds=shared_thresholds,
        shared_categories=shared_categories,
    )

    grads_0_train = model_0.per_observation_gradient(X0)
    grads_1_train = model_1.per_observation_gradient(X1)
    H0 = model_0.objective_hessian(X0)
    H1 = model_1.objective_hessian(X1)

    geom_0 = GodambeGeometry(gradients=grads_0_train, H=H0, ridge_gamma=ridge_gamma).fit()
    geom_1 = GodambeGeometry(gradients=grads_1_train, H=H1, ridge_gamma=ridge_gamma).fit()

    g0 = model_0.per_observation_gradient(X_eval)
    g1 = model_1.per_observation_gradient(X_eval)

    phi_raw = np.hstack([g0, g1])
    phi_godambe = np.hstack([geom_0.transform(g0), geom_1.transform(g1)])

    def hessian_features(H: np.ndarray, g: np.ndarray) -> np.ndarray:
        d = H.shape[0]
        H_reg = stable_symmetrize(H) + ridge_gamma * np.eye(d)
        A = psd_sqrt(inv(H_reg))
        return g @ A.T

    phi_hessian = np.hstack([hessian_features(H1, g1), hessian_features(H0, g0)])

    return FeatureBundle(
        phi_raw=phi_raw,
        phi_godambe=phi_godambe,
        phi_hessian=phi_hessian,
        delta_hj_0=_hj_discrepancy(H0, geom_0.J_),
        delta_hj_1=_hj_discrepancy(H1, geom_1.J_),
    )


def run_composite_hetero_study(config: CompositeSimulationConfig) -> Dict[str, float]:
    """Tier 2: full composite pipeline under heteroscedastic misspecification."""
    rng = np.random.default_rng(config.random_state + 1)
    gen_mask = generative_mask(config.p, rng)
    fit_mask = chain_mask(config.p)

    X, y, _, cont_idx = generate_class_conditional_dataset(
        config,
        gen_mask=gen_mask,
        fit_mask=fit_mask,
        heteroscedastic=True,
        outlier_frac=config.outlier_frac,
    )
    ord_idx = np.array([], dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )

    bundle_train = fit_feature_bundle(
        X_train,
        y_train,
        X_train,
        fit_mask,
        cont_idx,
        ord_idx,
        lambda_reg=config.lambda_reg,
        ridge_gamma=config.ridge_gamma,
    )
    bundle_test = fit_feature_bundle(
        X_train,
        y_train,
        X_test,
        fit_mask,
        cont_idx,
        ord_idx,
        lambda_reg=config.lambda_reg,
        ridge_gamma=config.ridge_gamma,
    )

    results = {
        "delta_hj_mean": 0.5 * (bundle_train.delta_hj_0 + bundle_train.delta_hj_1),
    }
    for name, phi_tr, phi_te in [
        ("raw", bundle_train.phi_raw, bundle_test.phi_raw),
        ("godambe", bundle_train.phi_godambe, bundle_test.phi_godambe),
        ("hessian", bundle_train.phi_hessian, bundle_test.phi_hessian),
    ]:
        metrics = _evaluate_linear_classifier(
            phi_tr,
            y_train,
            phi_te,
            y_test,
            C=config.lr_C,
        )
        results[f"{name}_accuracy"] = metrics["accuracy"]
        results[f"{name}_auc"] = metrics["auc"]
    return results


def _summarize_rows(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = rows[0].keys()
    out: Dict[str, Dict[str, float]] = {}
    for key in keys:
        vals = np.array([row[key] for row in rows], dtype=float)
        out[key] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return out


def run_simulation_battery(
    n_replicates: int = 30,
    base_seed: int = 0,
) -> Tuple[Dict[int, Dict[str, Dict[str, float]]], Dict[str, Dict[str, float]]]:
    score_sweep = run_score_geometry_sweep(
        sample_sizes=[60, 100, 160, 240, 360],
        n_replicates=n_replicates,
        base_seed=base_seed,
    )
    hetero_rows = [
        run_composite_hetero_study(CompositeSimulationConfig(random_state=base_seed + rep))
        for rep in range(n_replicates)
    ]
    return score_sweep, _summarize_rows(hetero_rows)
