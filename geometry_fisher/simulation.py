"""
Synthetic score-geometry study (Experiment 3).

Shows when inverse-Godambe whitening improves sample efficiency relative to
raw composite-style score vectors. The data-generating process controls H, J,
and labels directly so the comparison is not confounded by composite fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from numpy.linalg import inv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .geometry import psd_sqrt, stable_symmetrize


@dataclass
class ScoreGeometryConfig:
    n_samples: int = 120
    d: int = 40
    sparsity: int = 4
    test_size: float = 0.35
    lr_C: float = 0.005
    random_state: int = 0


def _random_spd(d: int, rng: np.random.Generator, floor: float = 0.5) -> np.ndarray:
    """Random symmetric positive-definite matrix."""
    M = rng.normal(size=(d, d))
    return stable_symmetrize(M @ M.T) + floor * np.eye(d)


def hj_discrepancy(H: np.ndarray, J: np.ndarray) -> float:
    """Relative Frobenius norm ||H - J|| / ||H||."""
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
    One replicate of the Experiment 3 score-geometry simulation.

    Data generation (each replicate)
    --------------------------------
    1. Draw sensitivity H and variability J as random SPD matrices in R^{d x d}
       with H != J (composite-likelihood regime).
    2. Form the inverse Godambe metric G^{-1} = H^{-1} J H^{-1} and its
       symmetric square root A so that A^T A = G^{-1}.
    3. Sample n raw score vectors g_i ~ N(0, J).
    4. Build Godambe features tilde_g_i = A g_i.
    5. Draw a sparse label vector w with ``sparsity`` nonzero entries and set
       y_i = 1 if w^T tilde_g_i exceeds the within-sample median (balanced labels).
    6. Stratified train/test split; fit L2-regularized logistic regression on
       raw g and on tilde_g separately (features column-standardized).

    The label rule is linear in Godambe features but induces a dense linear
    separator in raw score space; with strong regularization, Godambe wins at
    small n and converges toward raw as n grows.
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

    results = {"delta_hj": hj_discrepancy(H, J)}
    for name, ptr, pte in [
        ("raw", g_train, g_test),
        ("godambe", phi_train, phi_test),
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
    """Learning-curve summary: raw vs Godambe over sample size n."""
    sample_sizes = sample_sizes or [60, 100, 160, 240, 360]
    out: Dict[int, Dict[str, Dict[str, float]]] = {}
    for n in sample_sizes:
        rows = [
            run_score_geometry_study(
                ScoreGeometryConfig(n_samples=n, random_state=base_seed + rep)
            )
            for rep in range(n_replicates)
        ]
        out[n] = _summarize_rows(rows)
    return out


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
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Run Experiment 3 across sample sizes and replicates."""
    return run_score_geometry_sweep(
        sample_sizes=[60, 100, 160, 240, 360],
        n_replicates=n_replicates,
        base_seed=base_seed,
    )
