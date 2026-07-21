"""Verify implementation matches the thesis notebook protocol."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from geometry_fisher.data import load_heart_disease
from geometry_fisher.structure import StructuralMask
from geometry_fisher.composite import CompositeLikelihoodModel
from geometry_fisher.geometry import GodambeGeometry, stable_symmetrize


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "heart_disease_uci.csv"


@pytest.fixture(scope="module")
def heart_data():
    return load_heart_disease(
        path=str(DATA_PATH),
        binary_target=True,
        verbose=False,
    )


def test_dataset_size_matches_thesis_protocol(heart_data):
    X, y, _, _, _ = heart_data
    assert X.shape == (531, 9)
    assert np.array_equal(np.bincount(y), [207, 324])


def test_slope_encoding_matches_thesis(heart_data):
    X, _, variable_names, _, ordinal_idx = heart_data
    slope_idx = list(variable_names).index("slope")
    assert slope_idx == int(ordinal_idx[-1])
    assert set(X[:, slope_idx].astype(int)) == {0, 1, 2}


def test_hand_mask_has_56_parameters(heart_data):
    _, _, variable_names, _, _ = heart_data
    mask = StructuralMask.from_domain_knowledge(
        variable_names=variable_names,
        exogenous=["age", "sex"],
    )
    assert mask.n_params == 56


def test_godambe_uses_hessian_not_jacobian_covariance(heart_data):
    X, y, variable_names, continuous_idx, ordinal_idx = heart_data
    mask = StructuralMask.from_domain_knowledge(
        variable_names=variable_names,
        exogenous=["age", "sex"],
    )

    scaler_mean = X[:, continuous_idx].mean(axis=0)
    scaler_std = X[:, continuous_idx].std(axis=0)
    X_scaled = X.copy()
    X_scaled[:, continuous_idx] = (X[:, continuous_idx] - scaler_mean) / scaler_std

    X0 = X_scaled[y == 0]
    temp = CompositeLikelihoodModel(mask=mask, lambda_reg=0.01, max_iter=100)
    temp.continuous_idx_ = continuous_idx
    temp.ordinal_idx_ = ordinal_idx
    thresholds, categories = temp._estimate_thresholds_and_cats(X_scaled)

    model = CompositeLikelihoodModel(mask=mask, lambda_reg=0.01, max_iter=100)
    model.fit(
        X0,
        continuous_idx,
        ordinal_idx,
        variable_names,
        shared_thresholds=thresholds,
        shared_categories=categories,
    )

    grads = model.per_observation_gradient(X0)
    H = model.objective_hessian(X0)
    J = stable_symmetrize((grads.T @ grads) / X0.shape[0])

    assert H.shape == J.shape
    assert not np.allclose(H, J, rtol=1e-3, atol=1e-3)

    geometry = GodambeGeometry(
        gradients=grads,
        H=H,
        ridge_gamma=1e-3,
        shrink_j=False,
    ).fit()

    assert geometry.H_ is not None
    assert np.allclose(geometry.H_, H, rtol=1e-10, atol=1e-10)
    assert geometry.delta_ == 0.0


def test_sandwich_whitening_is_psd(heart_data):
    X, y, variable_names, continuous_idx, ordinal_idx = heart_data
    mask = StructuralMask.from_domain_knowledge(
        variable_names=variable_names,
        exogenous=["age", "sex"],
    )

    X_scaled = X.copy()
    X_scaled[:, continuous_idx] = (
        X[:, continuous_idx] - X[:, continuous_idx].mean(axis=0)
    ) / X[:, continuous_idx].std(axis=0)

    X0 = X_scaled[y == 0]
    temp = CompositeLikelihoodModel(mask=mask, lambda_reg=0.01, max_iter=80)
    temp.continuous_idx_ = continuous_idx
    temp.ordinal_idx_ = ordinal_idx
    thresholds, categories = temp._estimate_thresholds_and_cats(X_scaled)

    model = CompositeLikelihoodModel(mask=mask, lambda_reg=0.01, max_iter=80)
    model.fit(
        X0,
        continuous_idx,
        ordinal_idx,
        variable_names,
        shared_thresholds=thresholds,
        shared_categories=categories,
    )

    grads = model.per_observation_gradient(X0)
    H = model.objective_hessian(X0)
    geometry = GodambeGeometry(gradients=grads, H=H, ridge_gamma=1e-3).fit()

    eigvals = np.linalg.eigvalsh(geometry.G_inv_)
    assert np.min(eigvals) > -1e-8


def test_linear_feature_type_alias_maps_to_godambe():
    from geometry_fisher.features import normalize_feature_type

    with pytest.warns(FutureWarning, match="deprecated"):
        assert normalize_feature_type("linear") == "godambe"
    assert normalize_feature_type("godambe") == "godambe"
