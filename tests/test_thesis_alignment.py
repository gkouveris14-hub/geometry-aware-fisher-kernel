"""Checks that the code follows the thesis preprocessing and geometry definitions."""

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


def test_class_models_share_identical_ordinal_thresholds(heart_data):
    from geometry_fisher.pipeline import GeometryFisherClassifier

    X, y, variable_names, continuous_idx, ordinal_idx = heart_data
    mask = StructuralMask.from_domain_knowledge(
        variable_names=variable_names,
        exogenous=["age", "sex"],
    )

    clf = GeometryFisherClassifier(
        mask="hand",
        mask_object=mask,
        feature_type="godambe",
        verbose=False,
    )
    clf.fit(X, y, continuous_idx, ordinal_idx, variable_names)

    assert clf.model_0_.thresholds_.keys() == clf.model_1_.thresholds_.keys()
    for key in clf.model_0_.thresholds_:
        assert np.allclose(
            clf.model_0_.thresholds_[key],
            clf.model_1_.thresholds_[key],
        )


def _scale_continuous(X, continuous_idx):
    X_scaled = X.copy()
    X_scaled[:, continuous_idx] = (
        X[:, continuous_idx] - X[:, continuous_idx].mean(axis=0)
    ) / X[:, continuous_idx].std(axis=0)
    return X_scaled


def test_pc_and_stability_masks_build_from_data(heart_data):
    X, _, variable_names, continuous_idx, _ = heart_data
    X_scaled = _scale_continuous(X, continuous_idx)

    pc_mask = StructuralMask.from_pc_algorithm(
        X_scaled,
        variable_names,
        alpha=0.05,
        exogenous=["age", "sex"],
    )
    assert pc_mask.n_params > 0

    stability_mask = StructuralMask.from_stability_selection(
        X_scaled,
        variable_names,
        alpha=0.05,
        tau_stab=0.6,
        B=5,
        exogenous=["age", "sex"],
        random_state=42,
    )
    assert stability_mask.n_params > 0


def test_block_edges_curates_discovered_pc_mask(heart_data):
    X, _, variable_names, continuous_idx, _ = heart_data
    X_scaled = _scale_continuous(X, continuous_idx)

    mask = StructuralMask.from_pc_algorithm(
        X_scaled,
        variable_names,
        alpha=0.05,
        exogenous=["age", "sex"],
    )
    active = np.argwhere(mask.matrix == 1)
    assert active.size > 0

    target, source = mask.variable_names[active[0][0]], mask.variable_names[active[0][1]]
    curated = mask.block_edges([(target, source)])

    assert curated.n_params == mask.n_params - 1
    assert curated.matrix[active[0][0], active[0][1]] == 0


def test_forbidden_edges_in_mask_params(heart_data):
    from geometry_fisher.pipeline import GeometryFisherClassifier

    X, y, variable_names, continuous_idx, ordinal_idx = heart_data
    X_scaled = _scale_continuous(X, continuous_idx)

    base_mask = StructuralMask.from_pc_algorithm(
        X_scaled,
        variable_names,
        alpha=0.05,
        exogenous=["age", "sex"],
    )
    active = np.argwhere(base_mask.matrix == 1)
    target, source = (
        variable_names[active[0][0]],
        variable_names[active[0][1]],
    )

    clf = GeometryFisherClassifier(
        mask="pc",
        mask_params={
            "alpha": 0.05,
            "exogenous": ["age", "sex"],
            "forbidden_edges": [(target, source)],
        },
        feature_type="raw",
        verbose=False,
    )
    clf.fit(X, y, continuous_idx, ordinal_idx, variable_names)

    assert clf.mask_.n_params == base_mask.n_params - 1


def test_fixed_full_data_mask_is_constant_across_folds(heart_data):
    from geometry_fisher.cross_validation import CrossValidationExperiment

    X, y, variable_names, continuous_idx, ordinal_idx = heart_data

    experiment = CrossValidationExperiment(
        mask="pc",
        mask_params={"alpha": 0.05, "exogenous": ["age", "sex"]},
        discover_mask_on="full_data",
        feature_type="raw",
        outer_splits=3,
        random_state=42,
        verbose=False,
    )
    result = experiment.run(
        X,
        y,
        continuous_idx=continuous_idx,
        ordinal_idx=ordinal_idx,
        variable_names=variable_names,
    )

    assert result.fixed_mask is not None
    assert result.fixed_mask.n_params > 0
    assert len({r.n_params for r in result.fold_results}) == 1
    assert result.fold_results[0].n_params == result.fixed_mask.n_params


def test_structure_graph_plots_for_hand_mask(heart_data):
    import matplotlib

    matplotlib.use("Agg")
    from geometry_fisher.visualization import plot_structure_graph

    _, _, variable_names, _, _ = heart_data
    mask = StructuralMask.from_domain_knowledge(
        variable_names=variable_names,
        exogenous=["age", "sex"],
    )
    fig = plot_structure_graph(mask, exogenous=["age", "sex"])
    assert fig is not None


def test_only_raw_and_godambe_feature_types_are_supported():
    from geometry_fisher.features import FEATURE_TYPES, normalize_feature_type

    assert FEATURE_TYPES == ("raw", "godambe")
    assert normalize_feature_type("raw") == "raw"
    assert normalize_feature_type("godambe") == "godambe"
    with pytest.raises(ValueError, match="Unknown feature_type"):
        normalize_feature_type("linear")
