"""
High-level pipeline for the Geometry-Aware Generalized Fisher Kernel.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Union, Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .structure import StructuralMask
from .composite import CompositeLikelihoodModel
from .geometry import GodambeGeometry
from .features import build_feature_matrix, normalize_feature_type, requires_geometry


class GeometryFisherClassifier(BaseEstimator, ClassifierMixin):
    """Geometry-Aware Generalized Fisher Kernel classifier."""

    def __init__(
        self,
        mask: Union[str, StructuralMask] = "hand",
        mask_object: Optional[StructuralMask] = None,
        mask_params: Optional[Dict[str, Any]] = None,
        lambda_reg: float = 0.01,
        ridge_gamma: float = 1e-3,
        shrink_j: bool = False,
        feature_type: str = "godambe",
        C: float = 1.0,
        max_iter: int = 1000,
        scale_phi: bool = True,
        verbose: bool = False,
    ):
        self.mask = mask
        self.mask_object = mask_object
        self.mask_params = mask_params or {}
        self.lambda_reg = lambda_reg
        self.ridge_gamma = ridge_gamma
        self.shrink_j = shrink_j
        self.feature_type = normalize_feature_type(feature_type)
        self.C = C
        self.max_iter = max_iter
        self.scale_phi = scale_phi
        self.verbose = verbose

        self.model_0_: Optional[CompositeLikelihoodModel] = None
        self.model_1_: Optional[CompositeLikelihoodModel] = None
        self.geometry_0_: Optional[GodambeGeometry] = None
        self.geometry_1_: Optional[GodambeGeometry] = None
        self.clf_ = None
        self.continuous_idx_: Optional[np.ndarray] = None
        self.ordinal_idx_: Optional[np.ndarray] = None
        self.mask_: Optional[StructuralMask] = None
        self.scaler_: Optional[StandardScaler] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        continuous_idx: np.ndarray,
        ordinal_idx: np.ndarray,
        variable_names: Optional[List[str]] = None,
    ) -> "GeometryFisherClassifier":
        """
        Fit the classifier.

        Ordinal thresholds are estimated from all training rows (both classes)
        and passed identically to the class-0 and class-1 composite models.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)

        self.continuous_idx_ = np.asarray(continuous_idx)
        self.ordinal_idx_ = np.asarray(ordinal_idx)

        self.scaler_ = StandardScaler()
        X = X.copy()
        X[:, self.continuous_idx_] = self.scaler_.fit_transform(X[:, self.continuous_idx_])

        mask = self._resolve_mask(X, variable_names)
        self.mask_ = mask

        if self.verbose:
            print(f"Using mask with {mask.n_params} free parameters.")

        if mask.n_params == 0:
            raise ValueError(
                "The selected mask has 0 free parameters. "
                "Try lowering tau_stab, increasing B, or using a hand-specified mask."
            )

        X0 = X[y == 0]
        X1 = X[y == 1]

        temp_model = CompositeLikelihoodModel(mask=mask, lambda_reg=self.lambda_reg)
        temp_model.continuous_idx_ = np.asarray(continuous_idx)
        temp_model.ordinal_idx_ = np.asarray(ordinal_idx)
        shared_thresholds, shared_categories = temp_model._estimate_thresholds_and_cats(X)

        self.model_0_ = CompositeLikelihoodModel(
            mask=mask,
            lambda_reg=self.lambda_reg,
            max_iter=600,
        )
        self.model_1_ = CompositeLikelihoodModel(
            mask=mask,
            lambda_reg=self.lambda_reg,
            max_iter=800,
        )

        self.model_0_.fit(
            X0,
            continuous_idx,
            ordinal_idx,
            variable_names,
            shared_thresholds=shared_thresholds,
            shared_categories=shared_categories,
        )
        self.model_1_.fit(
            X1,
            continuous_idx,
            ordinal_idx,
            variable_names,
            shared_thresholds=shared_thresholds,
            shared_categories=shared_categories,
        )

        if requires_geometry(self.feature_type):
            grads_0_class = self.model_0_.per_observation_gradient(X0)
            grads_1_class = self.model_1_.per_observation_gradient(X1)
            H0 = self.model_0_.objective_hessian(X0)
            H1 = self.model_1_.objective_hessian(X1)

            self.geometry_0_ = GodambeGeometry(
                gradients=grads_0_class,
                H=H0,
                ridge_gamma=self.ridge_gamma,
                shrink_j=self.shrink_j,
            ).fit()
            self.geometry_1_ = GodambeGeometry(
                gradients=grads_1_class,
                H=H1,
                ridge_gamma=self.ridge_gamma,
                shrink_j=self.shrink_j,
            ).fit()
        else:
            self.geometry_0_ = None
            self.geometry_1_ = None

        Phi = self._build_phi(X)

        lr = LogisticRegression(
            C=self.C,
            max_iter=max(self.max_iter, 2000),
            solver="lbfgs",
            random_state=42,
        )
        self.clf_ = make_pipeline(StandardScaler(), lr) if self.scale_phi else lr
        self.clf_.fit(Phi, y)
        return self

    def _build_phi(self, X: np.ndarray) -> np.ndarray:
        grads_0 = self.model_0_.per_observation_gradient(X)
        grads_1 = self.model_1_.per_observation_gradient(X)
        return build_feature_matrix(
            self.feature_type,
            grads_0,
            grads_1,
            self.geometry_0_,
            self.geometry_1_,
        )

    def _resolve_mask(
        self,
        X: np.ndarray,
        variable_names: Optional[List[str]],
    ) -> StructuralMask:
        if isinstance(self.mask, StructuralMask):
            return self.mask
        if self.mask == "hand":
            if self.mask_object is None:
                raise ValueError("When mask='hand', you must provide mask_object.")
            return self.mask_object
        if self.mask in ("data_driven", "stability", "pc"):
            return StructuralMask.from_stability_selection(
                X=X,
                variable_names=variable_names
                if variable_names is not None
                else [f"X{i}" for i in range(X.shape[1])],
                alpha=self.mask_params.get("alpha", 0.05),
                tau_stab=self.mask_params.get("tau_stab", 0.6),
                B=self.mask_params.get("B", 50),
                exogenous=self.mask_params.get("exogenous", None),
                random_state=self.mask_params.get("random_state", 42),
            )
        raise ValueError(f"Unknown mask type: {self.mask}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float).copy()
        X[:, self.continuous_idx_] = self.scaler_.transform(X[:, self.continuous_idx_])
        return self._build_phi(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf_.predict(self.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf_.predict_proba(self.transform(X))

    def __repr__(self) -> str:
        return f"GeometryFisherClassifier(feature_type='{self.feature_type}')"
