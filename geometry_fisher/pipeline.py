"""
High-level pipeline for the Geometry-Aware Generalized Fisher Kernel.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Union, Dict, Any
from dataclasses import dataclass, field

from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

from .structure import StructuralMask
from .composite import CompositeLikelihoodModel
from .geometry import GodambeGeometry
from .features import build_features, FisherFeatures


class GeometryFisherClassifier(BaseEstimator, ClassifierMixin):
    """
    Geometry-Aware Generalized Fisher Kernel classifier.
    """

    def __init__(
        self,
        mask: Union[str, StructuralMask] = "hand",
        mask_object: Optional[StructuralMask] = None,
        mask_params: Optional[Dict[str, Any]] = None,
        lambda_reg: float = 0.01,
        ridge_gamma: float = 1e-4,
        feature_type: str = "linear",
        C: float = 1.0,
        max_iter: int = 1000,
    ):
        self.mask = mask
        self.mask_object = mask_object
        self.mask_params = mask_params or {}
        self.lambda_reg = lambda_reg
        self.ridge_gamma = ridge_gamma
        self.feature_type = feature_type
        self.C = C
        self.max_iter = max_iter

        # Learned attributes
        self.model_0_: Optional[CompositeLikelihoodModel] = None
        self.model_1_: Optional[CompositeLikelihoodModel] = None
        self.geometry_0_: Optional[GodambeGeometry] = None
        self.geometry_1_: Optional[GodambeGeometry] = None
        self.clf_: Optional[LogisticRegression] = None
        self.continuous_idx_: Optional[np.ndarray] = None
        self.ordinal_idx_: Optional[np.ndarray] = None
        self.mask_: Optional[StructuralMask] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        continuous_idx: np.ndarray,
        ordinal_idx: np.ndarray,
        variable_names: Optional[List[str]] = None,
    ) -> "GeometryFisherClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)

        self.continuous_idx_ = np.asarray(continuous_idx)
        self.ordinal_idx_ = np.asarray(ordinal_idx)

        # Resolve mask
        if isinstance(self.mask, StructuralMask):
            mask = self.mask
        elif self.mask == "hand":
            if self.mask_object is None:
                raise ValueError("When mask='hand', you must provide mask_object.")
            mask = self.mask_object
        elif self.mask in ("data_driven", "stability", "pc"):
            mask = StructuralMask.from_stability_selection(
                X=X,
                variable_names=variable_names if variable_names is not None else [f"X{i}" for i in range(X.shape[1])],
                alpha=self.mask_params.get("alpha", 0.05),
                tau_stab=self.mask_params.get("tau_stab", 0.6),
                B=self.mask_params.get("B", 50),
                exogenous=self.mask_params.get("exogenous", None),
                random_state=self.mask_params.get("random_state", 42),
            )
        else:
            raise ValueError(f"Unknown mask type: {self.mask}")

        self.mask_ = mask
        print(f"Using mask with {mask.n_params} free parameters.")

        if mask.n_params == 0:
            raise ValueError(
                "The selected mask has 0 free parameters. "
                "Try lowering tau_stab, increasing B, or using a hand-specified mask."
            )

        # Split by class
        X0 = X[y == 0]
        X1 = X[y == 1]

        # Fit class-conditional composite models
        self.model_0_ = CompositeLikelihoodModel(mask=mask, lambda_reg=self.lambda_reg)
        self.model_1_ = CompositeLikelihoodModel(mask=mask, lambda_reg=self.lambda_reg)

        self.model_0_.fit(X0, continuous_idx, ordinal_idx, variable_names)
        self.model_1_.fit(X1, continuous_idx, ordinal_idx, variable_names)

        # Compute gradients on the full training set
        grads_0 = self.model_0_.per_observation_gradient(X)
        grads_1 = self.model_1_.per_observation_gradient(X)

        # Fit Godambe geometries
        self.geometry_0_ = GodambeGeometry(gradients=grads_0[y == 0], ridge_gamma=self.ridge_gamma).fit()
        self.geometry_1_ = GodambeGeometry(gradients=grads_1[y == 1], ridge_gamma=self.ridge_gamma).fit()

        # Build features
        features = build_features(grads_0, grads_1, self.geometry_0_, self.geometry_1_)
        Phi = self._select_features(features)

        # Downstream classifier (with feature standardization)
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        self.clf_ = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=self.C,
                max_iter=max(self.max_iter, 2000),
                solver="lbfgs",
                random_state=42,
            )
        )
        self.clf_.fit(Phi, y)

        return self
    
    def _select_features(self, features) -> np.ndarray:
        if self.feature_type == "raw":
            return features.raw
        elif self.feature_type == "fisher_only":
            return features.fisher_only
        elif self.feature_type == "linear":
            return features.linear
        elif self.feature_type == "quadratic":
            return features.quadratic
        elif self.feature_type == "full":
            return features.full
        else:
            raise ValueError(
                f"Unknown feature_type: '{self.feature_type}'. "
                f"Choose from: 'raw', 'fisher_only', 'linear', 'quadratic', 'full'"
            )
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        grads_0 = self.model_0_.per_observation_gradient(X)
        grads_1 = self.model_1_.per_observation_gradient(X)
        features = build_features(grads_0, grads_1, self.geometry_0_, self.geometry_1_)
        return self._select_features(features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Phi = self.transform(X)
        return self.clf_.predict(Phi)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Phi = self.transform(X)
        return self.clf_.predict_proba(Phi)

    def __repr__(self) -> str:
        return f"GeometryFisherClassifier(feature_type='{self.feature_type}')"