"""
Structural mask handling for the Geometry-Aware Fisher Kernel.

Supports:
- Hand-specified (domain knowledge) masks
- Data-driven masks via stability selection + PC algorithm
- Custom user-provided masks
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Sequence, Union
from dataclasses import dataclass


@dataclass
class StructuralMask:
    """
    Binary mask that defines which directed dependencies are allowed.

    Attributes
    ----------
    matrix : np.ndarray
        Binary matrix of shape (p, p). matrix[i, j] = 1 means
        the edge j → i is allowed.
    variable_names : list of str, optional
        Names of the variables (for readability and exogeneity checks).
    """

    matrix: np.ndarray
    variable_names: Optional[List[str]] = None

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=int)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Mask must be a square matrix.")
        if not np.all(np.isin(self.matrix, [0, 1])):
            raise ValueError("Mask must contain only 0s and 1s.")
        # No self-loops
        np.fill_diagonal(self.matrix, 0)

    @property
    def n_params(self) -> int:
        """Number of free parameters (active edges)."""
        return int(self.matrix.sum())

    @property
    def p(self) -> int:
        """Number of variables."""
        return self.matrix.shape[0]

    def apply(self, W: np.ndarray) -> np.ndarray:
        """Apply the mask to a parameter matrix (element-wise)."""
        return W * self.matrix

    def enforce_exogeneity(self, exogenous: Sequence[str]) -> "StructuralMask":
        """
        Return a new mask where the given variables have no incoming edges.
        """
        if self.variable_names is None:
            raise ValueError("variable_names must be set to use enforce_exogeneity.")

        new_matrix = self.matrix.copy()
        for var in exogenous:
            if var not in self.variable_names:
                raise ValueError(f"Variable '{var}' not found in variable_names.")
            idx = self.variable_names.index(var)
            new_matrix[idx, :] = 0  # no incoming edges

        return StructuralMask(new_matrix, self.variable_names)

    @classmethod
    def from_domain_knowledge(
        cls,
        variable_names: List[str],
        exogenous: Optional[Sequence[str]] = None,
        allowed_edges: Optional[List[tuple]] = None,
        forbidden_edges: Optional[List[tuple]] = None,
    ) -> "StructuralMask":
        """
        Create a hand-specified mask from domain knowledge.

        Parameters
        ----------
        variable_names : list of str
            Ordered list of variable names.
        exogenous : list of str, optional
            Variables that should have no incoming edges (e.g. ["age", "sex"]).
        allowed_edges : list of (target, source) tuples, optional
            If provided, only these edges are allowed (plus any not forbidden).
        forbidden_edges : list of (target, source) tuples, optional
            Edges that must be zero.
        """
        p = len(variable_names)
        matrix = np.ones((p, p), dtype=int)
        np.fill_diagonal(matrix, 0)

        name_to_idx = {name: i for i, name in enumerate(variable_names)}

        if allowed_edges is not None:
            matrix[:] = 0
            for target, source in allowed_edges:
                i, j = name_to_idx[target], name_to_idx[source]
                matrix[i, j] = 1

        if forbidden_edges is not None:
            for target, source in forbidden_edges:
                i, j = name_to_idx[target], name_to_idx[source]
                matrix[i, j] = 0

        mask = cls(matrix, variable_names)

        if exogenous is not None:
            mask = mask.enforce_exogeneity(exogenous)

        return mask

    @classmethod
    def from_array(
        cls,
        matrix: np.ndarray,
        variable_names: Optional[List[str]] = None,
    ) -> "StructuralMask":
        """Create a mask directly from a binary numpy array."""
        return cls(matrix=matrix, variable_names=variable_names)

    def __repr__(self) -> str:
        return f"StructuralMask(p={self.p}, n_params={self.n_params})"