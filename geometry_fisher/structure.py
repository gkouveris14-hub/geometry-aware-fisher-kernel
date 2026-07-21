"""
Structural masks for the composite likelihood models.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Sequence
from dataclasses import dataclass


def _directed_adjacency_from_pc_graph(graph: np.ndarray) -> np.ndarray:
    """
    Convert a PC CPDAG adjacency matrix into a binary structural mask.

    Directed edges are oriented when the PC output identifies an arrow;
    undirected adjacencies are kept as both directions, matching the thesis.
    """
    p = graph.shape[0]
    matrix = np.zeros((p, p), dtype=int)

    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if graph[i, j] == 0 and graph[j, i] == 0:
                continue
            if graph[j, i] == 1 and graph[i, j] == -1:
                matrix[i, j] = 1
            elif graph[i, j] == 1 and graph[j, i] == -1:
                matrix[j, i] = 1
            else:
                matrix[i, j] = 1
                matrix[j, i] = 1

    np.fill_diagonal(matrix, 0)
    return matrix


@dataclass
class StructuralMask:
    """
    Binary mask that defines which directed dependencies are allowed.
    """

    matrix: np.ndarray
    variable_names: Optional[List[str]] = None

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=int)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Mask must be a square matrix.")
        if not np.all(np.isin(self.matrix, [0, 1])):
            raise ValueError("Mask must contain only 0s and 1s.")
        np.fill_diagonal(self.matrix, 0)

    @property
    def n_params(self) -> int:
        return int(self.matrix.sum())

    @property
    def p(self) -> int:
        return self.matrix.shape[0]

    def apply(self, W: np.ndarray) -> np.ndarray:
        return W * self.matrix

    def enforce_exogeneity(self, exogenous: Sequence[str]) -> "StructuralMask":
        if self.variable_names is None:
            raise ValueError("variable_names must be set to use enforce_exogeneity.")
        new_matrix = self.matrix.copy()
        for var in exogenous:
            if var not in self.variable_names:
                raise ValueError(f"Variable '{var}' not found in variable_names.")
            idx = self.variable_names.index(var)
            new_matrix[idx, :] = 0
        return StructuralMask(new_matrix, self.variable_names)

    @classmethod
    def from_domain_knowledge(
        cls,
        variable_names: List[str],
        exogenous: Optional[Sequence[str]] = None,
        allowed_edges: Optional[List[tuple]] = None,
        forbidden_edges: Optional[List[tuple]] = None,
    ) -> "StructuralMask":
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
        return cls(matrix=matrix, variable_names=variable_names)

    @classmethod
    def from_pc_algorithm(
        cls,
        X: np.ndarray,
        variable_names: List[str],
        alpha: float = 0.05,
        exogenous: Optional[Sequence[str]] = None,
    ) -> "StructuralMask":
        """Build a mask from one PC run on the training data (Experiment 2)."""
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz

        cg = pc(
            np.asarray(X, dtype=float),
            alpha=alpha,
            indep_test=fisherz,
            verbose=False,
            show_progress=False,
        )
        matrix = _directed_adjacency_from_pc_graph(cg.G.graph)

        mask = cls(matrix=matrix, variable_names=variable_names)

        if exogenous is not None:
            mask = mask.enforce_exogeneity(exogenous)

        return mask

    @classmethod
    def from_stability_selection(
        cls,
        X: np.ndarray,
        variable_names: List[str],
        alpha: float = 0.05,
        tau_stab: float = 0.6,
        B: int = 50,
        exogenous: Optional[Sequence[str]] = None,
        random_state: int = 42,
    ) -> "StructuralMask":
        """Build a mask from PC stability selection (Experiment 2)."""
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz

        n_samples, p = X.shape
        edge_freq = np.zeros((p, p))
        rng = np.random.RandomState(random_state)

        for _ in range(B):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_b = X[indices]

            cg = pc(
                X_b,
                alpha=alpha,
                indep_test=fisherz,
                verbose=False,
                show_progress=False,
            )
            edge_freq += _directed_adjacency_from_pc_graph(cg.G.graph)

        edge_freq /= B
        matrix = (edge_freq >= tau_stab).astype(int)
        np.fill_diagonal(matrix, 0)

        mask = cls(matrix=matrix, variable_names=variable_names)

        if exogenous is not None:
            mask = mask.enforce_exogeneity(exogenous)

        return mask

    def __repr__(self) -> str:
        return f"StructuralMask(p={self.p}, n_params={self.n_params})"
