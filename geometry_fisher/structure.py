"""
Structural masks for the composite likelihood models.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Sequence
from dataclasses import dataclass


def _directed_adjacency_from_pc_graph(graph: np.ndarray) -> np.ndarray:
    """
    Convert a causal-learn CPDAG into a binary structural mask.

    Encoding follows the thesis notebook (``Last_hope-Copy1.ipynb``):

    - ``graph[i, j] == 1``  → directed edge i → j
    - ``graph[i, j] == -1`` → directed edge j → i
    - ``graph[i, j] == 2``  → undirected adjacency (both directions kept)
    """
    p = graph.shape[0]
    matrix = np.zeros((p, p), dtype=int)

    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            val = graph[i, j]
            if val == 1:
                matrix[j, i] = 1
            elif val == -1:
                matrix[i, j] = 1
            elif val == 2:
                matrix[i, j] = 1
                matrix[j, i] = 1

    np.fill_diagonal(matrix, 0)
    return matrix


# Final curated PC mask from the thesis (pc_mask_visualization), 16 edges.
THESIS_PC_ALLOWED_EDGES: tuple[tuple[str, str], ...] = (
    ("thalch", "age"),
    ("thalch", "chol"),
    ("thalch", "exang"),
    ("thalch", "slope"),
    ("oldpeak", "trestbps"),
    ("oldpeak", "chol"),
    ("oldpeak", "exang"),
    ("oldpeak", "slope"),
    ("fbs", "age"),
    ("fbs", "trestbps"),
    ("fbs", "chol"),
    ("exang", "trestbps"),
    ("exang", "sex"),
    ("slope", "thalch"),
    ("slope", "oldpeak"),
    ("slope", "exang"),
)

THESIS_PC_REJECTED_EDGES: tuple[tuple[str, str], ...] = (
    ("age", "trestbps"),
    ("age", "thalch"),
    ("age", "fbs"),
    ("sex", "chol"),
    ("sex", "exang"),
)


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

    def block_edges(self, forbidden_edges: Sequence[tuple]) -> "StructuralMask":
        """
        Remove directed edges from the mask.

        Each entry is ``(target, source)``: source -> target is blocked.
        Use this after PC or stability selection to enforce domain constraints
        on a discovered structure.
        """
        if self.variable_names is None:
            raise ValueError("variable_names must be set to block edges by name.")
        if not forbidden_edges:
            return self

        new_matrix = self.matrix.copy()
        name_to_idx = {name: i for i, name in enumerate(self.variable_names)}
        for target, source in forbidden_edges:
            if target not in name_to_idx:
                raise ValueError(f"Variable '{target}' not found in variable_names.")
            if source not in name_to_idx:
                raise ValueError(f"Variable '{source}' not found in variable_names.")
            i, j = name_to_idx[target], name_to_idx[source]
            new_matrix[i, j] = 0
        return StructuralMask(new_matrix, self.variable_names)

    @staticmethod
    def _apply_domain_constraints(
        mask: "StructuralMask",
        exogenous: Optional[Sequence[str]] = None,
        forbidden_edges: Optional[Sequence[tuple]] = None,
    ) -> "StructuralMask":
        if exogenous is not None:
            mask = mask.enforce_exogeneity(exogenous)
        if forbidden_edges:
            mask = mask.block_edges(forbidden_edges)
        return mask

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

        return cls._apply_domain_constraints(
            mask,
            exogenous=exogenous,
            forbidden_edges=None,
        )

    @classmethod
    def from_array(
        cls,
        matrix: np.ndarray,
        variable_names: Optional[List[str]] = None,
    ) -> "StructuralMask":
        return cls(matrix=matrix, variable_names=variable_names)

    @classmethod
    def from_thesis_pc_reference(cls, variable_names: List[str]) -> "StructuralMask":
        """Return the fixed 16-edge PC mask published in the thesis."""
        return cls.from_domain_knowledge(
            variable_names=variable_names,
            allowed_edges=list(THESIS_PC_ALLOWED_EDGES),
        )

    @classmethod
    def from_pc_algorithm(
        cls,
        X: np.ndarray,
        variable_names: List[str],
        alpha: float = 0.05,
        exogenous: Optional[Sequence[str]] = None,
        forbidden_edges: Optional[Sequence[tuple]] = None,
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

        return cls._apply_domain_constraints(
            mask,
            exogenous=exogenous,
            forbidden_edges=forbidden_edges,
        )

    @classmethod
    def from_stability_selection(
        cls,
        X: np.ndarray,
        variable_names: List[str],
        alpha: float = 0.05,
        tau_stab: float = 0.6,
        B: int = 50,
        exogenous: Optional[Sequence[str]] = None,
        forbidden_edges: Optional[Sequence[tuple]] = None,
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

        return cls._apply_domain_constraints(
            mask,
            exogenous=exogenous,
            forbidden_edges=forbidden_edges,
        )

    def __repr__(self) -> str:
        return f"StructuralMask(p={self.p}, n_params={self.n_params})"


def prepare_matrix_for_pc(X: np.ndarray) -> np.ndarray:
    """Z-score every column before PC, matching the thesis notebook."""
    X_arr = np.asarray(X, dtype=float)
    mean = X_arr.mean(axis=0)
    std = X_arr.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X_arr - mean) / std


def scale_continuous_features(
    X: np.ndarray,
    continuous_idx: np.ndarray,
) -> np.ndarray:
    """Standardize continuous columns for composite-model fitting."""
    from sklearn.preprocessing import StandardScaler

    X_scaled = np.asarray(X, dtype=float).copy()
    X_scaled[:, continuous_idx] = StandardScaler().fit_transform(X[:, continuous_idx])
    return X_scaled


def discover_data_driven_mask(
    X: np.ndarray,
    variable_names: List[str],
    continuous_idx: np.ndarray,
    mask: str,
    mask_params: Optional[dict] = None,
) -> StructuralMask:
    """
    Discover a data-driven mask on the provided data matrix.

    Experiment 2 in the thesis runs PC once on the full pooled sample and
    reuses the curated mask in every CV fold.
    """
    params = mask_params or {}
    X_pc = prepare_matrix_for_pc(X)

    if mask == "pc":
        return StructuralMask.from_pc_algorithm(
            X_pc,
            list(variable_names),
            alpha=params.get("alpha", 0.05),
            exogenous=params.get("exogenous"),
            forbidden_edges=params.get("forbidden_edges"),
        )
    if mask in ("stability", "data_driven"):
        return StructuralMask.from_stability_selection(
            X_pc,
            list(variable_names),
            alpha=params.get("alpha", 0.05),
            tau_stab=params.get("tau_stab", 0.6),
            B=params.get("B", 50),
            exogenous=params.get("exogenous"),
            forbidden_edges=params.get("forbidden_edges"),
            random_state=params.get("random_state", 42),
        )
    raise ValueError(
        f"Cannot discover mask type {mask!r}. Use 'pc' or 'stability'."
    )
