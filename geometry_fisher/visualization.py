"""
Visualization utilities for the Geometry-Aware Fisher Kernel.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple

from .structure import StructuralMask
from .composite import CompositeLikelihoodModel
from .pipeline import GeometryFisherClassifier


def plot_dependency_heatmap(
    W: np.ndarray,
    variable_names: List[str],
    title: str = "Dependency Matrix",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "RdBu_r",
    center: float = 0.0,
    annot: bool = False,
    ax=None,
):
    """
    Plot a single dependency weight matrix as a heatmap.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.heatmap(
        W,
        xticklabels=variable_names,
        yticklabels=variable_names,
        cmap=cmap,
        center=center,
        annot=annot,
        fmt=".2f",
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Source")
    ax.set_ylabel("Target")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    return fig, ax


def plot_class_dependencies(
    model_0: CompositeLikelihoodModel,
    model_1: CompositeLikelihoodModel,
    variable_names: List[str],
    figsize: Tuple[int, int] = (16, 6),
):
    """
    Plot the two class-specific dependency matrices side by side.
    """
    W0 = model_0._theta_to_W(model_0.theta_hat_) if hasattr(model_0, "_theta_to_W") else _reconstruct_W(model_0)
    W1 = model_1._theta_to_W(model_1.theta_hat_) if hasattr(model_1, "_theta_to_W") else _reconstruct_W(model_1)

    # Convert to numpy if needed
    W0 = np.array(W0)
    W1 = np.array(W1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_dependency_heatmap(W0, variable_names, title="Class 0 (No Disease)", ax=axes[0])
    plot_dependency_heatmap(W1, variable_names, title="Class 1 (Disease)", ax=axes[1])

    plt.tight_layout()
    return fig


def plot_difference_heatmap(
    model_0: CompositeLikelihoodModel,
    model_1: CompositeLikelihoodModel,
    variable_names: List[str],
    figsize: Tuple[int, int] = (9, 7),
):
    """
    Plot the difference matrix W1 - W0.
    """
    W0 = np.array(_reconstruct_W(model_0))
    W1 = np.array(_reconstruct_W(model_1))
    diff = W1 - W0

    fig, ax = plt.subplots(figsize=figsize)
    plot_dependency_heatmap(
        diff,
        variable_names,
        title="Difference (Class 1 – Class 0)",
        cmap="RdBu_r",
        center=0.0,
        ax=ax,
    )
    plt.tight_layout()
    return fig


def plot_mask(mask: StructuralMask, figsize: Tuple[int, int] = (8, 7)):
    """
    Visualize a binary structural mask.
    """
    names = mask.variable_names if mask.variable_names is not None else [f"X{i}" for i in range(mask.p)]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        mask.matrix,
        xticklabels=names,
        yticklabels=names,
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )
    ax.set_title(f"Structural Mask ({mask.n_params} free parameters)", fontsize=13)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    return fig


def _reconstruct_W(model: CompositeLikelihoodModel) -> np.ndarray:
    """Helper to reconstruct the full W matrix from theta_hat and the mask."""
    p = model.n_features_
    W = np.zeros((p, p))
    active = np.argwhere(model.mask.matrix == 1)
    for idx, (i, j) in enumerate(active):
        W[i, j] = model.theta_hat_[idx]
    return W