"""Visualization utilities for dependency matrices and structural masks."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from typing import List, Optional, Sequence, Tuple

from .structure import StructuralMask
from .composite import CompositeLikelihoodModel

NODE_RADIUS = 0.085


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
    W0 = _reconstruct_W(model_0)
    W1 = _reconstruct_W(model_1)

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
    diff = _reconstruct_W(model_1) - _reconstruct_W(model_0)

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


def plot_mask(
    mask: StructuralMask,
    figsize: Tuple[int, int] = (8, 7),
    title: str | None = None,
):
    names = mask.variable_names if mask.variable_names is not None else [f"X{i}" for i in range(mask.p)]
    if title is None:
        title = f"Structural Mask ({mask.n_params} free parameters)"

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
    ax.set_title(title, fontsize=13)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    return fig


def _circular_layout(variable_names: Sequence[str]) -> dict[str, tuple[float, float]]:
    names = list(variable_names)
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(names), endpoint=False)
    return {
        name: (float(np.cos(angle)), float(np.sin(angle)))
        for name, angle in zip(names, angles)
    }


def _edge_endpoints(
    source_pos: tuple[float, float],
    target_pos: tuple[float, float],
    radius: float = NODE_RADIUS,
) -> tuple[tuple[float, float], tuple[float, float]]:
    dx = target_pos[0] - source_pos[0]
    dy = target_pos[1] - source_pos[1]
    dist = float(np.hypot(dx, dy))
    if dist < 1e-9:
        return source_pos, target_pos
    ux, uy = dx / dist, dy / dist
    start = (source_pos[0] + ux * radius, source_pos[1] + uy * radius)
    end = (target_pos[0] - ux * radius, target_pos[1] - uy * radius)
    return start, end


def _draw_directed_edge(
    ax,
    source_pos: tuple[float, float],
    target_pos: tuple[float, float],
    *,
    color: str,
    linestyle: str = "solid",
    linewidth: float = 1.8,
    alpha: float = 1.0,
    zorder: int = 2,
) -> None:
    start, end = _edge_endpoints(source_pos, target_pos)
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        alpha=alpha,
        zorder=zorder,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)


def _draw_structure_graph_on_ax(
    ax,
    mask: StructuralMask,
    *,
    title: str,
    exogenous: Optional[Sequence[str]] = None,
    blocked_mask: Optional[np.ndarray] = None,
    show_legend: bool = True,
) -> None:
    if mask.variable_names is None:
        raise ValueError("variable_names must be set to plot a structural graph.")

    names = list(mask.variable_names)
    layout = _circular_layout(names)
    exogenous_set = set(exogenous or [])

    ax.set_aspect("equal")
    ax.axis("off")

    if blocked_mask is not None:
        blocked_mask = np.asarray(blocked_mask, dtype=int)
        for i, target in enumerate(names):
            for j, source in enumerate(names):
                if blocked_mask[i, j] == 1:
                    _draw_directed_edge(
                        ax,
                        layout[source],
                        layout[target],
                        color="#c0392b",
                        linestyle="dashed",
                        linewidth=1.4,
                        alpha=0.85,
                        zorder=1,
                    )

    for i, target in enumerate(names):
        for j, source in enumerate(names):
            if mask.matrix[i, j] == 1:
                _draw_directed_edge(
                    ax,
                    layout[source],
                    layout[target],
                    color="#1f4e79",
                    zorder=3,
                )

    for name, (x, y) in layout.items():
        face = "#f4b942" if name in exogenous_set else "#d9e8f5"
        edge = "#b9770e" if name in exogenous_set else "#5d6d7e"
        circle = mpatches.Circle(
            (x, y),
            radius=NODE_RADIUS,
            facecolor=face,
            edgecolor=edge,
            linewidth=1.5,
            zorder=4,
        )
        ax.add_patch(circle)
        ax.text(
            x,
            y,
            name,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="semibold" if name in exogenous_set else "normal",
            zorder=5,
        )

    if show_legend:
        legend_handles = [
            mpatches.Patch(color="#1f4e79", label="Allowed edge (source → target)"),
        ]
        if blocked_mask is not None and blocked_mask.sum() > 0:
            legend_handles.append(
                mpatches.Patch(
                    facecolor="white",
                    edgecolor="#c0392b",
                    linestyle="--",
                    linewidth=1.4,
                    label="Blocked edge (removed by domain rules)",
                )
            )
        if exogenous_set:
            legend_handles.append(
                mpatches.Patch(color="#f4b942", label="Exogenous variable")
            )
        ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.05))

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.45, 1.25)
    ax.set_title(title, fontsize=12, pad=10)


def plot_structure_graph(
    mask: StructuralMask,
    *,
    title: str = "Structural graph",
    figsize: Tuple[int, int] = (9, 9),
    exogenous: Optional[Sequence[str]] = None,
    blocked_mask: Optional[np.ndarray] = None,
) -> plt.Figure:
    """
    Draw the structural mask as a directed graph.

    Matrix entry ``mask[i, j] == 1`` is an allowed edge **source j -> target i**.
    Optional ``blocked_mask`` draws removed edges as dashed red arrows.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _draw_structure_graph_on_ax(
        ax,
        mask,
        title=title,
        exogenous=exogenous,
        blocked_mask=blocked_mask,
    )
    plt.tight_layout()
    return fig


def plot_structure_curation(
    discovered: StructuralMask,
    curated: StructuralMask,
    *,
    title: str,
    exogenous: Optional[Sequence[str]] = None,
    figsize: Tuple[int, int] = (16, 7),
) -> plt.Figure:
    """Compare a discovered structure with its manually curated version."""
    if discovered.variable_names != curated.variable_names:
        raise ValueError("Discovered and curated masks must share variable_names.")

    blocked = discovered.matrix.astype(int) & ~curated.matrix.astype(int)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    _draw_structure_graph_on_ax(
        axes[0],
        discovered,
        title="Discovered structure",
        exogenous=exogenous,
        show_legend=False,
    )
    _draw_structure_graph_on_ax(
        axes[1],
        curated,
        title="After domain curation",
        exogenous=exogenous,
        blocked_mask=blocked,
    )

    legend_handles = [
        mpatches.Patch(color="#1f4e79", label="Allowed edge (source → target)"),
        mpatches.Patch(
            facecolor="white",
            edgecolor="#c0392b",
            linestyle="--",
            linewidth=1.4,
            label="Blocked edge (removed by domain rules)",
        ),
        mpatches.Patch(color="#f4b942", label="Exogenous variable"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def _reconstruct_W(model: CompositeLikelihoodModel) -> np.ndarray:
    p = model.n_features_
    W = np.zeros((p, p))
    active = np.argwhere(model.mask.matrix == 1)
    for idx, (i, j) in enumerate(active):
        W[i, j] = model.theta_hat_[idx]
    return W
