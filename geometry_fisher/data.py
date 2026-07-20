"""
Data loading utilities for the Geometry-Aware Fisher Kernel.
Matches the exact variable selection used in the thesis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, List


def load_heart_disease(
    path: str,
    binary_target: bool = True,
    only_cleveland: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Load and clean the Heart Disease data using the exact
    9 variables from the thesis.
    """
    df = pd.read_csv(path)

    # Keep only Cleveland if requested
    if only_cleveland and "dataset" in df.columns:
        df = df[df["dataset"] == "Cleveland"].copy()

    # Exact variable selection from the thesis notebook
    continuous_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
    ordinal_cols = ["sex", "fbs", "exang", "slope"]
    selected_cols = continuous_cols + ordinal_cols

    # Basic encoding for ordinal variables
    df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
    df["fbs"] = df["fbs"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0})
    df["exang"] = df["exang"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0})

    slope_map = {
        "upsloping": 1,
        "flat": 2,
        "downsloping": 3,
    }
    df["slope"] = df["slope"].map(slope_map)

    # Select columns + target
    df_work = df[selected_cols + ["num"]].copy()
    df_clean = df_work.dropna().reset_index(drop=True)

    X = df_clean[selected_cols].astype(float).values
    y = df_clean["num"].values

    if binary_target:
        y = (y > 0).astype(int)

    variable_names = selected_cols
    continuous_idx = np.array([0, 1, 2, 3, 4])          # first 5
    ordinal_idx = np.array([5, 6, 7, 8])                # last 4

    print(f"Loaded Heart Disease data (thesis selection): {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Variables: {variable_names}")
    print(f"Continuous indices: {continuous_idx}")
    print(f"Ordinal indices: {ordinal_idx}")

    return X, y, variable_names, continuous_idx, ordinal_idx