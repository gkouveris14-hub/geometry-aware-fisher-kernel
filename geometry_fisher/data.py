"""
Data loading utilities for the Geometry-Aware Fisher Kernel.
Matches the exact preprocessing used in the thesis notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, List


def load_heart_disease(
    path: str,
    binary_target: bool = True,
    only_cleveland: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Load and clean the Heart Disease data using the exact
    9 variables and encodings from the thesis notebook.

    By default uses all available centers (531 complete cases after dropna),
    matching the thesis experimental protocol.
    """
    df = pd.read_csv(path)

    if only_cleveland and "dataset" in df.columns:
        df = df[df["dataset"] == "Cleveland"].copy()

    continuous_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
    ordinal_cols = ["sex", "fbs", "exang", "slope"]
    selected_cols = continuous_cols + ordinal_cols

    df_work = df[selected_cols + ["num"]].copy()
    df_clean = df_work.dropna().reset_index(drop=True)

    # Encodings exactly as in Last_hope.ipynb
    df_clean["sex"] = (df_clean["sex"] == "Male").astype(int)
    df_clean["fbs"] = df_clean["fbs"].astype(int)
    df_clean["exang"] = df_clean["exang"].astype(int)

    slope_mapping = {
        "downsloping": 0,
        "flat": 1,
        "upsloping": 2,
    }
    df_clean["slope"] = df_clean["slope"].map(slope_mapping)

    if df_clean["slope"].isna().any():
        raise ValueError("Some 'slope' values could not be mapped.")

    X = df_clean[selected_cols].astype(float).values
    y = df_clean["num"].values

    if binary_target:
        y = (y > 0).astype(int)

    variable_names = selected_cols
    continuous_idx = np.array([0, 1, 2, 3, 4])
    ordinal_idx = np.array([5, 6, 7, 8])

    print(
        f"Loaded Heart Disease data (thesis selection): "
        f"{X.shape[0]} samples, {X.shape[1]} features"
    )
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Variables: {variable_names}")
    print(f"Continuous indices: {continuous_idx}")
    print(f"Ordinal indices: {ordinal_idx}")

    return X, y, variable_names, continuous_idx, ordinal_idx
