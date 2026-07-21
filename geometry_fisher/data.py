"""
Data loading for the UCI Heart Disease benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, List


def load_heart_disease(
    path: str,
    binary_target: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Load the 9-variable Heart Disease dataset used in the experiments.

    Uses all four centers (531 complete cases after dropna).
    """
    df = pd.read_csv(path)

    continuous_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
    ordinal_cols = ["sex", "fbs", "exang", "slope"]
    selected_cols = continuous_cols + ordinal_cols

    df_work = df[selected_cols + ["num"]].copy()
    df_clean = df_work.dropna().reset_index(drop=True)

    df_clean["sex"] = (df_clean["sex"] == "Male").astype(int)
    df_clean["fbs"] = df_clean["fbs"].astype(int)
    df_clean["exang"] = df_clean["exang"].astype(int)
    df_clean["slope"] = df_clean["slope"].map(
        {"downsloping": 0, "flat": 1, "upsloping": 2}
    )

    if df_clean["slope"].isna().any():
        raise ValueError("Some 'slope' values could not be mapped.")

    X = df_clean[selected_cols].astype(float).values
    y = df_clean["num"].values
    if binary_target:
        y = (y > 0).astype(int)

    variable_names = selected_cols
    continuous_idx = np.array([0, 1, 2, 3, 4])
    ordinal_idx = np.array([5, 6, 7, 8])

    if verbose:
        print(
            f"Loaded Heart Disease data: {X.shape[0]} samples, {X.shape[1]} features"
        )
        print(f"Class distribution: {np.bincount(y)}")

    return X, y, variable_names, continuous_idx, ordinal_idx
