"""
Data loading utilities for the Geometry-Aware Fisher Kernel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from pathlib import Path


def load_heart_disease(
    path: str,
    binary_target: bool = True,
    only_cleveland: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Load the Heart Disease UCI dataset from the modern CSV format
    (the one you have: heart_disease_uci.csv).
    """
    df = pd.read_csv(path)

    # Keep only Cleveland if requested (recommended for comparability with the thesis)
    if only_cleveland and "dataset" in df.columns:
        df = df[df["dataset"] == "Cleveland"].copy()

    # Drop rows with missing values
    df = df.dropna().reset_index(drop=True)

    # Target
    if "num" in df.columns:
        y = df["num"].values
    else:
        y = df["target"].values

    if binary_target:
        y = (y > 0).astype(int)

    # Select and encode features
    # We follow a clean mixed-type scheme close to the thesis

    # Continuous
    continuous_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
    # Note: your file uses "thalch" instead of "thalach"

    # Ordinal / categorical that we will treat as ordinal
    # We encode them as integers
    df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
    df["fbs"] = df["fbs"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0})
    df["exang"] = df["exang"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0})

    # Chest pain
    cp_map = {
        "typical angina": 1,
        "atypical angina": 2,
        "non-anginal": 3,
        "asymptomatic": 4,
    }
    df["cp"] = df["cp"].map(cp_map)

    # Resting ECG
    restecg_map = {
        "normal": 0,
        "st-t abnormality": 1,
        "lv hypertrophy": 2,
    }
    df["restecg"] = df["restecg"].map(restecg_map)

    # Slope
    slope_map = {
        "upsloping": 1,
        "flat": 2,
        "downsloping": 3,
    }
    df["slope"] = df["slope"].map(slope_map)

    # Thal
    thal_map = {
        "normal": 3,
        "fixed defect": 6,
        "reversable defect": 7,
    }
    df["thal"] = df["thal"].map(thal_map)

    # ca is already numeric in most rows
    df["ca"] = pd.to_numeric(df["ca"], errors="coerce")

    # Final feature list (order matters)
    feature_cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal"
    ]

    # Drop any remaining rows with NaNs after encoding
    df = df.dropna(subset=feature_cols + ["num"] if "num" in df.columns else feature_cols).reset_index(drop=True)

    X = df[feature_cols].astype(float).values
    y = (df["num"].values > 0).astype(int) if binary_target else df["num"].values

    variable_names = feature_cols

    continuous_idx = np.array([variable_names.index(c) for c in continuous_cols])
    ordinal_idx = np.array([i for i, name in enumerate(variable_names) if name not in continuous_cols])

    print(f"Loaded Heart Disease data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Continuous indices: {continuous_idx}")
    print(f"Ordinal indices: {ordinal_idx}")

    return X, y, variable_names, continuous_idx, ordinal_idx