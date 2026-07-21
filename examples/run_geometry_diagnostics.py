"""
Diagnostic study:
1. Raw vs Godambe with/without StandardScaler on Phi
2. ridge_gamma sweep for Godambe (linear)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from geometry_fisher.data import load_heart_disease
from geometry_fisher.nested_cv import NestedCVExperiment
from geometry_fisher.structure import StructuralMask

from config import DATA_PATH, OUTPUT_DIR

RIDGE_GAMMAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def _run_cv(
    *,
    mask,
    X,
    y,
    continuous_idx,
    ordinal_idx,
    variable_names,
    feature_type: str,
    scale_phi: bool,
    ridge_gamma: float = 1e-3,
) -> dict:
    experiment = NestedCVExperiment(
        mask="hand",
        mask_object=mask,
        feature_type=feature_type,
        lambda_reg=0.01,
        ridge_gamma=ridge_gamma,
        shrink_j=False,
        scale_phi=scale_phi,
        outer_splits=5,
        random_state=42,
        verbose=False,
    )
    result = experiment.run(
        X,
        y,
        continuous_idx=continuous_idx,
        ordinal_idx=ordinal_idx,
        variable_names=variable_names,
    )
    return {
        "feature_type": feature_type,
        "scale_phi": scale_phi,
        "ridge_gamma": ridge_gamma,
        "accuracy_mean": result.mean_accuracy,
        "accuracy_std": result.std_accuracy,
        "macro_f1_mean": result.mean_macro_f1,
        "macro_f1_std": result.std_macro_f1,
        "auc_mean": result.mean_auc,
        "auc_std": result.std_auc,
    }


def main() -> None:
    X, y, variable_names, continuous_idx, ordinal_idx = load_heart_disease(
        path=str(DATA_PATH),
        binary_target=True,
        only_cleveland=False,
    )
    mask = StructuralMask.from_domain_knowledge(
        variable_names=variable_names,
        exogenous=["age", "sex"],
    )

    rows: list[dict] = []

    print("=" * 78)
    print("DIAGNOSTIC 1: Effect of StandardScaler on Phi")
    print("=" * 78)

    for feature_type, label in [("raw", "Raw gradients"), ("linear", "Godambe")]:
        for scale_phi in [True, False]:
            scaler_label = "with Phi scaler" if scale_phi else "no Phi scaler"
            print(f"Running {label} ({scaler_label})...")
            row = _run_cv(
                mask=mask,
                X=X,
                y=y,
                continuous_idx=continuous_idx,
                ordinal_idx=ordinal_idx,
                variable_names=variable_names,
                feature_type=feature_type,
                scale_phi=scale_phi,
            )
            row["diagnostic"] = "scaler_ablation"
            row["label"] = label
            rows.append(row)

    scaler_df = pd.DataFrame([r for r in rows if r["diagnostic"] == "scaler_ablation"])
    print("\nScaler ablation results:")
    print(
        scaler_df[
            ["label", "scale_phi", "accuracy_mean", "accuracy_std", "auc_mean", "auc_std"]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    print("\n" + "=" * 78)
    print("DIAGNOSTIC 2: ridge_gamma sweep (Godambe / linear)")
    print("=" * 78)

    for scale_phi in [True, False]:
        scaler_label = "with Phi scaler" if scale_phi else "no Phi scaler"
        print(f"\n--- {scaler_label} ---")
        for ridge_gamma in RIDGE_GAMMAS:
            print(f"  ridge_gamma={ridge_gamma:g} ...", flush=True)
            row = _run_cv(
                mask=mask,
                X=X,
                y=y,
                continuous_idx=continuous_idx,
                ordinal_idx=ordinal_idx,
                variable_names=variable_names,
                feature_type="linear",
                scale_phi=scale_phi,
                ridge_gamma=ridge_gamma,
            )
            row["diagnostic"] = "ridge_sweep"
            row["label"] = "Godambe"
            rows.append(row)

    ridge_df = pd.DataFrame([r for r in rows if r["diagnostic"] == "ridge_sweep"])
    print("\nRidge sweep results:")
    for scale_phi in [True, False]:
        sub = ridge_df[ridge_df["scale_phi"] == scale_phi].copy()
        print(f"\nscale_phi={scale_phi}:")
        print(
            sub[["ridge_gamma", "accuracy_mean", "accuracy_std", "auc_mean", "auc_std"]]
            .to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "geometry_diagnostics.csv"
    all_df.to_csv(csv_path, index=False)

    best = ridge_df.loc[ridge_df["auc_mean"].idxmax()]
    print("\n" + "=" * 78)
    print("BEST GODAMBE CONFIG (by AUC in ridge sweep)")
    print("=" * 78)
    print(
        f"ridge_gamma={best['ridge_gamma']:g}, scale_phi={bool(best['scale_phi'])}, "
        f"Acc={best['accuracy_mean']:.3f}±{best['accuracy_std']:.3f}, "
        f"AUC={best['auc_mean']:.3f}±{best['auc_std']:.3f}"
    )
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
