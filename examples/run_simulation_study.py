"""
Run synthetic geometry studies (no Heart Disease).

Tier 1 — score-level simulation with known H, J and sparse Godambe-aligned labels.
Tier 2 — optional full composite pipeline under heteroscedastic misspecification.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from geometry_fisher.simulation import run_simulation_battery

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "docs" / "results"


def _format_sweep(title: str, sweep: dict) -> str:
    lines = [title, "-" * len(title)]
    for n in sorted(sweep):
        s = sweep[n]
        lines.append(
            f"n={n:3d}  raw {s['raw_auc']['mean']:.3f} ± {s['raw_auc']['std']:.3f}   "
            f"hessian {s['hessian_auc']['mean']:.3f} ± {s['hessian_auc']['std']:.3f}   "
            f"godambe {s['godambe_auc']['mean']:.3f} ± {s['godambe_auc']['std']:.3f}"
        )
    dhj = sweep[sorted(sweep)[-1]]["delta_hj"]
    lines.append(f"mean ||H-J||/||H|| at largest n: {dhj['mean']:.3f} ± {dhj['std']:.3f}")
    return "\n".join(lines)


def _format_block(title: str, summary: dict) -> str:
    lines = [title, "-" * len(title)]
    for method in ("raw", "hessian", "godambe"):
        acc = summary[f"{method}_accuracy"]
        auc = summary[f"{method}_auc"]
        lines.append(
            f"{method:8s}  accuracy {acc['mean']:.3f} ± {acc['std']:.3f}   "
            f"AUC {auc['mean']:.3f} ± {auc['std']:.3f}"
        )
    dhj = summary["delta_hj_mean"]
    lines.append(f"mean ||H-J||/||H|| : {dhj['mean']:.3f} ± {dhj['std']:.3f}")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Running synthetic geometry studies (30 replicates)...")
    score_sweep, hetero = run_simulation_battery(n_replicates=30, base_seed=0)

    print()
    print(_format_sweep("Tier 1 — score geometry learning curves (AUC)", score_sweep))
    print()
    print(_format_block("Tier 2 — composite heteroscedastic misspecification", hetero))

    payload = {
        "score_geometry_sweep": {str(k): v for k, v in score_sweep.items()},
        "composite_hetero": hetero,
    }
    json_path = OUTPUT_DIR / "simulation_geometry_results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = []
    for n, summary in score_sweep.items():
        for metric_key, stats in summary.items():
            rows.append(
                {
                    "study": "score_geometry",
                    "n_samples": n,
                    "metric": metric_key,
                    "mean": stats["mean"],
                    "std": stats["std"],
                }
            )
    for metric_key, stats in hetero.items():
        rows.append(
            {
                "study": "composite_hetero",
                "n_samples": None,
                "metric": metric_key,
                "mean": stats["mean"],
                "std": stats["std"],
            }
        )
    csv_path = OUTPUT_DIR / "simulation_geometry_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    for output_dir in (RESULTS_DIR, OUTPUT_DIR):
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            output_dir / "simulation_geometry_results.csv",
            index=False,
        )
        json_path_out = output_dir / "simulation_geometry_results.json"
        json_path_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print("Saved:")
    print(f"  {OUTPUT_DIR / 'simulation_geometry_results.csv'}")
    print(f"  {RESULTS_DIR / 'simulation_geometry_results.csv'}")


if __name__ == "__main__":
    main()
