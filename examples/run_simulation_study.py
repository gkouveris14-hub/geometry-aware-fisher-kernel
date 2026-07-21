"""
Run Experiment 3: synthetic score-geometry study (raw vs Godambe only).

Outputs are written to docs/results/ and examples/outputs/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from geometry_fisher.simulation import run_simulation_battery

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "docs" / "results"


def _format_sweep(sweep: dict) -> str:
    lines = [
        "Experiment 3 — score geometry (raw vs Godambe)",
        "-" * 50,
    ]
    for n in sorted(sweep):
        s = sweep[n]
        lines.append(
            f"n={n:3d}  raw acc {s['raw_accuracy']['mean']:.3f} ± {s['raw_accuracy']['std']:.3f}   "
            f"godambe acc {s['godambe_accuracy']['mean']:.3f} ± {s['godambe_accuracy']['std']:.3f}   "
            f"raw AUC {s['raw_auc']['mean']:.3f} ± {s['raw_auc']['std']:.3f}   "
            f"godambe AUC {s['godambe_auc']['mean']:.3f} ± {s['godambe_auc']['std']:.3f}"
        )
    dhj = sweep[sorted(sweep)[-1]]["delta_hj"]
    lines.append(f"mean ||H-J||/||H|| at largest n: {dhj['mean']:.3f} ± {dhj['std']:.3f}")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Running Experiment 3 (30 replicates per sample size)...")
    score_sweep = run_simulation_battery(n_replicates=30, base_seed=0)

    print()
    print(_format_sweep(score_sweep))

    payload = {"score_geometry_sweep": {str(k): v for k, v in score_sweep.items()}}
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

    for output_dir in (RESULTS_DIR, OUTPUT_DIR):
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            output_dir / "simulation_geometry_results.csv",
            index=False,
        )
        (output_dir / "simulation_geometry_results.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    print()
    print("Saved:")
    print(f"  {RESULTS_DIR / 'simulation_geometry_results.csv'}")
    print(f"  {OUTPUT_DIR / 'simulation_geometry_results.csv'}")


if __name__ == "__main__":
    main()
