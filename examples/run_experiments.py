"""
Run the Heart Disease comparison table from the thesis (Experiment 1).

Outputs are written to docs/results/experiment1_results.csv and
examples/outputs/results.csv.
"""

from geometry_fisher.experiments import run_experiments, save_results_table

from config import DATA_PATH, OUTPUT_DIR, RESULTS_DIR

table = run_experiments(
    str(DATA_PATH),
    outer_splits=5,
    random_state=42,
    lambda_reg=0.01,
    ridge_gamma=1e-3,
    verbose=True,
)

csv_paths = []
for output_dir in (RESULTS_DIR, OUTPUT_DIR):
    csv_path, json_path = save_results_table(
        table,
        output_dir,
        stem="experiment1_results" if output_dir == RESULTS_DIR else "results",
    )
    csv_paths.extend([csv_path, json_path])

print("\nSaved:")
for path in csv_paths:
    print(f"  {path}")
