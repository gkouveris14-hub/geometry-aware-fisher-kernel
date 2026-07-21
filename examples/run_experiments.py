"""
Run the Heart Disease comparison table from the thesis (Experiment 1).

Outputs are written to examples/outputs/results.csv.
"""

from geometry_fisher.experiments import run_experiments, save_results_table

from config import DATA_PATH, OUTPUT_DIR

table = run_experiments(
    str(DATA_PATH),
    outer_splits=5,
    random_state=42,
    lambda_reg=0.01,
    ridge_gamma=1e-3,
    verbose=True,
)

csv_path, json_path = save_results_table(table, OUTPUT_DIR)
print(f"\nSaved:\n  {csv_path}\n  {json_path}")
