"""
Generate the paper-ready comparison table:

  - Baselines: Logistic Regression, Random Forest, XGBoost
  - Ablation:  Raw gradient features (no Godambe geometry)
  - Proposed:  Godambe-whitened gradient features

Results are saved to examples/outputs/paper_experiments.csv
"""

from pathlib import Path

from geometry_fisher.paper_experiments import run_paper_experiments, save_paper_table

DATA_PATH = r"C:\ΑΡΧΕΙΑ\UNIC\Thesis\heart_disease_uci.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

table = run_paper_experiments(
    DATA_PATH,
    outer_splits=5,
    random_state=42,
    lambda_reg=0.01,
    ridge_gamma=1e-3,
    verbose=True,
)

csv_path, json_path = save_paper_table(table, OUTPUT_DIR)
print(f"\nSaved:\n  {csv_path}\n  {json_path}")
