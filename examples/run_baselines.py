"""
Reproduce thesis baseline comparisons on the Heart Disease dataset.

Expected thesis summary (531 samples, 5-fold CV):
- Logistic Regression: Acc ~0.780, AUC ~0.853
- Random Forest:       Acc ~0.780, AUC ~0.847
- XGBoost:             Acc ~0.787, AUC ~0.830
"""

from geometry_fisher.baselines import run_baseline_cv, summaries_to_dataframe
from geometry_fisher.data import load_heart_disease

DATA_PATH = r"C:\ΑΡΧΕΙΑ\UNIC\Thesis\heart_disease_uci.csv"

X, y, variable_names, _, _ = load_heart_disease(
    path=DATA_PATH,
    binary_target=True,
    only_cleveland=False,
)

result = run_baseline_cv(
    X,
    y,
    variable_names=variable_names,
    outer_splits=5,
    random_state=42,
    verbose=True,
)

summary_df = summaries_to_dataframe(result)
print("\nSummary table:")
print(summary_df.round(4).to_string(index=False))
