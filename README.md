# Geometry-Aware Generalized Fisher Kernel

Implementation of the method from:

> Konstantinos Gkouveris, *A Geometry-Aware Generalized Fisher Kernel Framework for Binary Classification of Mixed-Type Data under Composite Likelihood*, MSc Thesis, University of Nicosia, 2026.

## Method

The published classifier (`feature_type="linear"`) performs:

1. Class-specific composite likelihood fitting under a structural mask
2. Per-observation gradient features from both class models
3. Godambe sandwich whitening using the Hessian (H) and score covariance (J)
4. StandardScaler on features + logistic regression

Compare against `feature_type="raw"` for the internal ablation (unwhitened gradients).

## Dataset

Experiments use the UCI Heart Disease CSV with **9 selected variables** (5 continuous + 4 ordinal).

| Setting | Samples after cleaning | Class balance (0 / 1) |
|---------|------------------------|------------------------|
| **Default (all centers, thesis protocol)** | **531** | 207 / 324 |
| Cleveland only (`only_cleveland=True`) | 303 | 164 / 139 |

The raw CSV contains 920 rows across four hospitals. After selecting the 9 study variables and dropping rows with missing values, the **default loader keeps all centers (531 patients)**, matching `Last_hope.ipynb`.

The dataset file used in the thesis is bundled at `data/heart_disease_uci.csv` (UCI Heart Disease, four-center version).

## Installation

```bash
git clone https://github.com/gkouveris14-hub/geometry-aware-fisher-kernel.git
cd geometry-aware-fisher-kernel
pip install -e .
pip install -e ".[baselines]"   # optional: XGBoost baselines
```

## Reproduce paper experiments

```bash
python examples/run_paper_experiments.py
```

This runs 5-fold CV on the 531-sample thesis protocol and writes `examples/outputs/paper_experiments.csv`.

## Quick start

```python
from geometry_fisher.data import load_heart_disease
from geometry_fisher.structure import StructuralMask
from geometry_fisher.pipeline import GeometryFisherClassifier

X, y, names, cont_idx, ord_idx = load_heart_disease("data/heart_disease_uci.csv")

mask = StructuralMask.from_domain_knowledge(names, exogenous=["age", "sex"])

clf = GeometryFisherClassifier(
    mask="hand",
    mask_object=mask,
    feature_type="linear",   # proposed Godambe method
    lambda_reg=0.01,
    ridge_gamma=1e-3,
    scale_phi=True,
)
clf.fit(X, y, cont_idx, ord_idx, names)
```

## Package layout

```
geometry_fisher/
  composite.py          Composite likelihood (JAX)
  geometry.py           Godambe sandwich whitening
  features.py           Raw / Godambe feature construction
  pipeline.py           GeometryFisherClassifier
  nested_cv.py          Cross-validation experiment runner
  baselines.py            LR / RF / XGB baselines
  paper_experiments.py    Manuscript results table
  data.py                 Heart Disease loader
  structure.py            Structural masks
  visualization.py        Dependency heatmaps

examples/
  run_paper_experiments.py
  run_baselines.py
  run_heart_disease.py
  run_geometry_diagnostics.py
  plot_dependencies.py
```

## Tests

```bash
python -m pytest tests/ -v
```
