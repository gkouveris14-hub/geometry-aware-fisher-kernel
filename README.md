# Geometry-Aware Generalized Fisher Kernel

**Geometry-Aware Generalized Fisher Kernel Framework for Binary Classification of Mixed-Type Data under Composite Likelihood**

This repository contains a clean implementation of the method developed in:

> Konstantinos Gkouveris  
> *A Geometry-Aware Generalized Fisher Kernel Framework for Binary Classification of Mixed-Type Data under Composite Likelihood*  
> MSc Thesis, University of Nicosia, 2026

---

## Overview

The method combines:
- Class-specific **composite likelihood** models for mixed continuous + ordinal data
- A **structural mask** (hand-specified or data-driven via stability selection)
- **Godambe information geometry** (instead of classical Fisher information) to obtain a proper Riemannian metric under composite likelihood
- Geometry-aware gradient features that are fed into a downstream classifier

The framework supports a full ablation suite that isolates the contribution of:
- Raw gradients
- Fisher-only whitening (using only the variability matrix \(J\))
- Full Godambe whitening
- Quadratic (Mahalanobis) scores
- Combined linear + quadratic features

---

## Installation

```bash
git clone https://github.com/gkouveris14-hub/geometry-aware-fisher-kernel.git
cd geometry-aware-fisher-kernel
pip install -e .

Main dependencies:
numpy, scipy, scikit-learn, pandas, jax, jaxlib, optax, matplotlib, seaborn, causal-learn


**Quick Start**
from geometry_fisher.data import load_heart_disease
from geometry_fisher.structure import StructuralMask
from geometry_fisher.pipeline import GeometryFisherClassifier

# Load data
X, y, variable_names, continuous_idx, ordinal_idx = load_heart_disease(
    path="path/to/heart_disease_uci.csv",
    only_cleveland=False,
)

# Define a hand-specified structural mask
mask = StructuralMask.from_domain_knowledge(
    variable_names=variable_names,
    exogenous=["age", "sex"]
)

# Fit the geometry-aware classifier
clf = GeometryFisherClassifier(
    mask="hand",
    mask_object=mask,
    feature_type="linear",      # or "raw", "fisher_only", "quadratic", "full"
    lambda_reg=0.01,
)

clf.fit(X, y, continuous_idx, ordinal_idx, variable_names)

# Predict
y_pred = clf.predict(X)
y_proba = clf.predict_proba(X)


**Nested Cross-Validation**
from geometry_fisher.nested_cv import NestedCVExperiment

experiment = NestedCVExperiment(
    mask="hand",
    mask_object=mask,
    feature_type="linear",
    outer_splits=5,
)

result = experiment.run(X, y, continuous_idx, ordinal_idx, variable_names)

print(result.mean_accuracy, result.mean_auc)

**Ablation Variants**
feature_type,Description,Variant
"""raw""",Concatenated raw gradients,A
"""fisher_only""",Whitened only with (J^{-1/2}),B
"""linear""",Full Godambe whitening (thesis original),C
"""quadratic""","Mahalanobis scores ([q_0, q_1, q_1-q_0])",D
"""full""",Linear Godambe features + quadratic scores,E

**Data-Driven Mask (Stability selection)**
clf = GeometryFisherClassifier(
    mask="data_driven",
    mask_params={
        "alpha": 0.05,
        "tau_stab": 0.6,
        "B": 50,
        "exogenous": ["age", "sex"]
    },
    feature_type="linear",
)

**Visualization**
from geometry_fisher.visualization import (
    plot_class_dependencies,
    plot_difference_heatmap,
    plot_mask,
)

plot_mask(mask)
plot_class_dependencies(clf.model_0_, clf.model_1_, variable_names)
plot_difference_heatmap(clf.model_0_, clf.model_1_, variable_names)


geometry_fisher/
├── data.py              # Heart Disease data loader
├── structure.py         # StructuralMask (hand + stability selection)
├── composite.py         # Class-specific composite likelihood (JAX)
├── geometry.py          # Godambe geometry + shrinkage
├── features.py          # Feature construction (ablation suite)
├── pipeline.py          # High-level GeometryFisherClassifier
├── nested_cv.py         # Nested cross-validation experiment
└── visualization.py     # Heatmaps and dependency plots

examples/
├── run_heart_disease.py
├── run_ablations.py
└── plot_dependencies.py

