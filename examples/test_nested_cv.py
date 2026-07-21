import numpy as np
from geometry_fisher.structure import StructuralMask
from geometry_fisher.nested_cv import NestedCVExperiment

np.random.seed(42)
n_samples = 300
n_features = 6

variable_names = ["age", "sex", "bp", "chol", "thalach", "oldpeak"]
continuous_idx = np.array([0, 2, 3, 4, 5])
ordinal_idx = np.array([1])

X = np.random.randn(n_samples, n_features)
X[:, 1] = np.random.randint(0, 2, size=n_samples)
X[:, 0] = X[:, 0] * 10 + 50
y = (X[:, 4] + X[:, 5] + np.random.randn(n_samples) * 0.5 > 0).astype(int)

print("Running Nested CV with hand-specified mask...")

mask = StructuralMask.from_domain_knowledge(
    variable_names=variable_names,
    exogenous=["age", "sex"]
)

experiment = NestedCVExperiment(
    mask="hand",
    mask_object=mask,
    feature_type="godambe",
    outer_splits=3,          # small number for quick test
    random_state=42,
)

result = experiment.run(
    X, y,
    continuous_idx=continuous_idx,
    ordinal_idx=ordinal_idx,
    variable_names=variable_names,
)

print("\nDone.")