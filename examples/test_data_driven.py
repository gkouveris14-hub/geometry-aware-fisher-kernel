import numpy as np
from geometry_fisher.structure import StructuralMask
from geometry_fisher.pipeline import GeometryFisherClassifier

np.random.seed(42)
X = np.random.randn(200, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
names = ['a', 'b', 'c', 'd', 'e']

print("Testing data-driven mask...")

clf = GeometryFisherClassifier(
    mask="data_driven",
    mask_params={
        "alpha": 0.05,
        "tau_stab": 0.5,
        "B": 20,               # small for quick test
        "exogenous": ["a"]
    },
    feature_type="linear"
)

clf.fit(
    X, y,
    continuous_idx=[0, 1, 2, 3, 4],
    ordinal_idx=[],
    variable_names=names
)

print("Success!")
print("Learned mask n_params:", clf.mask_.n_params)
print("Mask matrix:")
print(clf.mask_.matrix)