"""
Simple end-to-end test of the GeometryFisherClassifier on synthetic data.
"""

import numpy as np
from geometry_fisher.structure import StructuralMask
from geometry_fisher.pipeline import GeometryFisherClassifier

print("=" * 60)
print("Geometry-Aware Fisher Kernel – End-to-End Test")
print("=" * 60)

# -------------------------------------------------
# 1. Create synthetic mixed-type data
# -------------------------------------------------
np.random.seed(42)
n_samples = 300
n_features = 6

# Variable names
variable_names = ["age", "sex", "bp", "chol", "thalach", "oldpeak"]

# Continuous indices: age, bp, chol, thalach, oldpeak
# Ordinal indices: sex (treated as ordinal 0/1 for simplicity)
continuous_idx = np.array([0, 2, 3, 4, 5])
ordinal_idx = np.array([1])

X = np.random.randn(n_samples, n_features)
X[:, 1] = np.random.randint(0, 2, size=n_samples)          # sex as 0/1
X[:, 0] = X[:, 0] * 10 + 50                               # age-like
y = (X[:, 4] + X[:, 5] + np.random.randn(n_samples) * 0.5 > 0).astype(int)

print(f"\nData shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")
print(f"Continuous indices: {continuous_idx}")
print(f"Ordinal indices: {ordinal_idx}")

# -------------------------------------------------
# 2. Create a hand-specified mask
# -------------------------------------------------
mask = StructuralMask.from_domain_knowledge(
    variable_names=variable_names,
    exogenous=["age", "sex"]
)
print(f"\nMask: {mask}")
print(f"Number of free parameters: {mask.n_params}")

# -------------------------------------------------
# 3. Create and fit the classifier
# -------------------------------------------------
print("\nFitting GeometryFisherClassifier...")

clf = GeometryFisherClassifier(
    mask="hand",
    mask_object=mask,
    lambda_reg=0.01,
    ridge_gamma=1e-3,
    feature_type="linear",
    C=1.0,
)

clf.fit(
    X, y,
    continuous_idx=continuous_idx,
    ordinal_idx=ordinal_idx,
    variable_names=variable_names,
)

print("Fitting completed successfully!")

# -------------------------------------------------
# 4. Inspect what was learned
# -------------------------------------------------
print("\n" + "-" * 40)
print("Model 0 (class 0):", clf.model_0_)
print("Model 1 (class 1):", clf.model_1_)
print("Geometry 0:", clf.geometry_0_)
print("Geometry 1:", clf.geometry_1_)

print(f"\nTheta hat (class 0) shape: {clf.model_0_.theta_hat_.shape}")
print(f"Theta hat (class 1) shape: {clf.model_1_.theta_hat_.shape}")

# -------------------------------------------------
# 5. Transform and predict
# -------------------------------------------------
print("\n" + "-" * 40)
print("Transforming data...")
Phi = clf.transform(X)
print(f"Feature matrix shape: {Phi.shape}")

print("\nPredicting...")
y_pred = clf.predict(X)
y_proba = clf.predict_proba(X)

accuracy = np.mean(y_pred == y)
print(f"Training accuracy: {accuracy:.3f}")

print("\nFirst 10 predictions vs true labels:")
for i in range(10):
    print(f"  True: {y[i]}  |  Pred: {y_pred[i]}  |  Proba: {y_proba[i]}")

print("\n" + "=" * 60)
print("TEST FINISHED SUCCESSFULLY")
print("=" * 60)