"""
Geometry-aware Fisher kernel for mixed-type data under composite likelihood.
"""

from .data import load_heart_disease
from .pipeline import GeometryFisherClassifier
from .structure import StructuralMask

__version__ = "0.1.0"
__all__ = [
    "GeometryFisherClassifier",
    "StructuralMask",
    "load_heart_disease",
]
