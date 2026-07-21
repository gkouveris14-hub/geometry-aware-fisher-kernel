"""Shared paths for example scripts."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "heart_disease_uci.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
RESULTS_DIR = REPO_ROOT / "docs" / "results"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
