"""Shared paths and helpers for the figure-reproduction scripts."""

from __future__ import annotations

from pathlib import Path

# scripts/plots/_common.py -> project root is two levels up from this file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "results"
FIGURE_DIR = PROJECT_ROOT / "figures"


def ensure_figure_dir(subdir: str | None = None) -> Path:
    """Create (if needed) and return a directory under ``figures/``."""
    path = FIGURE_DIR if subdir is None else FIGURE_DIR / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path
