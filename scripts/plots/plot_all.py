#!/usr/bin/env python3
"""Regenerate every data-driven figure used in the paper.

Run from the repository root::

    uv run python scripts/plots/plot_all.py

The figures are written to ``figures/``.  Figure 1 (the motivation diagram)
and the architecture drawings are schematic and are not reproduced here.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent
SCRIPTS = (
    "plot_motivation.py",
    "visualize_stability.py",
    "plot_feature_inversion.py",
    "plot_stability_drop.py",
    "plot_augmentation.py",
)


def main() -> None:
    # Make sibling modules (``_common``) importable when running by path.
    sys.path.insert(0, str(PLOT_DIR))
    for script in SCRIPTS:
        print(f"\n=== {script} ===")
        runpy.run_path(str(PLOT_DIR / script), run_name="__main__")


if __name__ == "__main__":
    main()
