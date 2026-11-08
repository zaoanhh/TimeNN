#!/usr/bin/env python3
"""Lightweight installation check.

Verifies that the package imports, that the core Time-NN block runs a forward
pass, and that the result tables are present and parseable.  This
script is a good first command after ``uv sync``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np
import torch

import timenn
from timenn.models.lms_net import FilterConfig, LMSBlock

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "results"


def main() -> None:
    print(f"timenn           {timenn.__version__}")
    print(f"torch            {torch.__version__}")
    print(f"numpy            {np.__version__}")
    print(f"matplotlib       {matplotlib.__version__}")

    block = LMSBlock(
        c_in=1,
        filter_config=FilterConfig(
            filter_type="morlet",
            scale_type="linear",
            num_scales=8,
            f_min=0.005,
            f_max=0.25,
            kernel_length=64,
            pool_type="learnable",
        ),
        intra_out_per_group=4,
        inter_out=32,
    )
    x = torch.randn(2, 1, 512)
    with torch.no_grad():
        high, stable = block(x)
    print(f"LMSBlock forward: high={tuple(high.shape)} stable={tuple(stable.shape)}")

    for name, required in (
        ("with_tau.csv", {"dataset", "model", "acc_warp_0.0", "acc_warp_0.8"}),
        ("aug_results.csv", {"dataset", "model", "aug_factor", "acc_warp_0.3"}),
        ("motivation_results.csv", {"Dataset", "Model", "TestAcc", "AugTestAcc"}),
    ):
        path = DATA / name
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        missing = required.difference(rows[0].keys())
        if missing:
            raise SystemExit(f"{name}: missing columns {sorted(missing)}")
        print(f"{name}: {len(rows)} rows OK")

    print("\nInstallation check passed.")


if __name__ == "__main__":
    main()
