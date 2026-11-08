#!/usr/bin/env python3
"""Reproduce the motivation figure (paper Fig. 2 / ``motivation1-sub.pdf``).

The experiment trains seven standard networks on Widar, XRF55 and XRF55-RFID
at the full training ratio, then evaluates each model on (i) the clean test set
and (ii) a time-warped test set.  The figure is a dumbbell plot: the circle is
the clean accuracy and the triangle is the warped accuracy.

Result data
-----------
``data/results/motivation_results.csv`` with columns
``Dataset, Ratio, NumSamples, Model, TestAcc, AugTestAcc, ModelPath``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import DATA_DIR, ensure_figure_dir
from timenn.plotting import (
    NATURE_COLOR_CYCLE,
    nature_style_context,
    save_figure,
)

# Order used in the paper.
MODELS = ("resnet18", "resnet34", "cnn", "inception", "transformer", "lstm", "gru")
MODEL_LABELS = ("ResNet-18", "ResNet-34", "CNN", "Inception", "Transformer", "LSTM", "GRU")
DATASETS = ("widar", "xrf55", "xrf55_rfid")
DATASET_LABELS = ("Widar", "XRF55", "XRF55-RFID")

# Width of one dumbbell column; the three datasets are offset by +/- this value.
DATASET_OFFSET = 0.12


def load_full_ratio_results(csv_path: Path) -> dict[str, dict[str, tuple[float, float]]]:
    """Return ``{dataset: {model: (clean_acc, warped_acc)}}`` at ratio 1.0."""
    data: dict[str, dict[str, tuple[float, float]]] = {}
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if float(row["Ratio"]) != 1.0:
                continue
            dataset = row["Dataset"].strip()
            model = row["Model"].strip().lower()
            data.setdefault(dataset, {})[model] = (
                float(row["TestAcc"]),
                float(row["AugTestAcc"]),
            )
    return data


def plot_dumbbell(ax, x: float, clean: float, warped: float, color: str) -> None:
    """Draw one vertical connector with a circle (clean) and triangle (warped)."""
    ax.plot([x, x], [clean, warped], color=color, linewidth=0.8, zorder=2)
    ax.scatter([x], [clean], color=color, s=22, zorder=5, marker="o")
    ax.scatter([x], [warped], color=color, s=22, zorder=5, marker="v")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DATA_DIR / "motivation_results.csv",
        help="Motivation experiment CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF (default: figures/motivation1-sub.pdf).",
    )
    args = parser.parse_args()

    output = args.output or ensure_figure_dir() / "motivation1-sub.pdf"
    data = load_full_ratio_results(args.csv)
    palette = list(NATURE_COLOR_CYCLE)
    centers = np.arange(len(MODELS), dtype=float)

    # The paper panel is a compact single-column figure.
    with nature_style_context(font_size=6, use_grid=False):
        fig, ax = plt.subplots(figsize=(3.35, 1.18))

        # The colours follow the paper: Widar=blue, XRF55=orange,
        # XRF55-RFID=green.
        for ds_idx, dataset in enumerate(DATASETS):
            ds_data = data.get(dataset, {})
            color = palette[(ds_idx + 1) % len(palette)]

            for model_idx, model in enumerate(MODELS):
                clean, warped = ds_data.get(model, (np.nan, np.nan))
                if np.isnan(clean) or np.isnan(warped):
                    continue
                x = centers[model_idx] + (ds_idx - 1) * DATASET_OFFSET
                plot_dumbbell(ax, x, clean, warped, color)

        ax.set_xticks(centers)
        ax.set_xticklabels(MODEL_LABELS)
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))

        # Dataset colour entries + shape legend.
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                color=palette[(i + 1) % len(palette)],
                linewidth=1.2,
                label=DATASET_LABELS[i],
            )
            for i in range(len(DATASETS))
        ]
        legend_elements.extend(
            [
                plt.Line2D(
                    [0], [0], marker="o", color="grey", linestyle="None",
                    markersize=4, label="Base",
                ),
                plt.Line2D(
                    [0], [0], marker="v", color="grey", linestyle="None",
                    markersize=4, label="Shift",
                ),
            ]
        )
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=5,
            handlelength=1.0,
            handletextpad=0.4,
            labelspacing=0.2,
            borderpad=0.2,
        )

        fig.tight_layout()
        save_figure(fig, output)

    print(f"[saved] {output}")
    for dataset in DATASETS:
        for model in MODELS:
            clean, warped = data.get(dataset, {}).get(model, (np.nan, np.nan))
            print(f"  {dataset:11s} {model:11s} {clean:.4f} -> {warped:.4f}")


if __name__ == "__main__":
    main()
