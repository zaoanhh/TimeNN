#!/usr/bin/env python3
"""Reproduce the augmentation-vs-stability figures (paper Figs. 7 and 8).

The CSV stores accuracy for

* ``simple_cnn`` trained with ``aug_factor`` copies of static augmentation
  (time shift / time warp / masking), and
* ``lms_cnn`` (TN-CNN) trained **without** augmentation

on SpeechCommands and Capture24.  Each row also records the model size
(``s``/``m``/``l``) and the fraction of the training set used.

Generated figures
-----------------
``figures/aug_<dataset>_radio/``
    fixed model size (``m``, ~9K parameters) and training ratios 0.2 and 0.5.

``figures/aug_<dataset>_size/``
    full training set and small (~3K) / large (~33K) model sizes.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import DATA_DIR, ensure_figure_dir
from timenn.plotting import (
    NATURE_PALETTE,
    nature_style_context,
    resolve_figsize,
    save_figure,
)

MODEL_LABELS = {"simple_cnn": "CNN", "lms_cnn": "TN-CNN"}
SIZE_LABELS = {"s": "small", "m": "medium", "l": "large"}
DATASET_LABELS = {"capture24": "Capture24", "speechcommands": "SpeechCommands"}

# Colour per (model, augmentation factor), matching the paper panels.
SERIES_COLORS: dict[tuple[str, int], str] = {
    ("simple_cnn", 1): NATURE_PALETTE["vermillion"],
    ("simple_cnn", 5): NATURE_PALETTE["orange"],
    ("simple_cnn", 10): NATURE_PALETTE["sky_blue"],
    ("lms_cnn", 1): NATURE_PALETTE["blue"],
    ("lms_cnn", 5): NATURE_PALETTE["bluish_green"],
    ("lms_cnn", 10): NATURE_PALETTE["reddish_purple"],
}
HATCH_LMS = "//"
HATCH_CNN = ""
WARP_VALUES: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)


@dataclass(frozen=True)
class ResultRow:
    dataset: str
    model: str
    model_size: str
    data_ratio: float
    aug_factor: int
    accs: dict[float, float]


def _warp_column(warp: float) -> str:
    return f"acc_warp_{warp:.1f}"


def load_rows(csv_path: Path, dataset: str) -> list[ResultRow]:
    rows: list[ResultRow] = []
    with csv_path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["dataset"] != dataset:
                continue
            accs = {
                warp: float(raw[_warp_column(warp)])
                for warp in WARP_VALUES
                if raw.get(_warp_column(warp)) not in (None, "")
            }
            if not accs:
                continue
            rows.append(
                ResultRow(
                    dataset=raw["dataset"],
                    model=raw["model"],
                    model_size=raw["model_size"],
                    data_ratio=float(raw["data_ratio"]),
                    aug_factor=int(float(raw["aug_factor"])),
                    accs=accs,
                )
            )
    return rows


def find_row(
    rows: list[ResultRow],
    model: str,
    model_size: str | None,
    data_ratio: float | None,
    aug_factor: int,
) -> ResultRow | None:
    for row in rows:
        if row.model != model or row.aug_factor != aug_factor:
            continue
        if model_size is not None and row.model_size != model_size:
            continue
        if data_ratio is not None and not np.isclose(row.data_ratio, data_ratio):
            continue
        return row
    return None


def series_label(model: str, aug_factor: int) -> str:
    if model == "lms_cnn":
        return MODEL_LABELS[model]
    return f"{MODEL_LABELS[model]} (aug={aug_factor})"


def collect_y_limits(curves: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate(curves)
    lower = max(0.0, float(values.min()) - 3.0)
    upper = min(100.0, float(values.max()) + 6.0)
    if upper - lower < 8.0:
        center = (upper + lower) / 2.0
        lower = max(0.0, center - 4.0)
        upper = min(100.0, center + 4.0)
    return lower, upper


def plot_group(
    series_rows: list[tuple[str, int, ResultRow | None]],
    output_path: Path,
) -> Path | None:
    """Draw grouped bars for a list of ``(model, aug_factor, row)`` series."""
    valid = [(model, aug, row) for model, aug, row in series_rows if row is not None]
    if not valid:
        print(f"  [skip] {output_path.name}: no data")
        return None

    curves: list[np.ndarray] = []
    labels: list[str] = []
    colors: list[str] = []
    hatches: list[str] = []
    for model, aug, row in valid:
        curves.append(np.array([row.accs[w] for w in WARP_VALUES]) * 100.0)
        labels.append(series_label(model, aug))
        colors.append(SERIES_COLORS.get((model, aug), NATURE_PALETTE["black"]))
        hatches.append(HATCH_LMS if model == "lms_cnn" else HATCH_CNN)

    n_series = len(curves)
    width = min(0.8 / max(n_series, 1), 0.32)
    centers = np.arange(len(WARP_VALUES), dtype=float)

    with nature_style_context(font_size=6.0, use_grid=True):
        fig, ax = plt.subplots(figsize=resolve_figsize("single"))
        for idx in range(n_series):
            offset = (idx - (n_series - 1) / 2.0) * width
            bars = ax.bar(
                centers + offset,
                curves[idx],
                width=width,
                color=colors[idx],
                edgecolor="black",
                linewidth=0.6,
                label=labels[idx],
            )
            for bar in bars:
                bar.set_hatch(hatches[idx])

        ax.set_xticks(centers)
        ax.set_xticklabels([f"{w:.1f}" for w in WARP_VALUES])
        ax.set_xlabel("Warp")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(collect_y_limits(curves))
        ax.legend(
            ncol=n_series if n_series <= 3 else 4,
            handlelength=1.0,
            handletextpad=0.3,
            columnspacing=0.6,
            fontsize=5.0,
            loc="upper right",
            frameon=False,
        )
        fig.tight_layout()
        save_figure(fig, output_path)

    print(f"[saved] {output_path}")
    return output_path


def plot_radio_group(rows: list[ResultRow], dataset: str) -> list[Path]:
    """Fixed model size, varying training-set fraction."""
    outputs: list[Path] = []
    out_dir = ensure_figure_dir(f"aug_{dataset}_radio")
    for ratio in (0.2, 0.5):
        series = [
            (
                model,
                aug,
                find_row(rows, model, model_size="m", data_ratio=ratio, aug_factor=aug),
            )
            for model in ("simple_cnn", "lms_cnn")
            for aug in (1, 10)
        ]
        output = out_dir / f"m_ratio{ratio:g}_aug1_10_accuracy.pdf"
        saved = plot_group(series, output)
        if saved:
            outputs.append(saved)
    return outputs


def plot_size_group(rows: list[ResultRow], dataset: str) -> list[Path]:
    """Full training set, varying model size."""
    outputs: list[Path] = []
    out_dir = ensure_figure_dir(f"aug_{dataset}_size")
    for size in ("s", "l"):
        series = [
            (
                model,
                aug,
                find_row(rows, model, model_size=size, data_ratio=1.0, aug_factor=aug),
            )
            for model in ("simple_cnn", "lms_cnn")
            for aug in (1, 5, 10)
        ]
        output = out_dir / f"size_{size}_ratio1_aug1_5_10_accuracy.pdf"
        saved = plot_group(series, output)
        if saved:
            outputs.append(saved)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DATA_DIR / "aug_results.csv")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", "capture24", "speechcommands"],
    )
    args = parser.parse_args()

    datasets = (
        ("capture24", "speechcommands")
        if args.dataset == "all"
        else (args.dataset,)
    )
    for dataset in datasets:
        rows = load_rows(args.csv, dataset)
        print(f"Loaded {len(rows)} rows for {DATASET_LABELS[dataset]}")
        plot_radio_group(rows, dataset)
        plot_size_group(rows, dataset)


if __name__ == "__main__":
    main()
