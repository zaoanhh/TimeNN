#!/usr/bin/env python3
"""Reproduce the temporal-stability bar charts from ``with_tau.csv``.

Generated figures
-----------------
* ``figures/lms_cnn_stability_bar/lms_cnn_vs_simple_cnn_tau_0_0_5_relative_drop.pdf``
* ``figures/lms_resnet_stability_bar/lms_resnet_vs_resnet1d_tau_0_0_3_relative_drop.pdf``
* ``figures/lmsnet_tau_0.1_0.8_relative_drop.pdf``
* ``figures/lms_tproj_vs_transformer_bars/timit_tau_0_0.2_accuracy.pdf``
* ``figures/lms_tenc1_vs_transformer_bars/timit_tau_0_0.2_accuracy.pdf``

The relative drop of a model at distortion ``tau`` is
``1 - acc(tau) / acc(0)`` and is averaged over all requested non-zero taus.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _common import DATA_DIR, FIGURE_DIR, ensure_figure_dir
from timenn.plotting import NATURE_PALETTE, plot_bar_chart

MODEL_LABELS = {
    "simple_cnn": "CNN",
    "resnet1d": "ResNet",
    "transformer": "Transformer",
    "lms_cnn": "TN-CNN",
    "lms_resnet": "TN-ResNet",
    "lms_tenc1": "TN-TEnc1",
    "lms_tproj": "TN-TProj",
    "lmsnet": "TN-Net",
}

DATASET_LABELS = {
    "capture24": "Capture24",
    "medley": "Medley",
    "speechcommands": "SpeechCommands",
    "timit": "TIMIT",
    "widar": "Widar",
    "xrf55_rfid": "XRF55-RFID",
    "xrf55_wifi": "XRF55-WiFi",
}

LMS_COLOR = NATURE_PALETTE["blue"]
BASELINE_COLOR = NATURE_PALETTE["vermillion"]
ACCURACY_TAUS = (0.0, 0.1, 0.2)


@dataclass(frozen=True)
class StabilityGroup:
    slug: str
    lms_model: str
    baseline_model: str
    datasets: tuple[str, ...]
    tau_end: float


@dataclass(frozen=True)
class AccuracyGroup:
    slug: str
    models: tuple[str, str]
    datasets: tuple[str, ...]


STABILITY_GROUPS = (
    StabilityGroup(
        slug="lms_cnn_stability_bar",
        lms_model="lms_cnn",
        baseline_model="simple_cnn",
        datasets=("capture24", "xrf55_rfid", "medley", "widar"),
        tau_end=0.4,
    ),
    StabilityGroup(
        slug="lms_resnet_stability_bar",
        lms_model="lms_resnet",
        baseline_model="resnet1d",
        datasets=("capture24", "timit", "speechcommands"),
        tau_end=0.3,
    ),
)

ACCURACY_GROUPS = (
    AccuracyGroup(
        slug="lms_tproj_vs_transformer_bars",
        models=("lms_tproj", "transformer"),
        datasets=("timit",),
    ),
    AccuracyGroup(
        slug="lms_tenc1_vs_transformer_bars",
        models=("lms_tenc1", "transformer"),
        datasets=("timit",),
    ),
)


def _tau_column(tau: float) -> str:
    return f"acc_warp_{tau:.1f}"


def load_results(csv_path: Path):
    """Return ``(taus, results)`` where results[(dataset, model)][tau] = acc."""
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    columns = sorted(
        (name for name in rows[0] if name.startswith("acc_warp_")),
        key=lambda name: float(name.rsplit("_", 1)[-1]),
    )
    taus = [float(name.rsplit("_", 1)[-1]) for name in columns]
    results: dict[tuple[str, str], dict[float, float]] = {}
    for row in rows:
        results[(row["dataset"], row["model"])] = {
            tau: float(row[name]) for tau, name in zip(taus, columns)
        }
    return taus, results


def curve(results, dataset: str, model: str, taus) -> np.ndarray:
    row = results[(dataset, model)]
    return np.array([row[tau] for tau in taus], dtype=float)


def relative_drop(results, dataset: str, model: str, taus) -> np.ndarray:
    values = curve(results, dataset, model, taus)
    if values[0] <= 0.0:
        raise ValueError(f"Non-positive reference accuracy for {dataset}/{model}")
    return (1.0 - values[1:] / values[0]) * 100.0


def plot_stability_group(group: StabilityGroup, taus: list[float], results) -> list[Path]:
    selected = [tau for tau in taus if tau <= group.tau_end]
    if len(selected) < 2:
        raise ValueError(f"Not enough tau points up to {group.tau_end}")

    lms_values = [
        relative_drop(results, ds, group.lms_model, selected).mean()
        for ds in group.datasets
    ]
    base_values = [
        relative_drop(results, ds, group.baseline_model, selected).mean()
        for ds in group.datasets
    ]
    y_lim = (0.0, min(100.0, max(lms_values + base_values) + 5.0))

    output_dir = ensure_figure_dir(group.slug)
    end_token = "5" if group.tau_end == 0.4 else f"{group.tau_end:.1f}".split(".")[1]
    primary = (
        output_dir
        / f"{group.lms_model}_vs_{group.baseline_model}_tau_0_0_{end_token}_relative_drop.pdf"
    )
    plot_bar_chart(
        values=[lms_values, base_values],
        save_path=primary,
        x_labels=[DATASET_LABELS[ds] for ds in group.datasets],
        series_labels=[
            MODEL_LABELS[group.lms_model],
            MODEL_LABELS[group.baseline_model],
        ],
        x_label="Dataset",
        y_label="Average relative accuracy drop (%)",
        colors=[LMS_COLOR, BASELINE_COLOR],
        figsize="single",
        font_size=6.0,
        use_grid=True,
        legend=True,
        y_lim=y_lim,
    )

    outputs = [primary]
    # Also emit the descriptive filename (tau_end 0.4) for clarity.
    if group.tau_end == 0.4:
        secondary = (
            output_dir
            / f"{group.lms_model}_vs_{group.baseline_model}_tau_0_0_4_relative_drop.pdf"
        )
        plot_bar_chart(
            values=[lms_values, base_values],
            save_path=secondary,
            x_labels=[DATASET_LABELS[ds] for ds in group.datasets],
            series_labels=[
                MODEL_LABELS[group.lms_model],
                MODEL_LABELS[group.baseline_model],
            ],
            x_label="Dataset",
            y_label="Average relative accuracy drop (%)",
            colors=[LMS_COLOR, BASELINE_COLOR],
            figsize="single",
            font_size=6.0,
            use_grid=True,
            legend=True,
            y_lim=y_lim,
        )
        outputs.append(secondary)

    print(
        f"[saved] {primary} | tau=0.0..{group.tau_end:.1f} | "
        f"avg_drop={np.mean(lms_values):.2f}% vs {np.mean(base_values):.2f}%"
    )
    return outputs


def plot_accuracy_group(group: AccuracyGroup, results) -> Path:
    """Grouped accuracy bars for a transformer variant and its baseline."""
    outputs = []
    for dataset in group.datasets:
        curves = [curve(results, dataset, model, ACCURACY_TAUS) * 100.0 for model in group.models]
        values = np.concatenate(curves)
        lower = max(0.0, float(values.min()) - 3.0)
        upper = min(100.0, float(values.max()) + 8.0)
        if upper - lower < 8.0:
            lower = max(0.0, (upper + lower) / 2.0 - 4.0)
            upper = min(100.0, lower + 8.0)

        output = (
            ensure_figure_dir(group.slug)
            / f"{dataset}_tau_0_{ACCURACY_TAUS[-1]:.1f}_accuracy.pdf"
        )
        plot_bar_chart(
            values=curves,
            save_path=output,
            x_labels=[f"{tau:.1f}" for tau in ACCURACY_TAUS],
            series_labels=[MODEL_LABELS[m] for m in group.models],
            x_label="Tau",
            y_label="Accuracy (%)",
            colors=[LMS_COLOR, BASELINE_COLOR],
            figsize="single",
            font_size=6.0,
            use_grid=True,
            legend=True,
            y_lim=(lower, upper),
        )
        print(f"[saved] {output} | {DATASET_LABELS[dataset]} {group.models}")
        outputs.append(output)
    return outputs[0]


def plot_lmsnet_relative_drop(taus: list[float], results) -> Path:
    drop_taus = [tau for tau in taus if 0.1 <= tau <= 0.8]
    dataset_drops = []
    valid_datasets = []
    for dataset in ("capture24", "medley"):
        drops = relative_drop(results, dataset, "lmsnet", [taus[0]] + drop_taus)
        dataset_drops.append(drops)
        valid_datasets.append(dataset)

    all_values = np.concatenate(dataset_drops)
    # +5 leaves the same headroom as the paper figure.
    y_lim = (0.0, min(100.0, float(all_values.max()) + 5.0))

    output = FIGURE_DIR / "lmsnet_tau_0.1_0.8_relative_drop.pdf"
    plot_bar_chart(
        values=dataset_drops,
        save_path=output,
        x_labels=[f"{tau:.1f}" for tau in drop_taus],
        series_labels=[DATASET_LABELS[ds] for ds in valid_datasets],
        x_label="Tau",
        y_label="Relative accuracy drop (%)",
        colors=[NATURE_PALETTE["blue"], NATURE_PALETTE["vermillion"]],
        figsize="single",
        font_size=6.0,
        use_grid=True,
        legend=True,
        y_lim=y_lim,
    )
    print(f"[saved] {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DATA_DIR / "with_tau.csv")
    args = parser.parse_args()

    taus, results = load_results(args.csv)
    print(f"Loaded {len(results)} model/dataset rows; taus={taus}")

    for group in STABILITY_GROUPS:
        plot_stability_group(group, taus, results)
    for group in ACCURACY_GROUPS:
        plot_accuracy_group(group, results)
    plot_lmsnet_relative_drop(taus, results)


if __name__ == "__main__":
    main()
