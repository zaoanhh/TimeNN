#!/usr/bin/env python3
"""Run the motivation experiment (paper Fig. 3) over datasets/models/ratios.

Example
-------
::

    uv run python scripts/experiments/run_motivation_experiment.py \
        --widar-root /data/widar3/CSI_Processed/20181130 \
        --xrf55-root /data/xrf55-all \
        --output-csv data/results/motivation_results.csv

The script skips ``(dataset, model, ratio)`` combinations that already have a
row in the output CSV, so it can be interrupted and resumed.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER = PROJECT_ROOT / "scripts" / "experiments" / "train_motivation_model.py"

DEFAULT_MODELS = ("cnn", "resnet18", "resnet34", "transformer", "inception", "lstm", "gru")
DEFAULT_DATASETS = ("widar", "xrf55", "xrf55_rfid")
DEFAULT_RATIOS = tuple(round(0.1 * i, 1) for i in range(1, 11))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widar-root", type=Path, required=True)
    parser.add_argument("--xrf55-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("runs/motivation/results.csv"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cpu"],
        help="One device per concurrent worker, e.g. cuda:0 cuda:1",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def root_for(dataset: str, args: argparse.Namespace) -> Path:
    if dataset == "widar":
        return args.widar_root
    if dataset in ("xrf55", "xrf55_rfid"):
        return args.xrf55_root
    raise ValueError(dataset)


def completed(output_csv: Path) -> set[tuple[str, str, float]]:
    if not output_csv.exists():
        return set()
    done = set()
    with output_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            done.add((row["Dataset"], row["Model"], float(row["Ratio"])))
    return done


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    done = completed(args.output_csv)

    queue = [
        (dataset, model, ratio)
        for dataset in args.datasets
        for model in args.models
        for ratio in args.ratios
        if (dataset, model, ratio) not in done
    ]
    print(f"{len(queue)} runs queued, {len(done)} already complete")

    running: list[tuple[subprocess.Popen, str]] = []
    while queue or running:
        for proc, label in list(running):
            if proc.poll() is not None:
                print(f"[done] {label} (exit {proc.returncode})")
                running.remove((proc, label))

        while queue and len(running) < len(args.devices):
            dataset, model, ratio = queue.pop(0)
            device = args.devices[len(running) % len(args.devices)]
            cmd = [
                sys.executable,
                str(WORKER),
                "--dataset", dataset,
                "--model", model,
                "--ratio", str(ratio),
                "--data-root", str(root_for(dataset, args)),
                "--output-csv", str(args.output_csv),
                "--device", device,
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
            ]
            label = f"{dataset}/{model}/{ratio:.1f}"
            print(f"[start] {label} on {device}")
            running.append((subprocess.Popen(cmd), label))

        time.sleep(2.0)

    print(f"All runs finished. Results: {args.output_csv}")


if __name__ == "__main__":
    main()
