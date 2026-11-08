#!/usr/bin/env python3
"""Train one baseline for the motivation experiment (paper Fig. 3).

This is the worker launched by ``run_motivation_experiment.py``.  It trains a
single ``(dataset, model, training-ratio)`` combination, evaluates the clean
test set and a time-warped/shifted test set, and appends one CSV row.

The seven baselines are the ones compared in the paper: ResNet-18, ResNet-34,
CNN, Inception, Transformer, LSTM and GRU.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from timenn.data.widar import build_widar_dataset
from timenn.data.xrf55_rfid import build_xrf55_dataset as build_xrf55_rfid_dataset
from timenn.data.xrf55_wifi_random import build_xrf55_dataset as build_xrf55_wifi_dataset
from timenn.models.inception import InceptionTime
from timenn.models.resnet1d import resnet18
from timenn.models.transformer import transformer_s
from timenn.models.variants import SimpleCNN1D, SimpleRNN
from timenn.transforms.augmentation import time_warp

MODELS = ("resnet18", "resnet34", "cnn", "inception", "transformer", "lstm", "gru")
DATASETS = ("widar", "xrf55", "xrf55_rfid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--ratio", type=float, required=True, help="training ratio")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="root directory of the benchmark (Widar/XRF55)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("runs/motivation/results.csv"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("runs/motivation/checkpoints"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def to_magnitude(dataset, num_workers: int) -> TensorDataset:
    """Materialise a dataset and convert complex CSI to magnitude."""
    xs, ys = [], []
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=num_workers)
    for batch in loader:
        x, y = batch[0], batch[1]
        xs.append(x.abs() if x.is_complex() else x)
        ys.append(y)
    return TensorDataset(torch.cat(xs).float(), torch.cat(ys).long())


def load_dataset(name: str, root: Path, ratio: float, num_workers: int):
    """Build a proportionally split train/val/test set for one benchmark."""
    split = {"train_ratio": ratio, "test_size": 0.2, "val_size": 0.1}

    if name == "widar":
        result = build_widar_dataset(
            root_dir=str(root),
            subcarrier_ids=list(range(30)),
            antenna_ids=[0],
            target_sample_rate=200,
            split_args=split,
        )
        if len(result) == 3:
            raise RuntimeError(f"failed to load Widar from {root}")
        train, val, test, classes = result
        return (
            to_magnitude(train, num_workers),
            to_magnitude(val, num_workers),
            to_magnitude(test, num_workers),
            classes,
        )

    if name == "xrf55":
        train, val, test, classes = build_xrf55_wifi_dataset(
            root_dir=str(root),
            split_args={"train_ratio": ratio, "test_ratio": 0.2, "val_ratio": 0.1},
            include_types=None,
        )
        return train, val, test, classes

    if name == "xrf55_rfid":
        return build_xrf55_rfid_dataset(
            root_dir=str(root),
            split_args=split,
            include_types=None,
        )

    raise ValueError(name)


def build_baseline(model_name: str, in_channels: int, num_classes: int) -> nn.Module:
    if model_name == "cnn":
        return SimpleCNN1D(in_channels, num_classes)
    if model_name == "resnet18":
        return resnet18(
            in_channels=in_channels, num_classes=num_classes,
            layers=[2, 2, 2, 2], base_filters=64, dropout=0.0,
        )
    if model_name == "resnet34":
        return resnet18(
            in_channels=in_channels, num_classes=num_classes,
            layers=[3, 4, 6, 3], base_filters=64, dropout=0.0,
        )
    if model_name == "transformer":
        return transformer_s(in_channels=in_channels, num_classes=num_classes, d_model=64)
    if model_name == "inception":
        return InceptionTime(in_channels, num_classes, num_blocks=3)
    if model_name == "lstm":
        return SimpleRNN(in_channels, num_classes, rnn_type="LSTM", hidden_size=128)
    if model_name == "gru":
        return SimpleRNN(in_channels, num_classes, rnn_type="GRU", hidden_size=128)
    raise ValueError(model_name)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = correct = total = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def shift_with_noise(x: np.ndarray, ratio: float = 0.2, noise_std: float = 0.01):
    """Randomly translate a ``[T, C]`` sample and fill the gap with noise."""
    t, c = x.shape
    shift = np.random.randint(-int(t * ratio), int(t * ratio))
    if shift == 0:
        return x
    out = np.zeros_like(x)
    if shift > 0:
        out[shift:] = x[:-shift]
        out[:shift] = np.random.randn(shift, c) * noise_std
    else:
        s = -shift
        out[:-s] = x[s:]
        out[-s:] = np.random.randn(s, c) * noise_std
    return out


@torch.no_grad()
def evaluate_warped(model, loader, device, factor: int = 5):
    """Evaluate on ``factor`` warped + translated copies of every test sample."""
    model.eval()
    correct = total = 0
    for batch in loader:
        x, y = batch[0], batch[1].to(device)
        augmented = []
        for sample in x.cpu().numpy():
            sample_tc = sample.T
            for _ in range(factor):
                warped = time_warp(sample_tc, sigma=0.4, knots=4)
                augmented.append(torch.from_numpy(shift_with_noise(warped).T))
        aug_x = torch.stack(augmented).to(device).float()
        aug_y = y.repeat_interleave(factor)
        pred = model(aug_x).argmax(1)
        correct += (pred == aug_y).sum().item()
        total += aug_y.numel()
    return correct / total


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    train_ds, val_ds, test_ds, num_classes = load_dataset(
        args.dataset, args.data_root, args.ratio, args.num_workers
    )
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} classes={num_classes}")

    sample_x = train_ds[0][0]
    model = build_baseline(args.model, sample_x.shape[0], num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=10
    )
    criterion = nn.CrossEntropyLoss()

    loader_kwargs = dict(
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.checkpoint_dir / f"model_{args.dataset}_{args.model}_{args.ratio:.1f}.pth"

    best_acc, stale = 0.0, 0
    for epoch in range(args.epochs):
        loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_acc)
        if val_acc > best_acc:
            best_acc, stale = val_acc, 0
            torch.save(model.state_dict(), ckpt)
        else:
            stale += 1
            if stale >= 20:
                print(f"[early-stop] epoch {epoch}")
                break
        if epoch % 10 == 0:
            print(f"epoch {epoch}: train={train_acc:.4f} val={val_acc:.4f}")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    test_acc = evaluate(model, test_loader, device)
    warped_acc = evaluate_warped(model, test_loader, device)
    print(f"[result] {args.dataset} {args.model} ratio={args.ratio}: "
          f"clean={test_acc:.4f} warped={warped_acc:.4f}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output_csv.exists()
    with args.output_csv.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                ["Dataset", "Ratio", "NumSamples", "Model", "TestAcc", "AugTestAcc", "ModelPath"]
            )
        writer.writerow(
            [args.dataset, args.ratio, len(train_ds), args.model,
             f"{test_acc:.5f}", f"{warped_acc:.5f}", str(ckpt)]
        )


if __name__ == "__main__":
    main()
