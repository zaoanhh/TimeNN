#!/usr/bin/env python3
"""Train and evaluate one Time-NN experiment from ``configs/lms_cnn_capture24``.

This is the training script used for the ``with_tau.csv`` sweep.  It trains one
``model × dataset`` configuration at a time; new runs write to ``runs/`` by
default so existing result tables are never overwritten.  It composes the packaged
Hydra configuration ``experiment/lms_cnn_capture24.yaml``, trains the selected
model, and then evaluates it on the clean test set and on five
independently time-warped copies of the test set for each ``tau`` in
``{0.1, ..., 0.9}``.

Usage
-----
::

    uv run python scripts/train.py --dataset capture24 --model lms_cnn \
        --output-path runs/with_tau.csv

The datasets themselves are not distributed with this repository; set the
``root_dir`` / ``datadir`` entries in ``configs/lms_cnn_capture24/dataset/*.yaml`` to
your local copies first.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make ``import timenn`` work when the package is not installed (e.g. running
# from a source checkout without ``uv sync``).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from timenn.data.factory import build_dataset, get_available_datasets  # noqa: E402
from timenn.models.factory import build_model, get_available_models  # noqa: E402
from timenn.transforms.time_warp import TimeWarpTransform  # noqa: E402

# Distortion strengths written to the result CSV (0.0 is the clean test set).
WARP_STRENGTHS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
WARP_EVAL_FACTOR = 5


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return 100.0 * (predictions == targets).sum().item() / len(targets)


def coverage_weight(target: float, epoch: int, warmup_epochs: int) -> float:
    """Linearly warm up the coverage regulariser from 0 to ``target``."""
    if target <= 0.0 or warmup_epochs <= 0:
        return target
    if warmup_epochs == 1:
        return target
    progress = max(0.0, min(1.0, epoch / float(warmup_epochs - 1)))
    return target * progress


def train_epoch(model, loader, criterion, optimizer, device, lam: float):
    """One supervised epoch, adding the filter-bank coverage loss when present."""
    model.train()
    total_loss = total_cls = total_cov = 0.0
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for batch in tqdm(loader, desc="train", leave=False, disable=not sys.stdout.isatty()):
        x, y = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        logits = model(x)
        cls_loss = criterion(logits, y)

        if hasattr(model, "get_total_coverage_loss"):
            cov_loss = model.get_total_coverage_loss()
            loss = cls_loss + lam * cov_loss
        else:
            cov_loss = torch.zeros((), device=device)
            loss = cls_loss

        loss.backward()
        optimizer.step()

        n = x.size(0)
        total_loss += loss.item() * n
        total_cls += cls_loss.item() * n
        total_cov += float(cov_loss) * n
        preds.append(logits.argmax(1).detach().cpu())
        targets.append(y.detach().cpu())

    n_total = sum(t.numel() for t in targets)
    preds_t = torch.cat(preds)
    targets_t = torch.cat(targets)
    return (
        total_loss / n_total,
        total_cls / n_total,
        total_cov / n_total,
        accuracy(preds_t, targets_t),
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device, lam: float = 0.0):
    """Clean evaluation (no warp)."""
    model.eval()
    total_loss = total_cls = total_cov = 0.0
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="eval", leave=False, disable=not sys.stdout.isatty()):
        x, y = batch[0].to(device), batch[1].to(device)
        logits = model(x)
        cls_loss = criterion(logits, y)
        if hasattr(model, "get_total_coverage_loss"):
            cov_loss = model.get_total_coverage_loss()
            loss = cls_loss + lam * cov_loss
        else:
            cov_loss = torch.zeros((), device=device)
            loss = cls_loss
        n = x.size(0)
        total_loss += loss.item() * n
        total_cls += cls_loss.item() * n
        total_cov += float(cov_loss) * n
        preds.append(logits.argmax(1).cpu())
        targets.append(y.cpu())

    n_total = sum(t.numel() for t in targets)
    return (
        total_loss / n_total,
        total_cls / n_total,
        total_cov / n_total,
        accuracy(torch.cat(preds), torch.cat(targets)),
    )


@torch.no_grad()
def evaluate_warped(
    model,
    loader,
    device,
    warp_strength: float,
    factor: int = WARP_EVAL_FACTOR,
    chunk_size: int = 16,
):
    """Accuracy under ``factor`` independent time warps per test sample."""
    model.eval()
    warp = TimeWarpTransform(
        warp_strength=warp_strength,
        shift_ratio=0.0,
        num_knots=6,
        same_warp=False,
    )
    correct = total = 0
    for batch in tqdm(
        loader,
        desc=f"warp={warp_strength:.1f}",
        leave=False,
        disable=not sys.stdout.isatty(),
    ):
        x, y = batch[0].to(device), batch[1].to(device)
        b, c, t = x.shape
        x_rep = x.unsqueeze(1).repeat(1, factor, 1, 1).view(b * factor, c, t)
        x_aug = warp(x_rep)
        y_rep = y.repeat_interleave(factor)
        logits = torch.cat(
            [model(x_aug[i : i + chunk_size]) for i in range(0, x_aug.shape[0], chunk_size)]
        )
        correct += (logits.argmax(1) == y_rep).sum().item()
        total += y_rep.numel()
    return correct / total


def build_optimizer(model, cfg: DictConfig):
    """Create Adam with a larger LR for the learnable filter bank.

    Gradients reaching the analytic filter parameters are diluted by the
    complex modulus and the pooling layers, so the paper uses a 10x learning
    rate for that parameter group.
    """
    base_lr = float(cfg.train.lr)
    weight_decay = float(cfg.train.get("weight_decay", 1e-4))
    bank_factor = float(cfg.train.get("filter_bank_lr_factor", 10.0))

    filter_bank, mixing, base = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower = name.lower()
        if "filter_bank" in lower:
            filter_bank.append(param)
        elif "mixing" in lower or "lms" in lower or "scattering" in lower:
            mixing.append(param)
        else:
            base.append(param)

    groups = [{"params": base, "lr": base_lr}]
    if mixing:
        groups.append({"params": mixing, "lr": base_lr * bank_factor})
    if filter_bank:
        groups.append({"params": filter_bank, "lr": base_lr * bank_factor})

    print(
        f"[optimizer] base={sum(p.numel() for p in base):,} "
        f"mixing={sum(p.numel() for p in mixing):,} "
        f"filter_bank={sum(p.numel() for p in filter_bank):,}"
    )
    return optim.Adam(groups, weight_decay=weight_decay)


def train_one(
    dataset_name: str,
    model_name: str,
    output_path: Path,
    device_str: str,
    config_dir: Path,
    epochs_override: Optional[int] = None,
) -> bool:
    experiment_name = f"{model_name}_{dataset_name}"
    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        cfg = compose(config_name=f"experiment/{experiment_name}")

    print(OmegaConf.to_yaml(cfg))
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.get("seed", 42)))
    print(f"[device] {device}")

    train_ds, val_ds, test_ds, num_classes = build_dataset(dataset_name, cfg.dataset)
    sample = train_ds[0][0]
    in_channels, input_length = sample.shape[0], sample.shape[1]
    print(
        f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"classes={num_classes} channels={in_channels} length={input_length}"
    )

    loader_kwargs = dict(
        batch_size=int(cfg.dataset.dataloader.batch_size),
        num_workers=int(cfg.train.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(
        train_ds, shuffle=True, drop_last=bool(cfg.dataset.dataloader.drop_last), **loader_kwargs
    )
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)

    model = build_model(
        model_name,
        cfg.model,
        c_in=in_channels,
        input_length=input_length,
        num_classes=num_classes,
    ).to(device)
    print(f"[model] {sum(p.numel() for p in model.parameters()):,} parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10
    )

    epochs = int(epochs_override or cfg.train.epochs)
    patience = int(cfg.train.early_stop_patience)
    lam = float(cfg.model.get("coverage_lambda", 0.1))
    warmup = int(cfg.train.get("coverage_warmup_epochs", 0))
    eval_chunk = int(cfg.train.get("eval_chunk_size", 16))

    ckpt_dir = (
        PROJECT_ROOT
        / "runs"
        / f"{experiment_name}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pth"

    best_acc, best_epoch, stale = 0.0, 0, 0
    for epoch in range(epochs):
        current_lam = coverage_weight(lam, epoch, warmup)
        tr_loss, tr_cls, tr_cov, tr_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, current_lam
        )
        va_loss, va_cls, va_cov, va_acc = evaluate(
            model, val_loader, criterion, device, current_lam
        )
        scheduler.step(va_loss)
        print(
            f"epoch {epoch + 1}/{epochs} lam={current_lam:.3f} | "
            f"train loss={tr_loss:.4f} acc={tr_acc:.2f}% | "
            f"val loss={va_loss:.4f} acc={va_acc:.2f}%"
        )

        if va_acc > best_acc:
            best_acc, best_epoch, stale = va_acc, epoch + 1, 0
            torch.save(model.state_dict(), best_path)
            print(f"  -> best ({va_acc:.2f}%)")
        else:
            stale += 1
            if stale >= patience:
                print(f"early stopping at epoch {epoch + 1}")
                break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"[checkpoint] {best_path} (epoch {best_epoch})")

    test_loss, test_cls, test_cov, test_acc = evaluate(
        model, test_loader, criterion, device, lam
    )
    print(f"[test] clean accuracy={test_acc:.2f}%")

    row: dict[str, object] = {
        "dataset": dataset_name,
        "model": model_name,
        "best_model_path": str(best_path.resolve()),
        "acc_warp_0.0": repr(test_acc / 100.0),
    }
    for warp in WARP_STRENGTHS[1:]:
        row[f"acc_warp_{warp:.1f}"] = repr(
            evaluate_warped(
                model,
                test_loader,
                device,
                warp,
                factor=WARP_EVAL_FACTOR,
                chunk_size=eval_chunk,
            )
        )
        print(f"  tau={warp:.1f}: {float(row[f'acc_warp_{warp:.1f}']) * 100:.2f}%")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[saved] {output_path}")
    GlobalHydra.instance().clear()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. capture24")
    parser.add_argument("--model", required=True, help="Model name, e.g. lms_cnn")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "runs" / "with_tau.csv",
    )
    parser.add_argument("--device", default="cuda:0", help="e.g. cuda:0 or cpu")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=PROJECT_ROOT / "configs" / "lms_cnn_capture24",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    args = parser.parse_args()

    if args.dataset not in get_available_datasets():
        parser.error(
            f"unknown dataset {args.dataset!r}; available: {get_available_datasets()}"
        )
    if args.model not in get_available_models():
        parser.error(f"unknown model {args.model!r}; available: {get_available_models()}")

    experiment_config = (
        args.config_dir / "experiment" / f"{args.model}_{args.dataset}.yaml"
    )
    if not experiment_config.exists():
        available = sorted(
            path.stem
            for path in (args.config_dir / "experiment").glob("*.yaml")
            if not path.stem.startswith("_")
        )
        parser.error(
            f"no config for {args.model}/{args.dataset}: expected "
            f"{experiment_config}. Available experiments: {available}"
        )

    ok = train_one(
        dataset_name=args.dataset,
        model_name=args.model,
        output_path=args.output_path,
        device_str=args.device,
        config_dir=args.config_dir,
        epochs_override=args.epochs,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
