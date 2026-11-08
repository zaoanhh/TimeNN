#!/usr/bin/env python3
"""Visualise feature-map stability under temporal distortion (paper Fig. 2).

A synthetic multi-component signal is passed through six feature extractors
twice: once unchanged and once after a random smooth time warp.  The figure
shows the element-wise difference ``f(tau(x)) - f(x)`` for every extractor.
A stable extractor (Time-NN) should produce a nearly uniform (white) map.

The extractors are *not trained* for this visualisation; the point is the
structural sensitivity of the operator itself.  Seeds are fixed so the figure
is reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from _common import PROJECT_ROOT, ensure_figure_dir
from timenn.models.inception import InceptionTime
from timenn.models.lms_net import FilterConfig, LMSBlock
from timenn.models.simple_cnn import SimpleCNN1D
from timenn.plotting import NATURE_PALETTE, nature_style_context, resolve_figsize, save_figure
from timenn.transforms.time_warp import apply_time_deformation

MODEL_NAMES = ["MLP", "CNN", "LSTM", "Transformer", "Inception", "TIME-NN"]
BOUNDARY_TRIM = 8


def generate_compound_signal(length: int = 512) -> torch.Tensor:
    """Low-frequency tone + mid tone + three Gaussian-windowed bursts."""
    t = np.linspace(0, 1, length)
    s_low = np.sin(2 * np.pi * 2.5 * t)
    s_mid = 0.6 * np.sin(2 * np.pi * 10 * t)
    s_burst1 = 0.5 * np.sin(2 * np.pi * 30 * t) * np.exp(-((t - 0.25) / 0.06) ** 2)
    s_burst2 = 0.5 * np.sin(2 * np.pi * 30 * t) * np.exp(-((t - 0.65) / 0.06) ** 2)
    s_burst3 = 0.4 * np.sin(2 * np.pi * 20 * t) * np.exp(-((t - 0.45) / 0.08) ** 2)
    env = 1.0 + 0.25 * np.sin(2 * np.pi * 1.5 * t)
    signal = (s_low + s_mid + s_burst1 + s_burst2 + s_burst3) * env
    signal = signal / np.max(np.abs(signal))
    return torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


class MLPFeatureExtractor(nn.Module):
    """Point-wise 1x1 convolutions: no temporal receptive field."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, 1), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 1), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNFeatureExtractor(nn.Module):
    """Small convolutional encoder."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.backbone = SimpleCNN1D(
            in_channels=1, layers=(32, hidden, hidden), kernel_size=5, stride=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class LSTMFeatureExtractor(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x.permute(0, 2, 1))
        return out.permute(0, 2, 1)


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :, : x.size(2)]


class TransformerFeatureExtractor(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.proj = nn.Conv1d(1, d_model, kernel_size=4, stride=4)
        self.pos = _PositionalEncoding(d_model, max_len=2000)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=128,
            dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.pos(x)
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        return x.permute(0, 2, 1)


class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = InceptionTime(
            in_channels=1, num_classes=10, num_blocks=2,
            num_modules_per_block=2, out_channels=16, kss=[10, 20, 40],
            bottleneck_channels=16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.model.blocks:
            x = block(x)
        return x


class LMSFeatureExtractor(nn.Module):
    """One Time-NN block; returns the stable macro-scale feature stream."""

    def __init__(self):
        super().__init__()
        filter_cfg = FilterConfig(
            filter_type="morlet",
            scale_type="linear",
            num_scales=8,
            min_sigma=0.05,
            f_min=0.005,
            f_max=0.25,
            kernel_length=64,
            reduce_scales=True,
            pool_type="learnable",
        )
        self.block = LMSBlock(
            c_in=1,
            filter_config=filter_cfg,
            intra_out_per_group=4,
            inter_out=64,
            blur_stride=2,
            blur_kernel_size=5,
            use_log=True,
            mixing_activation="gelu",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, s_out = self.block(x)
        return s_out


def build_extractors() -> dict[str, nn.Module]:
    return {
        "MLP": MLPFeatureExtractor(hidden=32),
        "CNN": CNNFeatureExtractor(hidden=64),
        "LSTM": LSTMFeatureExtractor(hidden_size=32),
        "Transformer": TransformerFeatureExtractor(d_model=64),
        "Inception": InceptionFeatureExtractor(),
        "TIME-NN": LMSFeatureExtractor(),
    }


def _align_time(src: torch.Tensor, target_len: int) -> torch.Tensor:
    """Linearly resize a feature map to ``target_len`` along time."""
    if src.shape[-1] == target_len:
        return src
    return F.interpolate(
        src.unsqueeze(0), size=target_len, mode="linear", align_corners=True
    ).squeeze(0)


def _trim(t: torch.Tensor) -> torch.Tensor:
    """Drop boundary samples affected by padding."""
    if t.shape[-1] <= 2 * BOUNDARY_TRIM:
        return t
    return t[..., BOUNDARY_TRIM:-BOUNDARY_TRIM]


def compute_feature_maps(seed: int = 123):
    """Run the six extractors on the clean and warped signal.

    The signal itself is deterministic; the seed is consumed by the warp and by
    the following model initialisation.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    x_orig = generate_compound_signal(512)
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_warp = apply_time_deformation(
        x_orig.clone(), warp_strength=0.22, num_knots=5, shift_ratio=0.05
    )

    extractors = build_extractors()
    for module in extractors.values():
        module.eval()

    feat_maps: dict[str, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for name in MODEL_NAMES:
            module = extractors[name]
            feat_maps[name] = {
                "orig": module(x_orig.clone()).squeeze(0),
                "warp": module(x_warp.clone()).squeeze(0),
            }
            print(
                f"  {name:>12s}: orig {list(feat_maps[name]['orig'].shape)}, "
                f"warp {list(feat_maps[name]['warp'].shape)}"
            )
    return x_orig.squeeze().numpy(), x_warp.squeeze().numpy(), feat_maps


def save_feature_maps(path: Path, x_orig, x_warp, feat_maps) -> None:
    """Save the exact arrays used for the paper figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"x_orig": x_orig, "x_warp": x_warp}
    for name in MODEL_NAMES:
        arrays[f"orig::{name}"] = feat_maps[name]["orig"].numpy()
        arrays[f"warp::{name}"] = feat_maps[name]["warp"].numpy()
    np.savez_compressed(path, **arrays)
    print(f"[saved] {path}")


def load_feature_maps(path: Path):
    """Load a previously saved feature-map bundle."""
    data = np.load(path)
    feat_maps = {
        name: {
            "orig": torch.from_numpy(data[f"orig::{name}"]),
            "warp": torch.from_numpy(data[f"warp::{name}"]),
        }
        for name in MODEL_NAMES
    }
    return data["x_orig"], data["x_warp"], feat_maps


def plot(x_orig, x_warp, feat_maps, output: Path) -> None:
    """Draw the one-row difference-map figure."""
    t = np.linspace(0, 1, 512)

    with nature_style_context(font_size=6.0, use_grid=False):
        n_cols = 1 + len(MODEL_NAMES)
        fig = plt.figure(figsize=resolve_figsize((9.0, 1.6)))
        gs = GridSpec(
            2, n_cols, figure=fig,
            height_ratios=[1, 0.08],
            hspace=0.15, wspace=0.35,
            left=0.04, right=0.98, bottom=0.18, top=0.92,
        )

        ax_sig = fig.add_subplot(gs[0, 0])
        ax_sig.plot(
            t, x_orig, linewidth=0.6,
            color=NATURE_PALETTE["blue"], label="Original",
        )
        ax_sig.plot(
            t, x_warp, linewidth=0.6,
            color=NATURE_PALETTE["vermillion"], label="Warped",
        )
        ax_sig.set_xlim(0, 1)
        ax_sig.set_ylim(-1.15, 1.15)
        ax_sig.set_yticks([-1, 0, 1.2])
        ax_sig.set_xlabel("Time", fontsize=5.5)
        ax_sig.legend(
            fontsize=6, loc="lower center", frameon=False,
            handlelength=1.0, borderpad=0.2, labelspacing=0.2,
            ncol=2, bbox_to_anchor=(0.6, 0.9),
        )
        ax_sig.tick_params(labelsize=5)

        for col_idx, name in enumerate(MODEL_NAMES):
            ax = fig.add_subplot(gs[0, col_idx + 1])
            orig = _trim(feat_maps[name]["orig"])
            warp = _trim(feat_maps[name]["warp"])
            diff = (_align_time(warp, orig.shape[-1]) - orig).numpy()
            v = max(abs(diff.min()), abs(diff.max()), 1e-8)
            ax.imshow(
                diff, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v,
                interpolation="bilinear",
            )
            ax.set_title(name, fontsize=6.5, fontweight="bold", pad=4)
            ax.tick_params(labelbottom=True, labelleft=False, labelsize=4.5)

        cax = fig.add_subplot(gs[1, :])
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=Normalize(vmin=-1, vmax=1))
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label("Δ (warped − original)", fontsize=5.5, labelpad=2)
        cbar.ax.tick_params(labelsize=4.5)
        cbar.outline.set_linewidth(0.4)

        save_figure(fig, output)

    print(f"[saved] {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output PDF (default: figures/stability_onediff.pdf).",
    )
    parser.add_argument(
        "--data", type=Path, default=None,
        help="Saved feature-map .npz (default: data/results/stability_feature_maps.npz).",
    )
    parser.add_argument(
        "--recompute", action="store_true",
        help="Recompute the maps from the model definitions instead of loading the saved arrays.",
    )
    parser.add_argument(
        "--save-data", action="store_true",
        help="Write the recomputed arrays to --data.",
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    output = args.output or ensure_figure_dir() / "stability_onediff.pdf"
    data_path = args.data or (PROJECT_ROOT / "data" / "results" / "stability_feature_maps.npz")

    if args.recompute or not data_path.exists():
        if not data_path.exists():
            print(f"[info] {data_path} not found; recomputing feature maps")
        x_orig, x_warp, feat_maps = compute_feature_maps(args.seed)
        if args.save_data:
            save_feature_maps(data_path, x_orig, x_warp, feat_maps)
    else:
        print(f"[info] loading saved feature maps from {data_path}")
        x_orig, x_warp, feat_maps = load_feature_maps(data_path)

    plot(x_orig, x_warp, feat_maps, output)


if __name__ == "__main__":
    main()
