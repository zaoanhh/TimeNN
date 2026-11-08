#!/usr/bin/env python3
"""Reproduce the feature-inversion waveform figure (paper Fig. 4a).

For each of three analytic signals (linear chirp, sine burst, square wave) a
frozen Time-NN block extracts ``s = Phi(x)``.  A new signal ``y`` is then
optimised so that ``Phi(y)`` matches ``s``; the figure compares ``x`` and ``y``
in a 0.1 s window.

The optimised waveforms are stored as 16-bit WAV files in
``data/results/feature_inversion/<signal>/{original,inverted}.wav``.  The plot
reads those files, so it does not need the model or a GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from _common import DATA_DIR, ensure_figure_dir
from timenn.plotting import NATURE_PALETTE, nature_style_context, resolve_figsize, save_figure

FS = 16_000  # sampling rate used in the inversion experiment
SIGNAL_NAMES = ("chirp", "sine", "square")
WINDOW_DURATION = 0.1  # seconds shown in the paper
START_SECONDS = 0.1  # start of the displayed window


def read_wav(path: Path) -> np.ndarray:
    """Read a PCM-16 WAV file and return float32 samples in [-1, 1]."""
    sample_rate, data = wavfile.read(path)
    if sample_rate != FS:
        raise ValueError(f"{path}: expected {FS} Hz, got {sample_rate} Hz")
    if data.ndim > 1:  # mono is stored as 1-D, but be defensive
        data = data[:, 0]
    return data.astype(np.float32) / 32768.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR / "feature_inversion",
        help="Directory containing <signal>/{original,inverted}.wav.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or ensure_figure_dir() / "waveform_comparison_combined.pdf"
    start = int(START_SECONDS * FS)
    n_samples = int(WINDOW_DURATION * FS)
    t = np.linspace(start / FS, (start + n_samples) / FS, n_samples)

    with nature_style_context():
        fig, axes = plt.subplots(3, 1, figsize=resolve_figsize("single"))
        for idx, name in enumerate(SIGNAL_NAMES):
            original = read_wav(args.data_dir / name / "original.wav")
            inverted = read_wav(args.data_dir / name / "inverted.wav")
            axes[idx].plot(
                t, original[start : start + n_samples],
                label="Original", color=NATURE_PALETTE["blue"],
            )
            axes[idx].plot(
                t, inverted[start : start + n_samples],
                label="Inverted", color=NATURE_PALETTE["vermillion"],
            )
            axes[idx].set_ylabel(name)
            axes[idx].set_ylim(-1.2, 1.2)
            axes[idx].set_xlim(t[0], t[-1])
            axes[idx].tick_params(labelbottom=False)

        axes[-1].tick_params(labelbottom=True)
        axes[-1].set_xlabel("Time (s)")
        fig.legend(
            ["Original", "Inverted"],
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 1.02),
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        save_figure(fig, output)

    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
