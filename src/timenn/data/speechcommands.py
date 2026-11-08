"""
SPEECHCOMMANDS dataset implementation using torchaudio.

This module provides:
- SpeechCommandsDataset: Dataset wrapper for torchaudio SPEECHCOMMANDS
- build_speechcommands_dataset: Factory function to create train/val/test datasets
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio


class NpySpeechCommandsDataset(Dataset):
    """Fast dataset reading from pre-split .npy files."""

    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)
        self.y = np.load(y_path)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), int(self.y[idx])


class SpeechCommandsDataset(Dataset):
    """Dataset wrapper for torchaudio SPEECHCOMMANDS dataset."""

    def __init__(
        self,
        root_dir: str,
        subset: str = "training",
        target_sample_rate: int = 16000,
        original_sample_rate: int = 16000,
        target_length: int = 16000,
        transform=None,
        exclude_labels: list = None,
        include_labels: list = None,
    ):
        """
        Args:
            root_dir: Root directory containing the SPEECHCOMMANDS dataset
            subset: "training", "validation", or "testing"
            target_sample_rate: Target sample rate for resampling
            original_sample_rate: Original sample rate of the dataset
            target_length: Target length for padding/truncating (in samples)
            transform: Optional transform to apply to the waveform
            exclude_labels: List of labels to exclude
            include_labels: List of labels to include (None for all)
        """
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=root_dir,
            download=False,  # Don't auto-download
            subset=subset,
        )
        self.target_sample_rate = target_sample_rate
        self.original_sample_rate = original_sample_rate
        self.target_length = target_length
        self.transform = transform

        # Create Resample object once for efficiency
        if self.original_sample_rate != self.target_sample_rate:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=self.original_sample_rate,
                new_freq=self.target_sample_rate,
            )
        else:
            self.resampler = None

        # Build label mapping with filtering applied
        self._build_label_mapping(exclude_labels, include_labels)

        # Filter indices to only include valid labels
        self._filter_indices()

    def _build_label_mapping(self, exclude_labels, include_labels):
        """Build label-to-index mapping from file paths (no WAV loading)."""
        all_labels = sorted(set(
            os.path.basename(os.path.dirname(p))
            for p in self.dataset._walker
        ))

        # Apply filtering
        if exclude_labels is not None:
            exclude_set = set(exclude_labels)
            all_labels = [l for l in all_labels if l not in exclude_set]
        elif include_labels is not None:
            include_set = set(include_labels)
            all_labels = [l for l in all_labels if l in include_set]

        self.labels = all_labels
        self.label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def _filter_indices(self):
        """Filter indices using file paths (no WAV loading)."""
        self.valid_indices = []
        for idx, path in enumerate(self.dataset._walker):
            label = os.path.basename(os.path.dirname(path))
            if label in self.label_to_idx:
                self.valid_indices.append(idx)

    def __len__(self):
        return len(self.valid_indices)

    def _resample(self, waveform: torch.Tensor) -> torch.Tensor:
        """Resample waveform to target sample rate."""
        if self.resampler is not None:
            waveform = self.resampler(waveform)
        return waveform

    def _pad_or_truncate(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pad or truncate waveform to target length."""
        current_length = waveform.shape[1]
        if current_length > self.target_length:
            # Truncate
            waveform = waveform[:, : self.target_length]
        elif current_length < self.target_length:
            # Pad with zeros
            pad_length = self.target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        return waveform

    def __getitem__(self, idx):
        """
        Returns:
            waveform: Tensor of shape [1, target_length] (after resampling and padding)
            label: Integer label index
        """
        # Map filtered index to actual dataset index
        actual_idx = self.valid_indices[idx]
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[actual_idx]

        # Resample if necessary
        waveform = self._resample(waveform)

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or truncate to target length
        waveform = self._pad_or_truncate(waveform)

        # Apply transform if any
        if self.transform is not None:
            waveform = self.transform(waveform)

        # Convert label to index
        label_idx = self.label_to_idx[label]

        return waveform, label_idx


def build_speechcommands_dataset(
    root_dir: str,
    target_sample_rate: int = 16000,
    original_sample_rate: int = 16000,
    target_length: int = 16000,
    exclude_labels: list = None,
    include_labels: list = None,
):
    """Build SPEECHCOMMANDS datasets. Uses pre-split .npy if available."""
    split_dir = os.path.join(root_dir, "SpeechCommands", "splits")
    paths = {
        name: (os.path.join(split_dir, f"X_{name}.npy"), os.path.join(split_dir, f"y_{name}.npy"))
        for name in ("train", "val", "test")
    }

    if all(os.path.exists(p[0]) for p in paths.values()):
        print("Loading pre-split data (mmap)...")
        train_ds = NpySpeechCommandsDataset(paths["train"][0], paths["train"][1])
        val_ds = NpySpeechCommandsDataset(paths["val"][0], paths["val"][1])
        test_ds = NpySpeechCommandsDataset(paths["test"][0], paths["test"][1])
        num_classes = len(np.unique(val_ds.y))
        print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
        return train_ds, val_ds, test_ds, num_classes

    # Fallback: load via torchaudio
    train_dataset = SpeechCommandsDataset(
        root_dir=root_dir,
        subset="training",
        target_sample_rate=target_sample_rate,
        original_sample_rate=original_sample_rate,
        target_length=target_length,
        exclude_labels=exclude_labels,
        include_labels=include_labels,
    )

    val_dataset = SpeechCommandsDataset(
        root_dir=root_dir,
        subset="validation",
        target_sample_rate=target_sample_rate,
        original_sample_rate=original_sample_rate,
        target_length=target_length,
        exclude_labels=exclude_labels,
        include_labels=include_labels,
    )

    test_dataset = SpeechCommandsDataset(
        root_dir=root_dir,
        subset="testing",
        target_sample_rate=target_sample_rate,
        original_sample_rate=original_sample_rate,
        target_length=target_length,
        exclude_labels=exclude_labels,
        include_labels=include_labels,
    )

    num_classes = len(train_dataset.labels)
    print(f"SPEECHCOMMANDS: Found {num_classes} classes: {train_dataset.labels}")

    return train_dataset, val_dataset, test_dataset, num_classes
