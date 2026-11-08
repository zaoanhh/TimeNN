"""
Medley-solos-DB dataset implementation for musical instrument recognition.

This module provides:
- MedleySolosDataset: PyTorch Dataset wrapper for Medley-solos-DB
- build_medley_dataset: Factory function to create train/val/test datasets

Dataset Description:
    Medley-solos-DB is a cross-collection dataset for automatic musical instrument
    recognition in solo recordings. It consists of 3-second audio clips from
    MedleyDB (training) and solosDB (test).

    Instrument Classes (8 total):
    0. clarinet
    1. distorted electric guitar
    2. female singer
    3. flute
    4. piano
    5. tenor saxophone
    6. trumpet
    7. violin

Audio Specifications:
    - Sample Rate: 44.1 kHz
    - Channels: Mono (1)
    - Bit Depth: 32-bit float
    - Duration: 2972 ms (65536 samples)

Reference:
    V. Lostanlen, C.E. Cella. "Deep convolutional networks on the pitch spiral
    for musical instrument recognition." Proceedings of ISMIR, 2016.
"""

import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.io import wavfile


# Instrument class mapping as per Medley-solos-DB specification
INSTRUMENT_CLASSES = [
    "clarinet",
    "distorted electric guitar",
    "female singer",
    "flute",
    "piano",
    "tenor saxophone",
    "trumpet",
    "violin",
]


class MedleySolosDataset(Dataset):
    """PyTorch Dataset wrapper for Medley-solos-DB audio clips."""

    def __init__(
        self,
        data_dir: str,
        metadata_path: str,
        subset: str = "training",
        target_sample_rate: int = 44100,
        original_sample_rate: int = 44100,
        target_length: int = 65536,
        transform=None,
    ):
        """
        Args:
            data_dir: Root directory containing audio files (data/ subfolder)
            metadata_path: Path to Medley-solos-DB_metadata.csv
            subset: "training", "validation", or "testing"
            target_sample_rate: Target sample rate for resampling
            original_sample_rate: Original sample rate of the dataset
            target_length: Target length for padding/truncating (in samples)
            transform: Optional transform to apply to the waveform
        """
        self.data_dir = data_dir
        self.subset = subset
        self.target_sample_rate = target_sample_rate
        self.original_sample_rate = original_sample_rate
        self.target_length = target_length
        self.transform = transform

        # Load metadata and filter by subset
        self.samples = self._load_metadata(metadata_path, subset)

        # Build instrument ID to label mapping
        self.idx_to_label = {i: label for i, label in enumerate(INSTRUMENT_CLASSES)}
        self.label_to_idx = {label: idx for idx, label in self.idx_to_label.items()}

    def _load_metadata(self, metadata_path: str, subset: str) -> list:
        """Load metadata from CSV and filter by subset."""
        samples = []
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["subset"] == subset:
                        samples.append(
                            {
                                "instrument": row["instrument"],
                                "instrument_id": int(row["instrument_id"]),
                                "song_id": row["song_id"],
                                "uuid4": row["uuid4"],
                            }
                        )
        except FileNotFoundError:
            print(f"Warning: Metadata file not found at {metadata_path}")
            return []
        except Exception as e:
            print(f"Error reading metadata: {e}")
            return []

        print(f"Loaded {len(samples)} samples for {subset} subset")
        return samples

    def _load_audio(self, filepath: str) -> torch.Tensor:
        """Load audio file using scipy.io.wavfile."""
        try:
            # Read WAV file using scipy
            sample_rate, audio_data = wavfile.read(filepath)

            # Convert to float tensor
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Handle different array shapes - ensure (channels, time)
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(1, -1)
            elif audio_data.ndim == 2:
                dims = audio_data.shape
                if dims[0] > dims[1]:  # type: ignore[index]
                    audio_data = audio_data.T
            else:
                raise ValueError(f"Unexpected audio shape: {audio_data.shape}")
            
            # audio_data = (audio_data - np.mean(audio_data, axis=-1, keepdims=True)) / (np.mean(audio_data, axis=-1, keepdims=True) + 1e-9)  # DC offset removal
            audio_data = (audio_data - np.mean(audio_data, axis=-1, keepdims=True))  # DC offset removal

            # Convert to tensor
            waveform = torch.from_numpy(audio_data).float()
            return waveform

        except Exception as e:
            print(f"Error loading audio {filepath}: {e}")
            # Return zeros on error
            return torch.zeros(1, self.target_length, dtype=torch.float32)

    def _resample(self, waveform: torch.Tensor) -> torch.Tensor:
        """Resample waveform to target sample rate."""
        if self.original_sample_rate != self.target_sample_rate:
            # Calculate resampling ratio
            ratio = self.target_sample_rate / self.original_sample_rate
            new_length = int(waveform.shape[-1] * ratio)
            waveform = torch.nn.functional.interpolate(
                waveform.unsqueeze(0),
                size=new_length,
                mode="linear",
                align_corners=False,
            ).squeeze(0)
        return waveform

    def _pad_or_truncate(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pad or truncate waveform to target length."""
        current_length = waveform.shape[-1]
        if current_length > self.target_length:
            # Truncate
            waveform = waveform[:, : self.target_length]
        elif current_length < self.target_length:
            # Pad with zeros
            pad_length = self.target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        return waveform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            waveform: Tensor of shape [1, target_length]
            label: Integer label index
        """
        sample = self.samples[idx]

        # Construct audio file path
        # Filename format: Medley-solos-DB_{subset}-{instrument_id}_{uuid4}.wav
        subset_prefix = "training" if self.subset == "training" else self.subset
        filename = f"Medley-solos-DB_{subset_prefix}-{sample['instrument_id']}_{sample['uuid4']}.wav"
        filepath = os.path.join(self.data_dir, "data", filename)

        # Load audio
        waveform = self._load_audio(filepath)

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

        # Get label index
        label_idx = sample["instrument_id"]

        return waveform, label_idx


def build_medley_dataset(
    data_dir: str,
    target_sample_rate: int = 44100,
    original_sample_rate: int = 44100,
    target_length: int = 65536,
) -> tuple:
    """
    Build Medley-solos-DB datasets for training, validation, and testing.

    Args:
        data_dir: Root directory containing Medley-solos-DB (with data/ and CSV)
        target_sample_rate: Target sample rate for resampling
        original_sample_rate: Original sample rate of the dataset
        target_length: Target length for padding/truncating (in samples)

    Returns:
        train_dataset, val_dataset, test_dataset, num_classes
    """
    metadata_path = os.path.join(data_dir, "Medley-solos-DB_metadata.csv")

    # Verify data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Create datasets for each subset
    train_dataset = MedleySolosDataset(
        data_dir=data_dir,
        metadata_path=metadata_path,
        subset="training",
        target_sample_rate=target_sample_rate,
        original_sample_rate=original_sample_rate,
        target_length=target_length,
    )

    val_dataset = MedleySolosDataset(
        data_dir=data_dir,
        metadata_path=metadata_path,
        subset="validation",
        target_sample_rate=target_sample_rate,
        original_sample_rate=original_sample_rate,
        target_length=target_length,
    )

    test_dataset = MedleySolosDataset(
        data_dir=data_dir,
        metadata_path=metadata_path,
        subset="test",
        target_sample_rate=target_sample_rate,
        original_sample_rate=original_sample_rate,
        target_length=target_length,
    )

    num_classes = len(INSTRUMENT_CLASSES)
    print(f"Medley-solos-DB: {num_classes} instrument classes")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Testing samples: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, num_classes
