"""
TIMIT Phone Segment Classification Dataset

This module provides a PyTorch Dataset class for TIMIT phoneme segment classification.
It implements centered windowing around phoneme boundaries with fixed-length audio windows.
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Tuple, List, Optional, Dict, Union
import torchaudio


# =============================================================================
# Speaker Lists for Data Splitting
# =============================================================================

# Core Test Set: 24 speakers for standard evaluation (from TIMIT documentation Table 3)
# 2 male + 1 female from each of 8 dialect regions
CORE_TEST_SPEAKERS = [
    # DR1: New England
    "DAB0",
    "WBT0",
    "ELC0",
    # DR2: Northern
    "TAS1",
    "WEW0",
    "PAS0",
    # DR3: North Midland
    "JMP0",
    "LNT0",
    "PKT0",
    # DR4: South Midland
    "LLL0",
    "TLS0",
    "JLM0",
    # DR5: Southern
    "BPM0",
    "KLT0",
    "NLP0",
    # DR6: New York City
    "CMJ0",
    "JDH0",
    "MGD0",
    # DR7: Western
    "GRT0",
    "NJM0",
    "DHC0",
    # DR8: Army Brat
    "JLN0",
    "PAM0",
    "MLD0",
]

# Development Set: 50 speakers for validation
DEV_50_SPEAKERS = [
    "FCMH0",
    "FVKB0",
    "FVMW0",
    "MAJC0",
    "MBDG0",
    "MBWM0",
    "MCSH0",
    "MWBT0",
    "FCLT0",
    "FJEM0",
    "FJRE0",
    "MBEO0",
    "MBJK0",
    "MCCS0",
    "MCED0",
    "MCTW0",
    "FALK0",
    "FAJW0",
    "MDLS0",
    "MDSS0",
    "METW0",
    "MGWT0",
    "MJAR0",
    "MMDB1",
    "FBMM0",
    "FDAC1",
    "FGET0",
    "MJSW0",
    "MJTC0",
    "MLJB0",
    "MLLL0",
    "MTAS1",
    "FCMJ0",
    "FDRW0",
    "FSSW0",
    "MABB0",
    "MABC0",
    "MAFM0",
    "MJBR0",
    "MRCZ0",
    "FDTB0",
    "FGJD0",
    "MJDM0",
    "MKHL0",
    "MLLW0",
    "FHLM0",
    "FKSV0",
    "MBTB0",
    "MGLR0",
    "MRE0",
]


# =============================================================================
# 61 to 48 Phoneme Mapping (Lee & Hon 1989 Standard)
# =============================================================================

# Standard 61->48 phoneme mapping for TIMIT
# Keys are the original 61 phonemes, values are the reduced 48 phonemes (or None to skip)
PHONEME_MAP_61_48 = {
    # --- Ignored (map to None, skip during data loading) ---
    "q": None,  # Glottal stop
    # --- Silence/Noise consolidation ---
    "h#": "sil",  # Begin/End marker
    "pau": "sil",  # Pause
    "epi": "sil",  # Epenthetic silence
    "sil": "sil",  # Silence itself
    # --- Vowel consolidation ---
    "ax-h": "ax",  # De-voiced schwa -> schwa
    "axr": "er",  # R-colored schwa -> er
    "ux": "uw",  # Fronted u -> uw
    # --- Consonant consolidation ---
    "em": "m",  # Syllabic m -> m
    "en": "n",  # Syllabic n -> n
    "nx": "n",  # Syllabic n variant
    "eng": "ng",  # Syllabic ng -> ng
    "hv": "hh",  # Voiced h -> h
    # --- Closure phones (RETAINED in 48-class standard) ---
    "pcl": "pcl",
    "tcl": "tcl",
    "kcl": "kcl",
    "qcl": None,  # qcl sometimes mapped to pcl
    "bcl": "bcl",
    "dcl": "dcl",
    "gcl": "gcl",
}


def get_label_mapping():
    """
    Generate complete 61->48 lookup table and phoneme-to-index mapping.

    Returns:
        phoneme_mapping: dict mapping 61 phonemes to 48 phonemes (or None to skip)
        phoneme_to_idx: dict mapping 48 phonemes to integer indices (0-47)
    """
    # All 61 original TIMIT labels (plus qcl which may appear in some variants)
    all_phonemes = [
        "aa",
        "ae",
        "ah",
        "ao",
        "aw",
        "ax",
        "ax-h",
        "axr",
        "ay",
        "b",
        "bcl",
        "ch",
        "d",
        "dcl",
        "dh",
        "dx",
        "eh",
        "el",
        "em",
        "en",
        "eng",
        "epi",
        "er",
        "ey",
        "f",
        "g",
        "gcl",
        "h#",
        "hh",
        "hv",
        "ih",
        "ix",
        "iy",
        "jh",
        "k",
        "kcl",
        "l",
        "m",
        "n",
        "ng",
        "nx",
        "ow",
        "oy",
        "p",
        "pau",
        "pcl",
        "q",
        "qcl",
        "r",
        "s",
        "sh",
        "sil",
        "t",
        "tcl",
        "th",
        "uh",
        "uw",
        "ux",
        "v",
        "w",
        "y",
        "z",
        "zh",
    ]

    # Build mapping dict
    phoneme_mapping = {}
    for p in all_phonemes:
        if p in PHONEME_MAP_61_48:
            phoneme_mapping[p] = PHONEME_MAP_61_48[p]
        else:
            phoneme_mapping[p] = p  # Keep unchanged

    # Remove None values (i.e., /q/) and create phoneme list
    valid_phonemes = sorted(
        list(set([v for v in phoneme_mapping.values() if v is not None]))
    )

    # Create phoneme to index mapping (0-47)
    phoneme_to_idx = {p: i for i, p in enumerate(valid_phonemes)}

    return phoneme_mapping, phoneme_to_idx, valid_phonemes


# Generate mappings at module load time
PHONEME_MAPPING, PHONEME_TO_IDX, PHONEME_LIST = get_label_mapping()


# =============================================================================
# TIMITPhonemeDataset Class
# =============================================================================


class TIMITPhonemeDataset(Dataset):
    """
    Dataset for TIMIT Phone Segment Classification.

    This dataset loads TIMIT audio files and extracts fixed-length windows centered
    around each phoneme segment for classification.

    Args:
        root_dir: Root directory of TIMIT dataset (contains TRAIN and TEST folders)
        subset: 'train', 'test', or 'val'
        window_size: Fixed window size in samples (default: 3072 for 192ms at 16kHz)
        use_all: If False, use Core Test Set for test and DEV_50 for val.
                 If True, use all available data (default: False)
        sample_rate: Expected sample rate of audio files (default: 16000)
        exclude_sa: Whether to exclude SA (dialect) sentences (default: True)
        transform: Optional transform to apply to waveform
        use_duration: Whether to return phoneme duration (default: False)
        normalize_duration: Whether to normalize duration using z-score (default: True)
        duration_stats: Tuple of (mean, std) for duration normalization. If None,
                       computed from training data. Only used when normalize_duration=True.

    Attributes:
        samples: List of tuples (wav_path, center_sample, label_idx)
        phoneme_counts: Dictionary counting occurrences of each phoneme
    """

    # Default duration statistics computed from TIMIT training set
    DEFAULT_DURATION_STATS = {
        "mean": 1500.0,  # Approximate mean duration in samples at 16kHz
        "std": 1200.0,  # Approximate std of duration in samples
    }

    def __init__(
        self,
        root_dir: str,
        subset: str = "train",
        window_size: int = 3072,
        use_all: bool = False,
        sample_rate: int = 16000,
        exclude_sa: bool = True,
        transform: Optional[callable] = None,  # type: ignore[assignment]
        use_duration: bool = False,
        normalize_duration: bool = True,
        duration_stats: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.subset = subset.lower()
        self.window_size = window_size
        self.use_all = use_all
        self.sample_rate = sample_rate
        self.exclude_sa = exclude_sa
        self.transform = transform
        self.use_duration = use_duration
        self.normalize_duration = normalize_duration and use_duration

        # Set duration normalization parameters
        if self.normalize_duration:
            if duration_stats is not None:
                self.duration_mean = duration_stats["mean"]
                self.duration_std = duration_stats["std"]
            else:
                # Use default statistics
                self.duration_mean = self.DEFAULT_DURATION_STATS["mean"]
                self.duration_std = self.DEFAULT_DURATION_STATS["std"]
        else:
            self.duration_mean = None
            self.duration_std = None

        # Validate subset
        if self.subset not in ["train", "test", "val"]:
            raise ValueError(
                f"Invalid subset: {subset}. Must be 'train', 'test', or 'val'"
            )

        # Half window for centered windowing
        self.half_window = window_size // 2

        # For validation set: select 30 non-Core-Test speakers from TEST directory
        self._val_speakers: set = set()
        if subset == "val" and not use_all:
            self._val_speakers = self._select_val_speakers()

        # Build samples index
        # Format: (wav_path, center_sample, label_idx, duration_samples)
        # duration_samples is only populated when use_duration=True
        self.samples: List[Tuple[str, int, int, int]] = []
        self.phoneme_counts: Dict[str, int] = {}

        self._build_index()

    def _select_val_speakers(self) -> set:
        """
        Select 30 non-Core-Test speakers from TEST directory for validation set.

        Returns:
            Set of folder names (e.g., {'FCMH0', 'FDAC1', ...}) to include in val set
        """
        test_dir = os.path.join(self.root_dir, "TEST")
        if not os.path.exists(test_dir):
            return set()

        # Collect all speakers from TEST directory
        all_test_speakers = []
        for dr_dir in sorted(os.listdir(test_dir)):
            dr_path = os.path.join(test_dir, dr_dir)
            if os.path.isdir(dr_path):
                for speaker_dir in sorted(os.listdir(dr_path)):
                    speaker_folder = speaker_dir.upper()
                    # Skip Core Test speakers
                    is_core = any(
                        speaker_folder.endswith(spk) for spk in CORE_TEST_SPEAKERS
                    )
                    if not is_core:
                        all_test_speakers.append(speaker_dir.upper())

        # Select first 30 speakers (sorted for reproducibility)
        selected = set(sorted(all_test_speakers)[:30])
        return selected

    def _build_index(self) -> None:
        """
        Scan TIMIT directory and build index of all valid phoneme segments.

        The index is built during __init__ to avoid repeated I/O during training.
        Each sample tuple contains: (wav_path, center_sample, label_idx)
        """
        # Determine which directory to scan based on subset
        if self.subset == "train":
            data_dir = os.path.join(self.root_dir, "TRAIN")
        else:  # test or val
            data_dir = os.path.join(self.root_dir, "TEST")

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"TIMIT directory not found: {data_dir}")

        # Collect all phoneme segments
        all_segments = []

        # Walk through all speaker directories (TIMIT has DR1-DR8 subdirectories)
        for dr_dir in sorted(os.listdir(data_dir)):
            dr_path = os.path.join(data_dir, dr_dir)
            if not os.path.isdir(dr_path):
                continue

            for speaker_dir in sorted(os.listdir(dr_path)):
                speaker_path = os.path.join(dr_path, speaker_dir)
                if not os.path.isdir(speaker_path):
                    continue

                # Find all WAV files in speaker directory (handle both .wav and .WAV)
                wav_files = glob.glob(os.path.join(speaker_path, "*.wav")) + glob.glob(
                    os.path.join(speaker_path, "*.WAV")
                )

                for wav_path in sorted(wav_files):
                    # Get base name without extension (handle both cases)
                    base_name = os.path.basename(wav_path)
                    for ext in [".wav", ".WAV"]:
                        if base_name.lower().endswith(ext.lower()):
                            base_name = base_name[: -len(ext)]
                            break

                    # Skip SA (dialect) sentences if required
                    if self.exclude_sa and base_name.startswith("SA"):
                        continue

                    # Find corresponding PHN file (handle case-insensitive extensions)
                    base_path = os.path.splitext(wav_path)[0]
                    phn_path = base_path + ".phn"

                    if not os.path.exists(phn_path):
                        # Try uppercase extension as fallback
                        phn_path = base_path + ".PHN"
                        if not os.path.exists(phn_path):
                            continue

                    # Parse PHN file and extract phoneme segments
                    segments = self._parse_phn_file(phn_path, wav_path)

                    # Filter based on subset requirements
                    # Match speaker ID suffix in folder name (e.g., 'DAB0' matches 'MDAB0' and 'FDAB0')
                    speaker_folder = speaker_dir.upper()

                    if self.subset == "test" and not self.use_all:
                        # Only include Core Test Set speakers (match by suffix)
                        if not any(
                            speaker_folder.endswith(spk) for spk in CORE_TEST_SPEAKERS
                        ):
                            continue

                    elif self.subset == "val":
                        if not self.use_all:
                            # From TEST directory, exclude Core Test speakers, select 30 speakers for val
                            # This list is pre-computed in __init__ to ensure consistency
                            if speaker_folder not in self._val_speakers:
                                continue
                        else:
                            # For 'val' with use_all=True, use non-core, non-dev speakers from TEST
                            if any(
                                speaker_folder.endswith(spk)
                                for spk in CORE_TEST_SPEAKERS
                            ) or any(
                                speaker_folder.endswith(spk) for spk in DEV_50_SPEAKERS
                            ):
                                continue

                    all_segments.extend(segments)

        # Build final sample list with label indices
        for wav_path, center_sample, phoneme, duration in all_segments:
            # Apply 61->48 mapping
            mapped_phoneme = self._map_phoneme(phoneme)

            # Skip phonemes mapped to None (e.g., 'q' glottal stop)
            if mapped_phoneme is None:
                continue

            # Convert to label index
            if mapped_phoneme not in PHONEME_TO_IDX:
                continue

            label_idx = PHONEME_TO_IDX[mapped_phoneme]

            self.samples.append((wav_path, center_sample, label_idx, duration))

            # Count phoneme occurrences
            self.phoneme_counts[mapped_phoneme] = (
                self.phoneme_counts.get(mapped_phoneme, 0) + 1
            )

    def _parse_phn_file(
        self, phn_path: str, wav_path: str
    ) -> List[Tuple[str, int, str, int]]:
        """
        Parse a TIMIT PHN file and extract phoneme segments.

        PHN file format:
            start_sample end_sample phoneme

        Example:
            0 2330 h#
            2330 2931 sh
            2931 3812 iy

        Args:
            phn_path: Path to the .phn file
            wav_path: Path to the corresponding .wav file

        Returns:
            List of tuples: (wav_path, center_sample, phoneme, duration_samples)
        """
        segments = []

        try:
            with open(phn_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) != 3:
                        continue

                    start_sample = int(parts[0])
                    end_sample = int(parts[1])
                    phoneme = parts[2]

                    # Calculate center sample
                    center_sample = (start_sample + end_sample) // 2
                    duration = end_sample - start_sample

                    segments.append((wav_path, center_sample, phoneme, duration))

        except (IOError, ValueError) as e:
            print(f"Warning: Error parsing {phn_path}: {e}")

        return segments

    def _map_phoneme(self, phoneme: str) -> Optional[str]:
        """
        Apply 61->48 phoneme mapping.

        Args:
            phoneme: Original 61 phoneme label

        Returns:
            Mapped 48 phoneme label, or None if phoneme should be skipped (e.g., 'q')
        """
        return PHONEME_MAPPING.get(phoneme, phoneme)

    def _load_audio_segment(self, wav_path: str, center_sample: int) -> torch.Tensor:
        """
        Load audio and extract fixed-length window centered at given sample.

        Uses replicate padding for boundary handling to maintain signal continuity.

        Args:
            wav_path: Path to WAV file
            center_sample: Sample index at center of window

        Returns:
            Waveform tensor of shape [1, window_size]
        """
        # Load audio
        waveform, orig_sample_rate = torchaudio.load(wav_path)

        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Ensure correct sample rate
        if orig_sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_sample_rate, self.sample_rate
            )

        # Calculate window boundaries
        start = center_sample - self.half_window
        end = center_sample + self.half_window

        # Handle boundaries with replicate padding
        if start < 0 or end > waveform.shape[1]:
            # Calculate padding needed
            pad_left = max(0, -start)
            pad_right = max(0, end - waveform.shape[1])

            # Extract available segment
            seg_start = max(0, start)
            seg_end = min(waveform.shape[1], end)
            segment = waveform[:, seg_start:seg_end]

            # Apply replicate padding
            segment = F.pad(segment, (pad_left, pad_right), mode="replicate")
        else:
            # Within bounds, extract directly
            segment = waveform[:, start:end]

        return segment

    def __len__(self) -> int:
        """Return total number of phoneme segments in dataset."""
        return len(self.samples)

    def _normalize_duration(self, duration: int) -> float:
        """
        Normalize duration using z-score with clipping.

        Args:
            duration: Raw phoneme duration in samples

        Returns:
            Normalized duration (z-score, clipped to [-3, 3])
        """
        if not self.normalize_duration:
            return float(duration)

        # Safely compute z-score with fallback
        mean_val = self.duration_mean if self.duration_mean is not None else 0.0
        std_val = self.duration_std if self.duration_std is not None else 1.0

        z_score = (duration - mean_val) / (std_val + 1e-8)
        # Clip to [-3, 3] to reduce impact of outliers
        z_score = max(-3.0, min(3.0, z_score))
        return z_score

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, float]]:
        """
        Get a single phoneme segment sample.

        Args:
            idx: Sample index

        Returns:
            If use_duration=False (default):
                Tuple of (waveform, label) where:
                    - waveform: Tensor of shape [1, window_size]
                    - label: Integer in range [0, 47]
            If use_duration=True:
                Tuple of (waveform, label, duration) where:
                    - waveform: Tensor of shape [1, window_size]
                    - label: Integer in range [0, 47]
                    - duration: Normalized phoneme duration (z-score, clipped to [-3, 3])
        """
        wav_path, center_sample, label_idx, duration = self.samples[idx]

        # Load audio segment
        waveform = self._load_audio_segment(wav_path, center_sample)

        # Apply optional transform
        if self.transform is not None:
            waveform = self.transform(waveform)

        # Z-score normalization: subtract mean and divide by std for zero-mean unit-variance
        # Shape: [1, window_size] -> compute mean and std along time dim (dim=-1)
        segment_mean = waveform.mean(dim=-1, keepdim=True)  # [1, 1]
        segment_std = waveform.std(dim=-1, keepdim=True)  # [1, 1]
        waveform = (waveform - segment_mean) / (
            segment_std + 1e-8
        )  # Z-score normalization

        if self.use_duration:
            normalized_duration = self._normalize_duration(duration)
            return waveform, label_idx, normalized_duration
        else:
            return waveform, label_idx

    def get_phoneme_distribution(self) -> Dict[str, int]:
        """
        Get distribution of phonemes in the dataset.

        Returns:
            Dictionary mapping phoneme to count
        """
        return self.phoneme_counts.copy()

    def get_num_classes(self) -> int:
        """
        Get number of phoneme classes.

        Returns:
            Number of unique phoneme classes (45 after 61->48 mapping)
        """
        return len(PHONEME_LIST)


# =============================================================================
# Utility Functions
# =============================================================================


def create_timit_datasets(
    root_dir: str,
    window_size: int = 3072,
    use_all: bool = False,
    sample_rate: int = 16000,
    exclude_sa: bool = True,
    use_duration: bool = False,
    normalize_duration: bool = True,
) -> Tuple[TIMITPhonemeDataset, TIMITPhonemeDataset, TIMITPhonemeDataset]:
    """
    Create train, validation, and test datasets for TIMIT.

    When use_duration=True and normalize_duration=True, the training set statistics
    (mean, std of duration) are computed and passed to validation/test sets for
    consistent normalization.

    Args:
        root_dir: Root directory of TIMIT dataset
        window_size: Fixed window size in samples
        use_all: Whether to use all data (True) or core/dev sets (False)
        sample_rate: Expected sample rate of audio files
        exclude_sa: Whether to exclude SA (dialect) sentences
        use_duration: Whether to return phoneme duration
        normalize_duration: Whether to normalize duration using z-score

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Create training dataset first to compute duration statistics
    train_dataset = TIMITPhonemeDataset(
        root_dir=root_dir,
        subset="train",
        window_size=window_size,
        use_all=True,  # Train always uses all available data
        sample_rate=sample_rate,
        exclude_sa=exclude_sa,
        use_duration=use_duration,
        normalize_duration=False,  # Compute raw duration first
    )

    # Compute duration statistics from training set
    duration_stats = None
    if use_duration and normalize_duration:
        durations = [sample[3] for sample in train_dataset.samples]
        duration_mean = float(np.mean(durations))
        duration_std = float(np.std(durations))
        duration_stats = {"mean": duration_mean, "std": duration_std}
        print(f"Duration statistics: mean={duration_mean:.2f}, std={duration_std:.2f}")

        # Recreate train dataset with normalization
        train_dataset = TIMITPhonemeDataset(
            root_dir=root_dir,
            subset="train",
            window_size=window_size,
            use_all=True,
            sample_rate=sample_rate,
            exclude_sa=exclude_sa,
            use_duration=use_duration,
            normalize_duration=True,
            duration_stats=duration_stats,
        )

    # Create val and test datasets with the same normalization statistics
    val_dataset = TIMITPhonemeDataset(
        root_dir=root_dir,
        subset="val",
        window_size=window_size,
        use_all=use_all,
        sample_rate=sample_rate,
        exclude_sa=exclude_sa,
        use_duration=use_duration,
        normalize_duration=normalize_duration,
        duration_stats=duration_stats,
    )

    test_dataset = TIMITPhonemeDataset(
        root_dir=root_dir,
        subset="test",
        window_size=window_size,
        use_all=use_all,
        sample_rate=sample_rate,
        exclude_sa=exclude_sa,
        use_duration=use_duration,
        normalize_duration=normalize_duration,
        duration_stats=duration_stats,
    )

    return train_dataset, val_dataset, test_dataset
