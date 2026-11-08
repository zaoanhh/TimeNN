"""Widar 3.0 CSI dataset reader.

The loader scans ``*.npy`` files, extracts the gesture label from the file
name, band-pass filters the Doppler spectrum, resamples to a common rate and
splits the data into train/validation/test sets.
"""

import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.signal import resample, butter, filtfilt


class WidarDataset(Dataset):
    def __init__(self, data, labels, filenames=None):
        """
        data: list or array of shape [N, C, T]
        labels: list or array of shape [N]
        filenames: list of strings (optional)
        """
        # Ensure data is tensor
        if isinstance(data, list):
            self.data = data
        else:
            self.data = (
                torch.tensor(data, dtype=torch.float32)
                if not isinstance(data, torch.Tensor)
                else data
            )

        self.labels = torch.tensor(labels, dtype=torch.long)
        self.filenames = filenames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.filenames is not None:
            return self.data[idx], self.labels[idx], self.filenames[idx]
        return self.data[idx], self.labels[idx]


def extract_label_from_filename(file_path):
    """
    Filename format: id-a-b-c-d-Rx.npy
    'a': gesture type (label)
    """
    file_name = os.path.basename(file_path)
    parts = file_name.split("-")
    if len(parts) >= 2:
        return int(parts[1])
    return None


def extract_rx_from_filename(file_path):
    """
    Filename format: ...-rX.npy where X is rx_id
    """
    file_name = os.path.basename(file_path).replace(".npy", "")
    parts = file_name.split("-")
    # The last part should be rX
    last_part = parts[-1]
    if last_part.startswith("r"):
        try:
            return int(last_part[1:])
        except ValueError:
            return None
    return None


def preprocess_widar_sample(
    file_path,
    subcarrier_ids,
    antenna_ids,
    target_sample_rate,
    bandpass_freqs=None,
    original_sample_rate=1000,
):
    """
    Load and preprocess a single Widar sample.
    """
    try:
        # 1. Load Data
        # Shape: [T, subcarrier, rxantenna_id, 1]
        data = np.load(file_path)

        # 2. Select Antennas
        # antenna_ids should be a list of 2 indices, e.g. [0, 1]
        # Validate antenna indices.
        if data.shape[2] <= max(antenna_ids):
            return None

        # Select the two antennas -> [T, sub, 1]
        data_ant = data[:, :, antenna_ids, 0]

        # 3. Select Subcarriers -> [T, N_sub, 1]
        data_sub = data_ant[:, subcarrier_ids, :]

        # 3.5 Pad/Truncate to target_length (1500) BEFORE downsampling
        CURRENT_FIXED_LENGTH = 2000
        current_T = data_sub.shape[0]
        if current_T > CURRENT_FIXED_LENGTH:
            data_sub = data_sub[:CURRENT_FIXED_LENGTH]
        elif current_T < CURRENT_FIXED_LENGTH:
            pad_len = CURRENT_FIXED_LENGTH - current_T
            # Pad along axis 0 (Time)
            # data_sub shape [T, N_sub, 1] -> pad ((0, pad), (0,0), (0,0))
            data_sub = np.pad(data_sub, ((0, pad_len), (0, 0), (0, 0)), mode="constant")

        # 4. Downsample Time
        T, N_sub, _ = data_sub.shape
        duration = T / original_sample_rate
        target_T = int(duration * target_sample_rate)

        if target_T < 2:
            return None

        # Resample along axis 0 (Time)
        # resample returns [target_T, N_sub, 1]
        data_resampled = resample(data_sub, target_T, axis=0)

        # 4.5 Pad to power of 2 for Kymatio compatibility
        # Kymatio requires T to be divisible by 2**J. Power of 2 is safest.
        next_pow2 = 1 if target_T == 0 else 2 ** (target_T - 1).bit_length()
        # Ensure minimum size for scattering (e.g. 2**J_max, J=5 -> 32)
        next_pow2 = max(next_pow2, 32)

        if next_pow2 > target_T:
            pad_len = next_pow2 - target_T
            # Pad along axis 0 (Time)
            # data_resampled shape [target_T, N_sub, 1]
            data_resampled = np.pad(
                data_resampled, ((0, pad_len), (0, 0), (0, 0)), mode="constant"
            )

        # 5. Conjugate Multiply
        # antenna1 * conj(antenna2)
        # [target_T, N_sub]
        # ant1 = data_resampled[..., 0]
        # ant2 = data_resampled[..., 1]
        # conj_mul = ant1 * np.conj(ant2)
        conj_mul = data_resampled[..., 0]

        # 6. Amplitude Mean Normalization
        # (x - u) / E(|x|)

        # Mean per subcarrier: [1, N_sub]
        u = np.mean(conj_mul, axis=0, keepdims=True)

        # E(|x|) per subcarrier
        # Mean of amplitude
        e_abs_x = np.mean(np.abs(conj_mul), axis=0, keepdims=True)

        # Avoid division by zero
        e_abs_x[e_abs_x == 0] = 1e-8

        normalized = (conj_mul - u) / e_abs_x

        # 6.5 Band Pass Filter (Configured)
        if bandpass_freqs is not None and len(bandpass_freqs) == 2:
            nyq = 0.5 * target_sample_rate
            low = bandpass_freqs[0] / nyq
            high = bandpass_freqs[1] / nyq
            if low > 0 and high < 1:
                b, a = butter(6, [low, high], btype="band")
                normalized = filtfilt(b, a, normalized, axis=0)

        # 7. Transpose to [N_sub, T] -> [Channel, Time]
        # output shape: [N_sub, T_new]
        final_data = normalized.T

        # Take magnitude to convert complex to real
        final_data = np.abs(final_data)

        # Return as float32 (magnitude)
        return final_data.astype(np.float32)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


from sklearn.preprocessing import LabelEncoder


def build_widar_dataset(
    root_dir,
    subcarrier_ids,
    antenna_ids,
    target_sample_rate,
    split_args,
    bandpass_freqs=None,
    exclude_labels=None,
    rx_device_ids=None,
):
    """
    root_dir: Path to dataset
    subcarrier_ids: List of subcarrier indices
    antenna_ids: List of 2 antenna indices
    target_sample_rate: int
    split_args: dict {'test_size': float, 'val_size': float, 'train_ratio': float}
    bandpass_freqs: list [low, high] (optional)
    exclude_labels: list of int (optional) - labels to ignore
    rx_device_ids: list of int (optional) - rx devices to include
    """
    print(f"Scanning files in {root_dir}...")
    all_files = glob.glob(os.path.join(root_dir, "**", "*.npy"), recursive=True)

    # Filter valid files and extract labels
    valid_files = []
    labels = []

    if exclude_labels is None:
        exclude_labels = []

    for f in all_files:
        # 1. Filter by Rx Device
        if rx_device_ids is not None:
            rx_id = extract_rx_from_filename(f)
            if rx_id is None or rx_id not in rx_device_ids:
                continue

        l = extract_label_from_filename(f)
        if l is not None:
            if l in exclude_labels:
                continue
            valid_files.append(f)
            labels.append(l)

    print(f"Found {len(valid_files)} valid files.")

    if len(valid_files) == 0:
        return None, None, None

    # Encode Labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)
    print(f"Encoded {num_classes} classes: {le.classes_}")

    # Stratified Split
    # Test Split
    test_size = split_args.get("test_size", 0.2)
    val_size = split_args.get("val_size", 0.2)
    # train_ratio is how much of the REMAINING data to use. 1.0 means use all remaining.
    train_ratio = split_args.get("train_ratio", 1.0)

    # 1. Split Test
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        valid_files,
        labels_encoded,
        test_size=test_size,
        stratify=labels_encoded,
        random_state=42,
    )

    # 2. Split Val from Train_Val
    # Adjust val_size relative to train_val size
    # logical val size is val_size (relative to total)
    # relative_val_size = val_size / (1.0 - test_size)
    relative_val_size = val_size / (1.0 - test_size)

    train_files_full, val_files, train_labels_full, val_labels = train_test_split(
        train_val_files,
        train_val_labels,
        test_size=relative_val_size,
        stratify=train_val_labels,
        random_state=42,
    )

    # 3. Sample Train
    if train_ratio < 1.0:
        # Sample subset of training data
        train_files, _, train_labels, _ = train_test_split(
            train_files_full,
            train_labels_full,
            train_size=train_ratio,
            stratify=train_labels_full,
            random_state=42,
        )
    else:
        train_files = train_files_full
        train_labels = train_labels_full

    print(
        f"Split sizes: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}"
    )

    # Helper to process a list of files
    def process_files(file_list, labels_list, desc):
        data_list = []
        label_list = []
        filename_list = []
        # Adaptive disable: disable if stdout is not a TTY (e.g. redirected to file)
        disable_tqdm = not sys.stdout.isatty()
        for f, l in tqdm(
            zip(file_list, labels_list),
            total=len(file_list),
            desc=desc,
            disable=disable_tqdm,
        ):
            d = preprocess_widar_sample(
                f,
                subcarrier_ids,
                antenna_ids,
                target_sample_rate,
                bandpass_freqs=bandpass_freqs,
            )
            if d is not None:
                data_list.append(d)
                label_list.append(l)
                filename_list.append(os.path.basename(f))
        return data_list, label_list, filename_list

    # Process datasets.

    print("Processing Train Set...")
    train_data, train_labels_clean, train_filenames = process_files(
        train_files, train_labels, "Train"
    )

    print("Processing Val Set...")
    val_data, val_labels_clean, val_filenames = process_files(
        val_files, val_labels, "Val"
    )

    print("Processing Test Set...")
    test_data, test_labels_clean, test_filenames = process_files(
        test_files, test_labels, "Test"
    )

    train_dataset = WidarDataset(train_data, train_labels_clean, train_filenames)
    val_dataset = WidarDataset(val_data, val_labels_clean, val_filenames)
    test_dataset = WidarDataset(test_data, test_labels_clean, test_filenames)

    return train_dataset, val_dataset, test_dataset, num_classes
