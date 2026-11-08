"""XRF55 RFID dataset reader.

Reads the RFID branch of XRF55, resamples each sample to a common length and
returns ``(train, val, test, num_classes)``.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from scipy.signal import resample

# Action ID to Category Mapping based on xrf55-data.md
ACTION_ID_TO_TYPE = {}
for i in range(1, 16):
    ACTION_ID_TO_TYPE[i] = "Human-Object Interaction Actions"
for i in range(16, 23):
    ACTION_ID_TO_TYPE[i] = "Human-Human Interaction Actions"
for i in range(23, 31):
    ACTION_ID_TO_TYPE[i] = "Fitness Actions"
for i in range(31, 45):
    ACTION_ID_TO_TYPE[i] = "Body Motion Actions"
for i in range(45, 56):
    ACTION_ID_TO_TYPE[i] = "Human-Computer Interaction Actions"


class XRF55Dataset(Dataset):
    def __init__(self, data, labels, filenames=None):
        """
        data: list or array of shape [N, C, T] (C=23, T=148 usually)
        labels: list or array of shape [N]
        filenames: list of strings (optional)
        """
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


def extract_metadata_from_filename(file_path):
    """
    Filename format: a_b_c.npy
    a: volunteer ID
    b: action class ID
    c: repetition
    """
    basename = os.path.basename(file_path)
    name_no_ext = os.path.splitext(basename)[0]
    parts = name_no_ext.split("_")

    if len(parts) >= 3:
        try:
            volunteer_id = int(parts[0])
            action_id = int(parts[1])
            repetition_id = int(parts[2])
            return volunteer_id, action_id, repetition_id
        except ValueError:
            return None, None, None
    return None, None, None


def load_and_process_file(file_path, target_sample_rate=30):
    """
    Load .npy file, linearize RSS, transpose if needed.
    Original shape can be [T, C] (e.g. 148, 23) or [C, T] (e.g. 23, 148)
    Target shape: [C, T]
    """
    ORIGINAL_RATE = 30.0
    EXPECTED_CHANNELS = 23

    try:
        if file_path.endswith(".npy"):
            data = np.load(file_path)
        elif file_path.endswith(".csv"):
            data = np.loadtxt(file_path, delimiter=",")
        else:
            return None

        if data.ndim != 2:
            return None

        # Linearize RSS: dBm -> linear
        data_linear = 10 ** (data / 10.0)

        # Determine if transpose is needed based on shape
        # If shape[1] == EXPECTED_CHANNELS (23), data is [T, C] -> transpose needed
        # If shape[0] == EXPECTED_CHANNELS (23), data is [C, T] -> no transpose needed
        if data_linear.shape[1] == EXPECTED_CHANNELS:
            # Data is [T, C], transpose to [C, T]
            data_linear = data_linear.T
        elif data_linear.shape[0] == EXPECTED_CHANNELS:
            # Data is already [C, T], no transpose needed
            pass
        else:
            # Ambiguous shape, check which dimension matches expected channels
            if abs(data_linear.shape[0] - EXPECTED_CHANNELS) < abs(
                data_linear.shape[1] - EXPECTED_CHANNELS
            ):
                # First dimension is closer to expected channels
                data_linear = data_linear
            else:
                # Second dimension is closer to expected channels
                data_linear = data_linear.T

        # Resample if needed (after potential transpose)
        T_orig, C = data_linear.shape
        if target_sample_rate != ORIGINAL_RATE:
            duration = T_orig / ORIGINAL_RATE
            target_T = int(duration * target_sample_rate)
            data_resampled = resample(data_linear, target_T, axis=0)
        else:
            data_resampled = data_linear

        # Ensure float32
        return data_resampled.astype(np.float32)

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def build_xrf55_dataset(
    root_dir,
    split_args,
    scenes=None,
    include_types=None,
    exclude_actions=None,
    exclude_repetitions=None,
    target_sample_rate=30,
):
    """
    root_dir: Directory containing scene subdirectories
    split_args: {'test_size': float, 'val_size': float, 'train_ratio': float}
    scenes: list of scene names to include (e.g. ['Scene1', 'Scene2'])
    include_types: list of strings (categories) to include using ACTION_ID_TO_TYPE
    exclude_actions: list of ints (action IDs) to exclude
    exclude_repetitions: list of ints (repetition IDs) to exclude
    """

    print(f"Scanning XRF55 files in {root_dir}...")

    all_files = []

    # Define search paths based on scenes
    if scenes is None:
        scenes = ["Scene1", "Scene1-part2", "Scene2", "Scene3", "Scene4"]
    search_paths = [os.path.join(root_dir, scene, "RFID") for scene in scenes]
    for path in search_paths:
        if not os.path.exists(path):
            print(f"Warning: Path {path} does not exist. Skipping.")
            continue
        print(f"Searching in {path}...")
        all_files.extend(glob.glob(os.path.join(path, "**", "*.npy"), recursive=True))
        all_files.extend(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))

    valid_files = []
    labels = []

    if exclude_actions is None:
        exclude_actions = []
    if exclude_repetitions is None:
        exclude_repetitions = []
    if include_types is None:
        include_types = []

    # Pre-check action-type filtering
    should_filter_types = len(include_types) > 0

    for f in all_files:
        vol_id, act_id, rep_id = extract_metadata_from_filename(f)
        if vol_id is None:
            continue

        # 1. Filter by Action Type (Category)
        if should_filter_types:
            category = ACTION_ID_TO_TYPE.get(act_id, "Unknown")
            if category not in include_types:
                continue

        # 2. Filter by Exclude Actions
        if act_id in exclude_actions:
            continue

        # 3. Filter by excluded repetitions
        if rep_id in exclude_repetitions:
            continue

        valid_files.append(f)
        labels.append(act_id)

    print(f"Found {len(valid_files)} valid files matching criteria.")

    if len(valid_files) == 0:
        return None, None, None, 0

    # Process Labels: Remap to 0..N-1
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)
    print(f"Encoded {num_classes} classes: {le.classes_}")

    # Split
    test_size = split_args.get("test_size", 0.2)
    val_size = split_args.get("val_size", 0.2)
    train_ratio = split_args.get("train_ratio", 1.0)

    # 1. TrainVal / Test
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        valid_files,
        labels_encoded,
        test_size=test_size,
        stratify=labels_encoded,
        random_state=42,
    )

    # 2. Train / Val
    relative_val_size = val_size / (1.0 - test_size)
    if relative_val_size >= 1.0:
        relative_val_size = 0.0

    train_files_full, val_files, train_labels_full, val_labels = train_test_split(
        train_val_files,
        train_val_labels,
        test_size=relative_val_size,
        stratify=train_val_labels,
        random_state=42,
    )

    # 3. Sample Train
    if train_ratio < 1.0:
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

    def process_batch(files, encoded_labels, desc):
        data_list = []
        final_labels = []
        final_names = []

        for f, l in tqdm(zip(files, encoded_labels), total=len(files), desc=desc):
            d = load_and_process_file(f, target_sample_rate)
            if d is not None:
                data_list.append(d)
                final_labels.append(l)
                final_names.append(os.path.basename(f))

        return data_list, final_labels, final_names

    print("Processing Train Set...")
    train_data, train_y, train_n = process_batch(train_files, train_labels, "Train")

    print("Processing Val Set...")
    val_data, val_y, val_n = process_batch(val_files, val_labels, "Val")

    print("Processing Test Set...")
    test_data, test_y, test_n = process_batch(test_files, test_labels, "Test")

    train_dataset = XRF55Dataset(train_data, train_y, train_n)
    val_dataset = XRF55Dataset(val_data, val_y, val_n)
    test_dataset = XRF55Dataset(test_data, test_y, test_n)

    return train_dataset, val_dataset, test_dataset, num_classes
