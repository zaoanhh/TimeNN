"""
Capture24 Dataset for Human Activity Recognition.

Based on the Dataset class from test_capture24_time_effort.py.
Loads numpy data and supports augmentation.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from timenn.transforms.augmentation import Augment


class Capture24Dataset(TorchDataset):
    """
    Dataset for Capture24 activity recognition.

    Args:
        data_cfg: Configuration dictionary with:
            - datadir: Path to data directory
            - label_method: Label method to use (e.g., 'WillettsSpecific2018')
            - labels: Dictionary of label methods and their class names
            - seq_length: Sequence length for temporal chunks
        X: Input features numpy array
        Y: Target labels numpy array
        transform: Augmentation transform to apply
        seq_length: Sequence length for temporal chunks (overrides config)
    """

    def __init__(
        self,
        data_cfg,
        X,
        Y=None,
        transform=None,
        seq_length=1,
    ):
        self.data_cfg = data_cfg
        self.X = X
        self.Y = Y
        self.transform = transform
        self.seq_length = seq_length

        # Get class names from config
        label_method = data_cfg.get("label_method", "WillettsSpecific2018")
        self.class_names = data_cfg["labels"].get(label_method, [])
        self.label_method = label_method

        # Process sequence length
        if self.seq_length > 1:
            # Chunk the data into sequences
            nX = int((len(X) // self.seq_length) * self.seq_length)
            nY = int((len(Y) // self.seq_length) * self.seq_length)
            X = [X[i : i + self.seq_length] for i in range(0, nX, self.seq_length)]
            Y = [Y[i : i + self.seq_length] for i in range(0, nY, self.seq_length)]
            self.X = np.asarray(X)
            self.Y = np.asarray(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        x = self._prepare_x(x)

        if self.Y is not None:
            y = self.Y[index]
            y = self._prepare_y(y)
            return x, y

        return x

    def _prepare_x(self, x):
        """Prepare input features with mean normalization along time axis."""
        if self.transform is not None:
            x = self.transform(x)

        # Mean normalization along time axis: (x - mean(x)) / |mean(x)|
        # Shape before: [L, C] where L=time, C=channels
        mean_per_channel = x.mean(axis=0)  # [C]
        x = x - mean_per_channel  # Center: subtract mean [L, C]
        abs_mean = np.abs(mean_per_channel) + 1e-8  # [C], avoid division by zero
        x = x / abs_mean  # Scale by absolute mean

        # Transpose from [L, C] to [C, L]
        x = x.T
        return x

    def _prepare_y(self, y):
        """Prepare target labels."""
        if self.seq_length > 1:
            return np.asarray([self._to_class_idx(_y) for _y in y])
        return self._to_class_idx(y)

    def _to_class_idx(self, y):
        """Convert label string to class index."""
        return self.class_names.index(y) if isinstance(y, str) else int(y)

    @property
    def num_classes(self):
        """Get number of classes."""
        return len(self.class_names)

    @property
    def class_to_idx(self):
        """Get class to index mapping."""
        return {cls: idx for idx, cls in enumerate(self.class_names)}


def create_capture24_datasets(
    datadir,
    label_method="WillettsSpecific2018",
    val_size=20,
    train_radio=1.0,
    seq_length=1,
    augment_cfg=None,
    labels_dict=None,
):
    """
    Create train, validation, and test datasets for Capture24.

    Args:
        datadir: Path to prepared data directory
        label_method: Label method to use
        val_size: Number of subjects for validation
        train_radio: Fraction of training data to use
        seq_length: Sequence length for temporal chunks
        augment_cfg: Augmentation configuration dictionary
        labels_dict: Dictionary of label methods and class names

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if labels_dict is None:
        labels_dict = {
            "WillettsSpecific2018": [
                "bicycling",
                "household-chores",
                "manual-work",
                "mixed-activity",
                "sitting",
                "sleep",
                "sports",
                "standing",
                "vehicle",
                "walking",
            ]
        }

    # Data configuration
    data_cfg = {
        "datadir": datadir,
        "label_method": label_method,
        "labels": labels_dict,
        "seq_length": seq_length,
        "val_size": val_size,
    }

    split_dir = os.path.join(datadir, "splits")
    X_train_path = os.path.join(split_dir, "X_train.npy")
    Y_train_path = os.path.join(split_dir, "Y_train.npy")
    X_val_path = os.path.join(split_dir, "X_val.npy")
    Y_val_path = os.path.join(split_dir, "Y_val.npy")
    X_test_path = os.path.join(split_dir, "X_test.npy")
    Y_test_path = os.path.join(split_dir, "Y_test.npy")

    if all(os.path.exists(p) for p in [X_train_path, X_val_path, X_test_path,
                                        Y_train_path, Y_val_path, Y_test_path]):
        print("Loading pre-split data...")
        X_train = np.load(X_train_path)
        Y_train = np.load(Y_train_path)
        X_val = np.load(X_val_path)
        Y_val = np.load(Y_val_path)
        X_test = np.load(X_test_path)
        Y_test = np.load(Y_test_path)
        print(f"  Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")
    else:
        print("Loading data files...")
        X = np.load(os.path.join(datadir, "X.npy"))
        print(f"  X.npy loaded: {X.shape}")
        y_file = f"Y_{label_method}.npy"
        Y = np.load(os.path.join(datadir, y_file))
        print(f"  {y_file} loaded: {Y.shape}")
        pid = np.load(os.path.join(datadir, "P.npy"))
        print(f"  P.npy loaded: {pid.shape}")

        print("Splitting data by subject...")
        whr_deriv = np.isin(pid, [f"P{i:03d}" for i in range(1, 101)])
        X_deriv, Y_deriv, pid_deriv = X[whr_deriv], Y[whr_deriv], pid[whr_deriv]
        X_test, Y_test, _ = X[~whr_deriv], Y[~whr_deriv], pid[~whr_deriv]
        print(f"  Train/Val split: {X_deriv.shape[0]} samples")
        print(f"  Test split: {X_test.shape[0]} samples")

        print("Creating validation split...")
        np.random.seed(42)
        whr_val = np.isin(
            pid_deriv,
            np.random.choice(np.unique(pid_deriv), size=val_size, replace=False),
        )
        X_val, Y_val, _ = X_deriv[whr_val], Y_deriv[whr_val], pid_deriv[whr_val]
        X_train, Y_train, _ = (
            X_deriv[~whr_val],
            Y_deriv[~whr_val],
            pid_deriv[~whr_val],
        )

        del X, X_deriv, pid, pid_deriv

        print(f"  Train samples: {X_train.shape[0]}")
        print(f"  Val samples: {X_val.shape[0]}")

    # Create augmentation transform
    if augment_cfg is not None:
        transform = Augment(
            augment_cfg.get("jitter", {}).get("sigma", 0.0),
            augment_cfg.get("jitter", {}).get("prob", 0.0),
            augment_cfg.get("shift", {}).get("window", 0),
            augment_cfg.get("shift", {}).get("prob", 0.0),
            augment_cfg.get("twarp", {}).get("sigma", 0.0),
            augment_cfg.get("twarp", {}).get("knots", 4),
            augment_cfg.get("twarp", {}).get("prob", 0.0),
            augment_cfg.get("mwarp", {}).get("sigma", 0.0),
            augment_cfg.get("mwarp", {}).get("knots", 4),
            augment_cfg.get("mwarp", {}).get("prob", 0.0),
        )
    else:
        transform = None

    # Apply training data subset
    if train_radio < 1.0:
        total_len = len(X_train)
        train_size = int(total_len * train_radio)
        all_indices = torch.randperm(total_len)
        train_indices = all_indices[:train_size]
        X_train = X_train[train_indices]
        Y_train = Y_train[train_indices]
        print(f"  Train subset: {X_train.shape[0]} samples ({train_radio * 100:.1f}%)")

    # Create datasets
    train_dataset = Capture24Dataset(
        data_cfg, X_train, Y_train, transform=transform, seq_length=seq_length
    )
    val_dataset = Capture24Dataset(
        data_cfg, X_val, Y_val, transform=None, seq_length=seq_length
    )
    test_dataset = Capture24Dataset(
        data_cfg, X_test, Y_test, transform=None, seq_length=seq_length
    )

    print(
        f"Dataset created with {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples"
    )
    print(f"Number of classes: {train_dataset.num_classes}")
    print(f"Class names: {train_dataset.class_names}")

    return train_dataset, val_dataset, test_dataset
