
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

def preprocess_sample_simple(file_path):
    """
    Load a ``.npy`` sample, convert complex CSI to magnitude, and apply
    per-sample mean normalisation ``(x - mean) / mean``.
    """
    try:
        data = np.load(file_path)
        # data shape expected: [270, 1000]
        
        # Per-sample scalar mean normalisation.
        
        if np.iscomplexobj(data):
            data = np.abs(data)
            
        u = np.mean(data)
        if u == 0:
            u = 1e-8
        
        normalized = (data - u) / u
        return normalized.astype(np.float32)
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

class XRF55WarpDataset(Dataset):
    def __init__(self, data_list, label_list, filename_list):
        self.data_list = data_list
        self.label_list = torch.tensor(label_list, dtype=torch.long)
        self.filename_list = filename_list
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        return self.data_list[idx], self.label_list[idx]

def build_warp_dataset(root_dir, csv_path, scenes, train_filter, test_filter, train_ratio=1.0, val_size=0.1):
    """
    Build dataset for Warp Study.
    
    Args:
        root_dir: Base directory of the dataset.
        csv_path: Path to clustering CSV (contains full_path, scene_name, gesture_id, scattering_label)
        scenes: List of scene names to include (e.g. ['Scene1', 'Scene2'])
        train_filter: List of scattering_labels for Training set
        test_filter: List of scattering_labels for Test set
        train_ratio: Float, fraction of training data to use
        val_size: Float, fraction of training data to use for validation
        
    Returns:
        train_ds, val_ds, test_ds, num_classes
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # REQUIRED Columns: full_path, scene_name, gesture_id, scattering_label
    required_cols = ['full_path', 'scene_name', 'gesture_id', 'scattering_label']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV missing column: {col}")
            
    # Filter by Scenes
    if scenes:
        df = df[df['scene_name'].isin(scenes)]
        
    if len(df) == 0:
        raise ValueError("No data found after scene filtering.")
        
    print(f"Total samples after scene filtering: {len(df)}")
    
    # Group samples by scattering cluster and split according to the
    # requested train/test cluster filters.

    # Encode Gesture Labels First
    all_gestures = sorted(df['gesture_id'].unique())
    le = LabelEncoder()
    le.fit(all_gestures)
    num_classes = len(le.classes_)
    print(f"Num Classes: {num_classes}, Classes: {le.classes_}")
    
    # Create Lists for Split
    full_files = df['full_path'].tolist()
    full_gestures = df['gesture_id'].tolist()
    full_clusters = df['scattering_label'].tolist()
    
    full_gestures_enc = le.transform(full_gestures)
    
    # Group indices by cluster label
    indices_by_cluster = {}
    for idx, c_lbl in enumerate(full_clusters):
        if c_lbl not in indices_by_cluster:
            indices_by_cluster[c_lbl] = []
        indices_by_cluster[c_lbl].append(idx)
        
    train_indices = []
    test_indices = []
    
    train_filter_set = set(train_filter)
    test_filter_set = set(test_filter)
    
    all_present_clusters = sorted(indices_by_cluster.keys())
    
    # Fraction of a cluster used for test when it appears in both filters.
    disjoint_test_ratio = 0.2
    
    for c_lbl in all_present_clusters:
        indices = indices_by_cluster[c_lbl]
        in_train = c_lbl in train_filter_set
        in_test = c_lbl in test_filter_set
        
        if in_train and in_test:
            # Disjoint split
            tr_i, te_i = train_test_split(indices, test_size=disjoint_test_ratio, random_state=42)
            train_indices.extend(tr_i)
            test_indices.extend(te_i)
        elif in_train:
            train_indices.extend(indices)
        elif in_test:
            # For a test-only cluster, keep the same test fraction as above.
            _, te_i = train_test_split(indices, test_size=disjoint_test_ratio, random_state=42)
            test_indices.extend(te_i)
            
    print(f"Indices after filtering: Train={len(train_indices)}, Test={len(test_indices)}")
    
    if len(train_indices) == 0:
         print("Warning: Training set is empty!")

    # Construct Train Lists
    train_files = [full_files[i] for i in train_indices]
    train_labels_enc = [full_gestures_enc[i] for i in train_indices]

    # Split Train/Val
    if len(train_files) > 0:
        if len(train_files) > 1:
            try:
                tr_files, val_files, tr_lbls, val_lbls = train_test_split(
                    train_files, train_labels_enc, test_size=val_size, stratify=train_labels_enc, random_state=42
                )
            except ValueError:
                 tr_files, val_files, tr_lbls, val_lbls = train_test_split(
                    train_files, train_labels_enc, test_size=val_size, random_state=42
                )
        else:
            # If 1 sample, all to train, val empty
             tr_files, val_files, tr_lbls, val_lbls = train_files, [], train_labels_enc, []
    else:
        tr_files, val_files, tr_lbls, val_lbls = [], [], [], []

    # Apply Train Ratio (Subset Training Data)
    if train_ratio < 1.0 and len(tr_files) > 0:
        subset_size = int(len(tr_files) * train_ratio)
        if subset_size > 0:
            try:
                tr_files, _, tr_lbls, _ = train_test_split(
                    tr_files, tr_lbls, train_size=train_ratio, stratify=tr_lbls, random_state=42
                )
            except:
                 tr_files, _, tr_lbls, _ = train_test_split(
                    tr_files, tr_lbls, train_size=train_ratio, random_state=42
                )
                
    # Construct Test Lists
    test_files = [full_files[i] for i in test_indices]
    test_labels_enc = [full_gestures_enc[i] for i in test_indices]
    
    print(f"Final Count - Train: {len(tr_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # Load Data Helper
    def load_data_list(files, lbls, desc):
        data_list = []
        valid_lbls = []
        valid_names = []
        for f, l in tqdm(zip(files, lbls), total=len(files), desc=desc):
            d = preprocess_sample_simple(f)
            if d is not None:
                data_list.append(d)
                valid_lbls.append(l)
                valid_names.append(os.path.basename(f))
        return data_list, valid_lbls, valid_names

    # Load All
    tr_data, tr_l, tr_n = load_data_list(tr_files, tr_lbls, "Loading Train")
    val_data, val_l, val_n = load_data_list(val_files, val_lbls, "Loading Val")
    te_data, te_l, te_n = load_data_list(test_files, test_labels_enc, "Loading Test")
    
    train_ds = XRF55WarpDataset(tr_data, tr_l, tr_n)
    val_ds = XRF55WarpDataset(val_data, val_l, val_n)
    test_ds = XRF55WarpDataset(te_data, te_l, te_n)
    
    return train_ds, val_ds, test_ds, num_classes
