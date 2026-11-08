"""XRF55 Wi-Fi split with a random (rather than repetition-based) split.

A plain random split is used by the motivation study so that the clean/warped
accuracy gap is not confounded by subject identity.
"""


import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

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

class XRF55WifiDataset(Dataset):
    def __init__(self, data, labels, filenames=None, augment=False, noise_std=0.01, target_sample_rate=None, demean=False):
        """
        data: list or array of shape [N, C, T]
        labels: list or array of shape [N]
        filenames: list of strings (optional)
        augment: bool
        noise_std: float
        target_sample_rate: int (optional) - if set, resample to this rate (original 200Hz)
        demean: bool (optional) - if True, substract mean along time dim
        """
        if isinstance(data, list):
            self.data = data
        else:
            self.data = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data
            
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.filenames = filenames
        self.augment = augment
        self.noise_std = noise_std
        self.target_sample_rate = target_sample_rate
        self.demean = demean
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Ensure tensor
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)
            
        # Demean
        if self.demean:
            # sample: [270, T]
            # mean along T (dim 1)
            mean = sample.mean(dim=1, keepdim=True)
            sample = sample - mean

        # Resample
        if self.target_sample_rate is not None:
             # sample: [C, T] -> [1, C, T] for interpolate
             # Original rate 200 Hz
             # Target length: T' = T * (target / 200).
             
             original_rate = 200.0
             if self.target_sample_rate != original_rate:
                 new_len = int(sample.shape[1] * (self.target_sample_rate / original_rate))
                 sample = sample.unsqueeze(0) # [1, C, T]
                 sample = torch.nn.functional.interpolate(sample, size=new_len, mode='linear', align_corners=False)
                 sample = sample.squeeze(0) # [C, T']

        if self.augment:
             # Add Gaussian Noise
             noise = torch.randn_like(sample) * self.noise_std
             sample = sample + noise

        if self.filenames is not None:
            return sample, self.labels[idx], self.filenames[idx]
        return sample, self.labels[idx]

def extract_metadata_from_filename(filename):
    """
    Filename format: a_b_c.npy 
    a: volunteer ID
    b: action class ID
    c: repetition
    """
    name_no_ext = os.path.splitext(filename)[0]
    parts = name_no_ext.split('_')
    if len(parts) >= 3:
        try:
            return int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            pass
    return None, None, None

def load_sample(file_path):
    """
    Load data from single wifi file.
    Expected raw shape: [T, 270].
    Output shape: [270, T]
    """
    if not os.path.exists(file_path):
        return None
    
    try:
        # Load Data
        data = np.load(file_path) # [T, 270] ?
        
        # Handle Complex
        if np.iscomplexobj(data):
            data = np.abs(data)
            
        data = data.astype(np.float32)
        
        # Check shape
        # Most samples have T=1000, C=270. Convert to [C, T].
        if data.shape[1] == 270:
            data = data.T # [270, T]
        elif data.shape[0] == 270:
            pass # Already [270, T]
            
        return data
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def build_xrf55_dataset(root_dir, 
                       split_args, 
                       scenes=None,
                       include_types=None, 
                       subcarrier_ids=None, # Not used for wifi all
                       augment_conf=None,
                       target_sample_rate=None,
                       demean=False):
    """
    root_dir: Path to xrf55-all/
    split_args: dict, must contain:
        - test_ratio: float (e.g. 0.2)
        - val_ratio: float (e.g. 0.1, relative to TOTAL)
        - train_ratio: float (e.g. 1.0, sub-sample train set)
    """
    if scenes is None:
        scenes = ["Scene1", "Scene1-part2", "Scene2", "Scene3", "Scene4"]

    print(f"Scanning XRF55 WiFi files in {root_dir}...")
    
    valid_samples = [] 
    labels = []
    
    for scene in scenes:
        scene_path = os.path.join(root_dir, scene)
        wifi_path = os.path.join(scene_path, 'WiFi')
        
        if not os.path.exists(wifi_path):
            continue
            
        files = glob.glob(os.path.join(wifi_path, '*.npy'))
        for f in files:
            fname = os.path.basename(f)
            vol, act, rep = extract_metadata_from_filename(fname)
            
            if vol is None: continue
            
            if include_types:
                cat = ACTION_ID_TO_TYPE.get(act, "Unknown")
                if cat not in include_types:
                    continue
                    
            valid_samples.append({
                'path': f,
                'label': act,
                'name': f"{scene}_{fname}",
                'rep': rep
            })
            labels.append(act)
            
    print(f"Found {len(valid_samples)} samples.")
    
    if len(valid_samples) == 0:
        return None, None, None, 0

    # Encode Labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)
    print(f"Encoded {num_classes} classes: {le.classes_}")
    
    # --- Random split ---
    train_ratio_subsample = split_args.get('train_ratio', 1.0)
    test_ratio = split_args.get('test_ratio', 0.2)
    val_ratio_total = split_args.get('val_ratio', 0.1) 
    
    all_indices = np.arange(len(valid_samples))
    all_labels = labels_encoded
    
    # 1. Split Test
    # If 0.2, then 20% test, 80% remaining
    remaining_idx, test_idx = train_test_split(
        all_indices, 
        test_size=test_ratio, 
        stratify=all_labels, 
        random_state=42
    )
    remaining_labels = all_labels[remaining_idx]
    
    # 2. Split validation
    # Validation is a fraction of the total dataset; convert it to a fraction
    # of the remaining training data.
    remaining_proportion = 1.0 - test_ratio
    if remaining_proportion > 0:
        val_relative_size = val_ratio_total / remaining_proportion
    else:
        val_relative_size = 0.0
        
    # Clamp to a valid ratio.
    if val_relative_size >= 1.0 or val_relative_size < 0:
        print(f"Warning: Invalid Val split size {val_relative_size}, falling back to 0.1 of remaining")
        val_relative_size = 0.1

    train_idx, val_idx = train_test_split(
        remaining_idx,
        test_size=val_relative_size,
        stratify=remaining_labels,
        random_state=42
    )
    
    # 3. Train Subsample
    if train_ratio_subsample < 1.0:
        train_labels_current = all_labels[train_idx]
        train_idx, _ = train_test_split(
            train_idx,
            train_size=train_ratio_subsample,
            stratify=train_labels_current,
            random_state=42
        )
        
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    def create_dataset(indices, desc):
        data_list = []
        label_list = []
        name_list = []
        
        # Adaptive disable: disable if stdout is not a TTY
        disable_tqdm = not sys.stdout.isatty()
        for idx in tqdm(indices, desc=desc, disable=disable_tqdm):
            sample = valid_samples[idx]
            lbl_enc = labels_encoded[idx]
            
            d = load_sample(sample['path'])
            
            if d is not None:
                data_list.append(d)
                label_list.append(lbl_enc)
                name_list.append(sample['name'])
        
        augment = False
        noise_std = 0.01
        if augment_conf and desc == "Train":
            augment = augment_conf.get('enabled', False)
            noise_std = augment_conf.get('noise_std', 0.01)
                
        return XRF55WifiDataset(data_list, label_list, name_list, 
                                        augment=augment, 
                                        noise_std=noise_std, 
                                        target_sample_rate=target_sample_rate, 
                                        demean=demean)

    train_ds = create_dataset(train_idx, "Train")
    val_ds = create_dataset(val_idx, "Val")
    test_ds = create_dataset(test_idx, "Test")
    
    return train_ds, val_ds, test_ds, num_classes
