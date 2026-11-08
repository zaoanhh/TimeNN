
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from timenn.data.xrf55_wifi import extract_metadata_from_filename, load_sample, ACTION_ID_TO_TYPE
from timenn.transforms.xrf55 import XRF55Augment

class XRF55UserRatioDataset(Dataset):
    def __init__(self, data, augment_fn=None):
        """
        data_items: list of dicts ``{"path", "label", "name", "augment"}``.
        augment_fn: ``XRF55Augment`` instance or ``None``.
        """
        self.data_items = data
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.data_items)
    
    def __getitem__(self, idx):
        item = self.data_items[idx]
        
        # Load samples lazily; expanding the index list is cheap, but loading
        # every augmented copy into RAM would not be.
        data = load_sample(item['path'])
        if data is None:
            # Unreadable samples are replaced with zeros.
            data = np.zeros((270, 1000), dtype=np.float32)
            
        # Convert to tensor
        data = torch.tensor(data, dtype=torch.float32)

        # Augment
        if item['augment'] and self.augment_fn is not None:
            # The augmenters operate on numpy arrays.
            data_np = data.numpy()
            data_aug_np = self.augment_fn(data_np)
            data = torch.tensor(data_aug_np, dtype=torch.float32)
        
        label = item['label']
        vol = item.get('vol', -1)
            
        return data, label, vol

def build_xrf55_user_ratio_dataset(root_dir, 
                             ratio, 
                             aug_type, 
                             scenes=None,
                             include_types=None):
    """
    ratio: float, 0.1 to 1.0. Ratio of users in Set A.
    aug_type: str.
    """
    if scenes is None:
        scenes = ["Scene1", "Scene1-part2", "Scene2", "Scene3", "Scene4"]
        
    print(f"Scanning XRF55 WiFi files in {root_dir}...")
    
    # 1. Collect all valid samples
    valid_samples = [] 
    
    # Collect all files first to identify volunteers.
    volunteers = set()
    
    # Scan logic (shared with dataset_xrf55_wifi.py)
    for scene in scenes:
        scene_path = os.path.join(root_dir, scene)
        wifi_path = os.path.join(scene_path, 'WiFi')
        if not os.path.exists(wifi_path): continue
        
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
                'label': act, # raw ID
                'vol': vol,
                'rep': rep,
                'name': fname
            })
            volunteers.add(vol)
            
    print(f"Found {len(valid_samples)} samples from {len(volunteers)} volunteers.")
    
    # 2. Split Volunteers into Set A (Augment) and Set B (No Augment)
    volunteers = sorted(list(volunteers))
    n_vol = len(volunteers)
    n_a = int(n_vol * ratio)
    
    # The first ``n_a`` sorted volunteer IDs form Set A.
    set_a_vols = set(volunteers[:n_a])
    set_b_vols = set(volunteers[n_a:])
    
    print(f"Augmentation Ratio {ratio}: {len(set_a_vols)} Users in Set A (Augment), {len(set_b_vols)} Users in Set B.")

    # 3. Partition samples into Train/Val/Test
    # Train: Rep 1-14
    # Val: Rep 15-17
    # Test: Rep 18+
    
    # Encode labels
    all_labels = [s['label'] for s in valid_samples]
    le = LabelEncoder()
    le.fit(all_labels)
    num_classes = len(le.classes_)
    
    augmenter = XRF55Augment(aug_type)
    
    train_items = []
    val_items = []
    test_items = []
    
    for sample in valid_samples:
        vol = sample['vol']
        rep = sample['rep']
        lbl = le.transform([sample['label']])[0]
        path = sample['path']
        
        item_base = {'path': path, 'label': lbl, 'augment': False, 'vol': vol}
        
        # Test Set: All users get augmented 4x + 1x original
        if rep >= 18:
            # Add Original
            test_items.append(item_base)
            # Add 4 Augmented copies
            for _ in range(4):
                item_aug = item_base.copy()
                item_aug['augment'] = True
                test_items.append(item_aug)
                
        # Validation Set
        elif rep >= 15:
            # Same logic as Train set as per feedback
            if vol in set_a_vols:
                # Add Original
                val_items.append(item_base)
                # Add 4 Augmented copies
                for _ in range(4):
                    item_aug = item_base.copy()
                    item_aug['augment'] = True
                    val_items.append(item_aug)
            else:
                # Only Original
                val_items.append(item_base)
                
        # Train Set
        else: # rep <= 14
            if vol in set_a_vols:
                 # Add Original
                train_items.append(item_base)
                # Add 4 Augmented copies
                for _ in range(4):
                    item_aug = item_base.copy()
                    item_aug['augment'] = True
                    train_items.append(item_aug)
            else:
                # Only Original
                train_items.append(item_base)
                
    # Create Datasets
    # Pass `augmenter` to datasets
    
    train_ds = XRF55UserRatioDataset(train_items, augment_fn=augmenter)
    val_ds = XRF55UserRatioDataset(val_items, augment_fn=augmenter)
    test_ds = XRF55UserRatioDataset(test_items, augment_fn=augmenter)
    
    return train_ds, val_ds, test_ds, num_classes, set_a_vols, set_b_vols, le

