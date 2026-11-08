
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from timenn.data.xrf55_wifi import extract_metadata_from_filename, load_sample, ACTION_ID_TO_TYPE
from timenn.transforms.xrf55_combination import XRF55CombinationAugment

class FixedAugmentDataset(Dataset):
    def __init__(self, data_items, aug_type_str, num_aug=4, seed=42):
        """
        data_items: list of dicts {'path': p, 'label': l} (Original samples)
        aug_type_str: str, e.g. "shift+warp"
        num_aug: int, number of augmented copies per original sample
        seed: int, random seed for reproducibility
        """
        self.expanded_items = []
        self.augmenter = XRF55CombinationAugment(aug_type_str)
        
        # Set persistent seed for reproducibility of the DATASET construction
        rng = np.random.RandomState(seed)
        
        print(f"Generating Fixed Augmentation Dataset (1x Original + {num_aug}x Augmented)...")
        
        for item in tqdm(data_items, desc="Augmenting"):
            # 1. Add Original
            self.expanded_items.append({
                'path': item['path'],
                'label': item['label'],
                'augment': False
            })
            
            # 2. Add augmented copies
            # Store a per-copy seed so that lazy augmentation is deterministic.
            for i in range(num_aug):
                item_seed = rng.randint(0, 2**32 - 1)
                self.expanded_items.append({
                    'path': item['path'],
                    'label': item['label'],
                    'augment': True,
                    'seed': item_seed
                })
                
    def __len__(self):
        return len(self.expanded_items)
    
    def __getitem__(self, idx):
        item = self.expanded_items[idx]
        
        # Load
        data = load_sample(item['path'])
        if data is None:
             data = np.zeros((270, 1000), dtype=np.float32)
             
        # Augment if needed
        if item['augment']:
            # The augmenter uses numpy's global RNG; restore it afterwards.
            state = np.random.get_state()
            np.random.seed(item['seed'])
            
            try:
                data = self.augmenter(data)
            finally:
                np.random.set_state(state)
                
        data = torch.tensor(data, dtype=torch.float32)
        label = item['label']
        
        return data, label

def build_fixed_validation_dataset(root_dir, 
                                   aug_type_str,
                                   scenes=None,
                                   include_types=None):
    
    if scenes is None:
        scenes = ["Scene1", "Scene1-part2", "Scene2", "Scene3", "Scene4"]
        
    print("Scanning XRF55 WiFi files for Validation/Test...")
    
    valid_samples = []
    labels = []
    
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
            
            # Collect all validation/test items, then split by repetition.
            valid_samples.append({
                'path': f,
                'label': act,
                'rep': rep 
            })
            labels.append(act)
            
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)
    
    # Split items
    # Validation: Rep 15-17
    # Test: Rep 18+
    # Train: Rep 1-14 (handled by the training script)
    
    val_items_raw = []
    test_items_raw = []
    train_items_raw = []
    
    for i, s in enumerate(valid_samples):
        s['label'] = labels_encoded[i]
        r = s['rep']
        if r >= 18:
            test_items_raw.append(s)
        elif r >= 15:
            val_items_raw.append(s)
        else:
            train_items_raw.append(s)
            
    # Build Fixed Datasets
    val_ds = FixedAugmentDataset(val_items_raw, aug_type_str, num_aug=4, seed=1001)
    test_ds = FixedAugmentDataset(test_items_raw, aug_type_str, num_aug=4, seed=2002)
    
    # Return train items so main script can build a simple online-augment dataset
    return train_items_raw, val_ds, test_ds, num_classes, le

class OnlineAugmentDataset(Dataset):
    """
    Simple wrapper for online augmentation during training.
    """
    def __init__(self, data_items, aug_type_str):
        self.data_items = data_items
        self.augmenter = XRF55CombinationAugment(aug_type_str)
        
    def __len__(self):
        return len(self.data_items)
        
    def __getitem__(self, idx):
        item = self.data_items[idx]
        data = load_sample(item['path'])
        
        if data is None:
             data = np.zeros((270, 1000), dtype=np.float32)

        # distinct random augmentation every time
        data = self.augmenter(data)
        
        data = torch.tensor(data, dtype=torch.float32)
        label = item['label']
        
        return data, label
