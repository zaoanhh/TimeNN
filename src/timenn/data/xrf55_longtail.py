
import os
import glob
from sklearn.preprocessing import LabelEncoder
from timenn.data.xrf55_fixed_augment import FixedAugmentDataset
from timenn.data.xrf55_wifi import extract_metadata_from_filename

def build_longtail_dataset(root_dir, 
                           aug_type_str,
                           scenes=None):
    
    if scenes is None:
        scenes = ["Scene1", "Scene1-part2"]
        
    print(f"Scanning XRF55 WiFi files for Long-Tail Dataset in {scenes}...")
    
    valid_samples = []
    labels = []
    
    # Head: Actions 1-30 (All 30 users)
    # Tail: Actions 31-55 (First 3 users: 1, 2, 3)
    
    for scene in scenes:
        scene_path = os.path.join(root_dir, scene)
        wifi_path = os.path.join(scene_path, 'WiFi')
        if not os.path.exists(wifi_path): 
            print(f"Warning: {wifi_path} does not exist.")
            continue
        
        files = glob.glob(os.path.join(wifi_path, '*.npy'))
        for f in files:
            fname = os.path.basename(f)
            vol, act, rep = extract_metadata_from_filename(fname)
            
            if vol is None: continue
            
            # Long-Tail Logic
            if 1 <= act <= 30:
                # Head: Keep all users
                pass
            elif 31 <= act <= 55:
                # Tail: Keep only users 1, 2, 3
                if vol > 3:
                     continue
            else:
                # Out of range
                continue
                
            # Split repetitions: 1-14 train, 15-17 validation, 18-20 test.
            
            valid_samples.append({
                'path': f,
                'label': act,
                'rep': rep,
                'vol': vol
            })
            labels.append(act)
            
    # Encode labels
    # Labels are encoded over the actions present in this split.
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)
    print(f"Found {len(valid_samples)} samples across {num_classes} classes.")
    
    # Split items
    train_items_raw = []
    val_items_raw = []
    test_items_raw = []
    
    for i, s in enumerate(valid_samples):
        s['label'] = labels_encoded[i] # Update to encoded
        r = s['rep']
        
        if 1 <= r <= 14:
            train_items_raw.append(s)
        elif 15 <= r <= 17:
             val_items_raw.append(s)
        elif 18 <= r <= 20:
             test_items_raw.append(s)
        else:
            # Repetitions beyond 20 are not used.
            pass
             
    print(f"Split counts - Train: {len(train_items_raw)}, Val: {len(val_items_raw)}, Test: {len(test_items_raw)}")
    
    # Build Fixed Datasets for Val/Test (1x Original + 4x Aug = 5x Total)
    # Val
    val_ds = FixedAugmentDataset(val_items_raw, aug_type_str, num_aug=4, seed=1001)
    
    # Test
    test_ds = FixedAugmentDataset(test_items_raw, aug_type_str, num_aug=4, seed=2002)
    
    # Train items returned raw for OnlineAugmentDataset
    return train_items_raw, val_ds, test_ds, num_classes, le
