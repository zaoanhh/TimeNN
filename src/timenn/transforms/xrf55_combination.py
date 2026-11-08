
import numpy as np
from scipy.interpolate import interp1d

class XRF55CombinationAugment(object):
    def __init__(self, aug_type_str, warp_params=None):
        """
        aug_type_str: str, e.g., "shift", "shift+warp", "shift+mask+warp"
        warp_params: dict, optional parameters for warp (sigma, knots)
        """
        self.aug_types = aug_type_str.split('+')
        self.warp_params = warp_params if warp_params is not None else {}
        self.seed = None
        # Seeding for fixed augmentation is handled by the dataset builder.

    def __call__(self, x):
        """
        x: [C, T] numpy array
        """
        x_aug = x.copy()
        
        # Apply augmentations in order
        for aug in self.aug_types:
            if aug == 'shift':
                x_aug = self.shift_zero_padding(x_aug)
            elif aug == 'mask':
                x_aug = self.time_mask(x_aug)
            elif aug == 'warp':
                x_aug = self.time_warp(x_aug)
                
        return x_aug

    def shift_zero_padding(self, x, max_shift_ratio=0.2):
        C, T = x.shape
        shift_range = int(T * max_shift_ratio)
        shift = np.random.randint(-shift_range, shift_range + 1)
        
        if shift == 0:
            return x
        
        x_new = np.zeros_like(x)
        if shift > 0:
            x_new[:, shift:] = x[:, :-shift]
        else:
            x_new[:, :shift] = x[:, -shift:]
            
        return x_new

    def time_mask(self, x, mask_prob=0.5, max_mask_ratio=0.1):
        C, T = x.shape
        x_new = x.copy()
        device_splits = [0, 90, 180, 270]
        
        for i in range(3):
            if np.random.rand() < mask_prob:
                start = device_splits[i]
                end = device_splits[i+1]
                if end > C: end = C
                
                mask_len = int(T * np.random.uniform(0, max_mask_ratio))
                mask_start = np.random.randint(0, T - mask_len)
                x_new[start:end, mask_start:mask_start+mask_len] = 0.0
        return x_new

    def time_warp(self, x, sigma=0.2, knots=4):
        sigma = self.warp_params.get('sigma', sigma)
        knots = self.warp_params.get('knots', knots)

        C, T = x.shape
        
        # Interpolate a monotone warp from randomly displaced knots.
        old_time = np.linspace(0, T-1, knots+2)
        warp_steps = np.random.randn(knots+2) * sigma
        new_time = old_time + warp_steps
        
        new_time[0] = 0
        new_time[-1] = T-1
        new_time = np.sort(new_time)
        
        f_time = interp1d(new_time, old_time, kind='linear', fill_value="extrapolate")
        sample_indices = f_time(np.arange(T))
        sample_indices = np.clip(sample_indices, 0, T-1)
        
        x_new = np.zeros_like(x)
        for c in range(C):
            x_new[c] = np.interp(sample_indices, np.arange(T), x[c])
            
        return x_new
