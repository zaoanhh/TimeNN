
import numpy as np
from scipy.interpolate import interp1d

class XRF55Augment(object):
    def __init__(self, aug_type):
        """
        aug_type: str, one of ['shift', 'mask', 'warp']
        """
        self.aug_type = aug_type

    def __call__(self, x):
        """
        x: [C, T] numpy array
        """
        if self.aug_type == 'shift':
            return self.shift_zero_padding(x)
        elif self.aug_type == 'mask':
            return self.time_mask(x)
        elif self.aug_type == 'warp':
            return self.time_warp(x)
        else:
            return x

    def shift_zero_padding(self, x, max_shift_ratio=0.2):
        """
        1. Zero-padding data translation. All channels share a translation factor.
        x: [C, T]
        """
        C, T = x.shape
        # Shift range: [-T*ratio, T*ratio]
        shift_range = int(T * max_shift_ratio)
        shift = np.random.randint(-shift_range, shift_range + 1)
        
        if shift == 0:
            return x
        
        x_new = np.zeros_like(x)
        if shift > 0:
            # Shift right, pad left with zeros
            x_new[:, shift:] = x[:, :-shift]
        else:
            # Shift left, pad right with zeros
            x_new[:, :shift] = x[:, -shift:]
            
        return x_new

    def time_mask(self, x, mask_prob=0.1, max_mask_ratio=0.1):
        """
        2. Time mask. Same device shares a mask sequence (3 devices).
        x: [C, T]. 
        C=270. 0-89: Dev1, 90-179: Dev2, 180-269: Dev3.
        """
        C, T = x.shape
        x_new = x.copy()
        
        # Define device ranges
        device_splits = [0, 90, 180, 270]
        
        for i in range(3):
            if np.random.rand() < mask_prob:
                # Generate mask for this device
                start = device_splits[i]
                end = device_splits[i+1]
                
                if end > C: break
                
                mask_len = int(T * np.random.uniform(0, max_mask_ratio))
                mask_start = np.random.randint(0, T - mask_len)
                
                x_new[start:end, mask_start:mask_start+mask_len] = 0.0
                
        return x_new

    def time_warp(self, x, sigma=0.2, knots=4):
        """
        3. Time warp. All channels share the same warp.
        x: [C, T]
        """
        C, T = x.shape
        
        # Build a monotone warp curve from a few randomly displaced knots.
        old_time = np.linspace(0, T - 1, knots + 2)
        warp_steps = np.random.randn(knots + 2) * sigma
        new_time = old_time + warp_steps
        new_time[0] = 0
        new_time[-1] = T - 1
        new_time = np.sort(new_time)

        # Map every output index to a (fractional) index in the input.
        f_time = interp1d(new_time, old_time, kind="linear", fill_value="extrapolate")
        sample_indices = f_time(np.arange(T))
        
        # Clip to range
        sample_indices = np.clip(sample_indices, 0, T-1)
        
        # Apply interpolation for data
        x_new = np.zeros_like(x)
        for c in range(C):
            x_new[c] = np.interp(sample_indices, np.arange(T), x[c])
            
        return x_new

