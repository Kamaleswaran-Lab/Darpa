"""Signal preprocessing and filtering utilities"""

import numpy as np
from scipy.signal import butter, filtfilt

# Filter/denoise settings (shared with plotting)
FS_HZ = 500
LOWPASS_HZ = 25
FILTER_ORDER = 4


def _filter_denoise_signal(x: np.ndarray, mask: np.ndarray, fs: int = FS_HZ,
                           lowpass_hz: float = LOWPASS_HZ, order: int = FILTER_ORDER) -> np.ndarray:
    """Butterworth low-pass filter. Only filter valid (non-missing) segments; missing values remain NaN."""
    x = np.asarray(x, dtype=np.float64)
    out = x.copy()
    valid = ~mask
    
    # If all values are missing, return as-is
    if not np.any(valid):
        return out.astype(np.float32)
    
    # Only filter non-missing values; missing positions remain NaN
    valid_indices = np.where(valid)[0]
    if len(valid_indices) < 2:
        # Need at least 2 points to filter
        return out.astype(np.float32)
    
    # Extract valid segment
    valid_values = x[valid_indices]
    
    # Apply filter only to valid segment
    nyq = 0.5 * fs
    cut = min(lowpass_hz / nyq, 0.99)
    b, a = butter(order, cut, btype="low")
    filtered_valid = filtfilt(b, a, valid_values)
    
    # Put filtered values back at valid positions, keep NaN at missing positions
    out[valid_indices] = filtered_valid
    out[mask] = np.nan
    
    return out.astype(np.float32)
