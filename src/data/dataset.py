"""
Dataset classes for multimodal age group prediction
Handles missing data using attention masking (AIM-inspired approach)
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

from .preprocessing import _filter_denoise_signal, FS_HZ, LOWPASS_HZ, FILTER_ORDER


class MultimodalAgeDataset(Dataset):
    """Dataset for multimodal age group prediction"""
    def __init__(self, aorta_path, brach_path, normalize=True, fit_scaler=None,
                 apply_filter=True, filter_fs_hz=FS_HZ, filter_lowpass_hz=LOWPASS_HZ, 
                 filter_order=FILTER_ORDER):
        """
        Args:
            aorta_path: Path to AortaP CSV file
            brach_path: Path to BrachP CSV file
            normalize: Whether to normalize the data
            fit_scaler: If provided, use this scaler instead of fitting new one
            apply_filter: Whether to apply Butterworth low-pass filter (default: True)
            filter_fs_hz: Sampling frequency in Hz (default: 500)
            filter_lowpass_hz: Low-pass cutoff frequency in Hz (default: 25)
            filter_order: Filter order (default: 4)
        """
        # Load data
        self.df_aorta = pd.read_csv(aorta_path)
        self.df_brach = pd.read_csv(brach_path)
        
        # Extract features (columns 1 to 336, excluding index)
        self.aorta_data = self.df_aorta.iloc[:, 1:337].values.astype(np.float32)
        self.brach_data = self.df_brach.iloc[:, 1:337].values.astype(np.float32)
        
        # Extract targets (if available)
        if 'target' in self.df_aorta.columns:
            self.targets = self.df_aorta['target'].values.astype(np.int64)
        else:
            self.targets = None
        
        # Create masks for missing values (True = missing)
        self.aorta_mask = pd.isna(self.df_aorta.iloc[:, 1:337]).values
        self.brach_mask = pd.isna(self.df_brach.iloc[:, 1:337]).values
        
        # Filter and denoise (Butterworth low-pass); only filter valid segments, missing remain NaN
        if apply_filter:
            for i in range(len(self.aorta_data)):
                self.aorta_data[i] = _filter_denoise_signal(
                    self.aorta_data[i], self.aorta_mask[i], fs=filter_fs_hz,
                    lowpass_hz=filter_lowpass_hz, order=filter_order
                )
                self.brach_data[i] = _filter_denoise_signal(
                    self.brach_data[i], self.brach_mask[i], fs=filter_fs_hz,
                    lowpass_hz=filter_lowpass_hz, order=filter_order
                )
        
        # Normalize data
        self.normalize = normalize
        if normalize:
            if fit_scaler is None:
                # Fit scalers on non-missing values
                self.aorta_scaler = StandardScaler()
                self.brach_scaler = StandardScaler()
                
                # Fit on all non-missing values
                aorta_flat = self.aorta_data[~self.aorta_mask].reshape(-1, 1)
                brach_flat = self.brach_data[~self.brach_mask].reshape(-1, 1)
                
                self.aorta_scaler.fit(aorta_flat)
                self.brach_scaler.fit(brach_flat)
            else:
                self.aorta_scaler = fit_scaler['aorta']
                self.brach_scaler = fit_scaler['brach']
            
            # Transform data (missing values remain NaN)
            for i in range(len(self.aorta_data)):
                valid_aorta = ~self.aorta_mask[i]
                valid_brach = ~self.brach_mask[i]
                
                if valid_aorta.any():
                    self.aorta_data[i, valid_aorta] = self.aorta_scaler.transform(
                        self.aorta_data[i, valid_aorta].reshape(-1, 1)
                    ).flatten()
                
                if valid_brach.any():
                    self.brach_data[i, valid_brach] = self.brach_scaler.transform(
                        self.brach_data[i, valid_brach].reshape(-1, 1)
                    ).flatten()
        
        # Fill missing values with 0 (will be masked in attention)
        self.aorta_data = np.nan_to_num(self.aorta_data, nan=0.0)
        self.brach_data = np.nan_to_num(self.brach_data, nan=0.0)
    
    def __len__(self):
        return len(self.aorta_data)
    
    def __getitem__(self, idx):
        aorta = torch.FloatTensor(self.aorta_data[idx])
        brach = torch.FloatTensor(self.brach_data[idx])
        aorta_m = torch.BoolTensor(self.aorta_mask[idx])
        brach_m = torch.BoolTensor(self.brach_mask[idx])
        
        if self.targets is not None:
            target = torch.LongTensor([self.targets[idx]])[0]
            return aorta, brach, aorta_m, brach_m, target
        else:
            return aorta, brach, aorta_m, brach_m
    
    def get_scalers(self):
        """Return fitted scalers for use on test data"""
        if self.normalize:
            return {'aorta': self.aorta_scaler, 'brach': self.brach_scaler}
        return None


def create_dataloaders(
    aorta_train_path,
    brach_train_path,
    batch_size=32,
    num_workers=0,
    normalize=True,
    apply_filter=True,
    filter_fs_hz=FS_HZ,
    filter_lowpass_hz=LOWPASS_HZ,
    filter_order=FILTER_ORDER,
    val_split=0.2,
    seed=42,
):
    """
    Create train and validation dataloaders (no external test set).
    Splits the provided training set into train/val using the given ratio.
    """
    # Create full training dataset
    full_dataset = MultimodalAgeDataset(
        aorta_train_path,
        brach_train_path,
        normalize=normalize,
        apply_filter=apply_filter,
        filter_fs_hz=filter_fs_hz,
        filter_lowpass_hz=filter_lowpass_hz,
        filter_order=filter_order,
    )

    # Get scalers from training data
    scalers = full_dataset.get_scalers()

    # Train/val split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # DataLoaders
    pin_mem = True if torch.cuda.is_available() else False
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    return train_loader, val_loader, scalers


if __name__ == "__main__":
    # Test data loader
    train_loader, test_loader, _ = create_dataloaders(
        '/hpc/group/kamaleswaranlab/DARPA_Challenge/data/aortaP_train_data.csv',
        '/hpc/group/kamaleswaranlab/DARPA_Challenge/data/brachP_train_data.csv',
        '/hpc/group/kamaleswaranlab/DARPA_Challenge/data/aortaP_test_data.csv',
        '/hpc/group/kamaleswaranlab/DARPA_Challenge/data/brachP_test_data.csv',
        batch_size=4
    )
    
    # Test batch
    for batch in train_loader:
        aorta, brach, aorta_m, brach_m, target = batch
        print(f"Batch shape - Aorta: {aorta.shape}, Brach: {brach.shape}")
        print(f"Mask shape - Aorta: {aorta_m.shape}, Brach: {brach_m.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Missing values in batch - Aorta: {aorta_m.sum()}, Brach: {brach_m.sum()}")
        break
