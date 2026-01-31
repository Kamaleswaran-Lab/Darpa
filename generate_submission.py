#!/usr/bin/env python3
"""
Author: Md Hassanuzzaman
PhD Student, Duke University

Bootstrap Ensemble Submission Generator

Generates age group predictions using 10 bootstrap models with probability averaging.
Outputs predictions in OCA-SENTINEL format.
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import glob
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import model and data utilities
from src.models import AgeGroupPredictor
from src.data.preprocessing import _filter_denoise_signal, FS_HZ, LOWPASS_HZ, FILTER_ORDER


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: Device to load on
        
    Returns:
        model: Loaded model ready for inference
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate model with default config
    model_config = {
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 512,
        'fusion_dim': 256,
        'num_classes': 6,
        'dropout': 0.1,
        'max_len': 336,
        'pooling': 'mean'
    }
    
    model = AgeGroupPredictor(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def load_models_from_checkpoints(checkpoint_dir='checkpoints', device='cuda'):
    """
    Load all bootstrap models from checkpoint directory.
    Expects files named: best_model_bootstrap_1.pth, best_model_bootstrap_2.pth, etc.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        device: Device to load models on
        
    Returns:
        models: List of loaded models
    """
    models = []
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'best_model_bootstrap_*.pth')))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    for i, checkpoint_file in enumerate(checkpoint_files, 1):
        try:
            model = load_model_from_checkpoint(checkpoint_file, device)
            models.append(model)
            print(f"  ✓ Model {i}: {os.path.basename(checkpoint_file)}")
        except Exception as e:
            print(f"  ✗ Failed to load {os.path.basename(checkpoint_file)}: {e}")
    
    if not models:
        raise RuntimeError("No models were successfully loaded")
    
    return models


def collate_fn(batch):
    """Custom collate function for handling variable-length masks"""
    aorta_seqs, brach_seqs, aorta_masks, brach_masks = zip(*batch)
    
    # Stack sequences (they should all be the same length)
    aorta_batch = torch.stack(aorta_seqs, dim=0)  # [B, seq_len]
    brach_batch = torch.stack(brach_seqs, dim=0)  # [B, seq_len]
    aorta_mask_batch = torch.stack(aorta_masks, dim=0)  # [B, seq_len]
    brach_mask_batch = torch.stack(brach_masks, dim=0)  # [B, seq_len]
    
    return aorta_batch, brach_batch, aorta_mask_batch, brach_mask_batch


def load_csv_data(aorta_path, brach_path, batch_size=32, apply_filter=True, normalize=True, filter_config=None):
    """
    Load CSV test data with proper preprocessing (filtering and normalization).
    
    Args:
        aorta_path: Path to AortaP test data CSV
        brach_path: Path to BrachP test data CSV
        batch_size: Batch size for DataLoader
        apply_filter: Whether to apply Butterworth filter
        normalize: Whether to normalize data
        filter_config: Dict with filter_fs_hz, filter_lowpass_hz, filter_order
        
    Returns:
        DataLoader: Test data loader
        dict: Scalers used for normalization
        int: Number of samples
    """
    if filter_config is None:
        filter_config = {
            'filter_fs_hz': FS_HZ,
            'filter_lowpass_hz': LOWPASS_HZ,
            'filter_order': FILTER_ORDER
        }
    
    # Load CSV files
    print(f"Loading AortaP data from {os.path.basename(aorta_path)}...")
    df_aorta = pd.read_csv(aorta_path)
    
    print(f"Loading BrachP data from {os.path.basename(brach_path)}...")
    df_brach = pd.read_csv(brach_path)
    
    # Extract features (columns 1 to 336, excluding index)
    aorta_data = df_aorta.iloc[:, 1:337].values.astype(np.float32)
    brach_data = df_brach.iloc[:, 1:337].values.astype(np.float32)
    
    print(f"  - Aorta shape: {aorta_data.shape}")
    print(f"  - Brach shape: {brach_data.shape}")
    
    # Create masks for missing values (True = missing)
    aorta_mask = pd.isna(df_aorta.iloc[:, 1:337]).values
    brach_mask = pd.isna(df_brach.iloc[:, 1:337]).values
    
    # Apply filter and denoise
    if apply_filter:
        print("Applying Butterworth low-pass filter...")
        for i in range(len(aorta_data)):
            if i % 100 == 0:
                print(f"  Filtering sample {i}/{len(aorta_data)}")
            aorta_data[i] = _filter_denoise_signal(
                aorta_data[i], aorta_mask[i],
                fs=filter_config['filter_fs_hz'],
                lowpass_hz=filter_config['filter_lowpass_hz'],
                order=filter_config['filter_order']
            )
            brach_data[i] = _filter_denoise_signal(
                brach_data[i], brach_mask[i],
                fs=filter_config['filter_fs_hz'],
                lowpass_hz=filter_config['filter_lowpass_hz'],
                order=filter_config['filter_order']
            )
    
    # Fit scalers on test data (since we don't have training data)
    # In production, this would be fitted on training data and saved
    scalers = None
    if normalize:
        print("Normalizing data...")
        aorta_scaler = StandardScaler()
        brach_scaler = StandardScaler()
        
        # Fit on non-missing values
        aorta_valid = aorta_data[~aorta_mask].reshape(-1, 1)
        brach_valid = brach_data[~brach_mask].reshape(-1, 1)
        
        if len(aorta_valid) > 0:
            aorta_scaler.fit(aorta_valid)
        if len(brach_valid) > 0:
            brach_scaler.fit(brach_valid)
        
        scalers = {'aorta': aorta_scaler, 'brach': brach_scaler}
        
        # Transform data
        for i in range(len(aorta_data)):
            valid_aorta = ~aorta_mask[i]
            valid_brach = ~brach_mask[i]
            
            if valid_aorta.any():
                aorta_data[i, valid_aorta] = aorta_scaler.transform(
                    aorta_data[i, valid_aorta].reshape(-1, 1)
                ).flatten()
            
            if valid_brach.any():
                brach_data[i, valid_brach] = brach_scaler.transform(
                    brach_data[i, valid_brach].reshape(-1, 1)
                ).flatten()
    
    # Fill missing values with 0 (will be masked in attention)
    aorta_data = np.nan_to_num(aorta_data, nan=0.0)
    brach_data = np.nan_to_num(brach_data, nan=0.0)
    
    # Create simple dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, aorta_data, brach_data, aorta_mask, brach_mask):
            # Data shape: (N, seq_len) where each row is a time series
            self.aorta = torch.from_numpy(aorta_data).float()  # [N, seq_len]
            self.brach = torch.from_numpy(brach_data).float()  # [N, seq_len]
            self.aorta_mask = torch.from_numpy(aorta_mask).bool()  # [N, seq_len]
            self.brach_mask = torch.from_numpy(brach_mask).bool()  # [N, seq_len]
        
        def __len__(self):
            return len(self.aorta)
        
        def __getitem__(self, idx):
            # Return (seq_len,) tensors and masks
            return self.aorta[idx], self.brach[idx], self.aorta_mask[idx], self.brach_mask[idx]
    
    dataset = SimpleDataset(aorta_data, brach_data, aorta_mask, brach_mask)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    return loader, scalers, len(dataset)


def predict_ensemble(models, test_loader, method='averaging', device='cuda'):
    """
    Generate ensemble predictions from multiple models.
    
    Args:
        models: List of trained models
        test_loader: DataLoader for test data
        method: 'averaging' or 'voting'
        device: Device for inference
        
    Returns:
        predictions: Array of class predictions (0-5)
        probabilities: Array of prediction probabilities
    """
    all_predictions = []
    all_probabilities = []
    
    # Get predictions from each model
    for model_idx, model in enumerate(models):
        print(f"  Running inference with model {model_idx + 1}/{len(models)}...", end='', flush=True)
        
        model_preds = []
        model_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                aorta_data, brach_data, aorta_mask, brach_mask = batch
                
                aorta_data = aorta_data.to(device)
                brach_data = brach_data.to(device)
                aorta_mask = aorta_mask.to(device)
                brach_mask = brach_mask.to(device)
                
                # Forward pass
                outputs = model(aorta_data, brach_data, aorta_mask, brach_mask)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                model_probs.append(probs.cpu().numpy())
                model_preds.append(preds.cpu().numpy())
        
        model_preds_all = np.concatenate(model_preds)
        model_probs_all = np.concatenate(model_probs)
        
        all_predictions.append(model_preds_all)
        all_probabilities.append(model_probs_all)
        print(" ✓")
    
    # Combine predictions
    if method == 'averaging':
        # Average probabilities across models
        avg_probs = np.mean(all_probabilities, axis=0)
        ensemble_predictions = np.argmax(avg_probs, axis=1)
    elif method == 'voting':
        # Majority voting
        ensemble_predictions = np.zeros(len(all_predictions[0]), dtype=int)
        for i in range(len(all_predictions[0])):
            votes = [preds[i] for preds in all_predictions]
            ensemble_predictions[i] = Counter(votes).most_common(1)[0][0]
        avg_probs = np.mean(all_probabilities, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return ensemble_predictions, avg_probs


def main():
    parser = argparse.ArgumentParser(
        description='Generate Bootstrap Ensemble Predictions'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints',
        help='Directory containing model checkpoints (default: checkpoints)'
    )
    parser.add_argument(
        '--aorta_file',
        type=str,
        default='aortaP_test_data.csv',
        help='Path to AortaP test data CSV (default: aortaP_test_data.csv)'
    )
    parser.add_argument(
        '--brach_file',
        type=str,
        default='brachP_test_data.csv',
        help='Path to BrachP test data CSV (default: brachP_test_data.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='OCA-SENTINEL_output.json',
        help='Output file for predictions (default: OCA-SENTINEL_output.json)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='averaging',
        choices=['averaging', 'voting'],
        help='Ensemble method (default: averaging)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: cuda if available, else cpu)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for inference (default: 16)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Bootstrap Ensemble Submission Generator")
    print("=" * 80)
    print()
    
    try:
        # Load models
        print(f"Loading models from {args.checkpoint_dir}...")
        models = load_models_from_checkpoints(args.checkpoint_dir, args.device)
        print(f"✓ Loaded {len(models)} models\n")
        
        # Load test data
        print(f"Loading test data...")
        test_loader, scalers, n_samples = load_csv_data(
            args.aorta_file, 
            args.brach_file, 
            batch_size=args.batch_size,
            apply_filter=True,
            normalize=True
        )
        print(f"✓ Loaded {n_samples} test samples\n")
        
        # Make predictions
        print(f"Making ensemble predictions using {args.method}...")
        predictions, probabilities = predict_ensemble(
            models,
            test_loader,
            method=args.method,
            device=args.device
        )
        print(f"✓ Generated {len(predictions)} predictions\n")
        
        # Convert to OCA-SENTINEL format with integer keys
        # Keys are numeric indices (0, 1, 2, ...), values are integer class predictions
        # Note: JSON will convert integer keys to strings
        predictions_dict = {i: int(pred) for i, pred in enumerate(predictions)}
        
        # Save output with proper formatting (indent=2, sort_keys=True for numeric ascending order)
        with open(args.output, 'w') as f:
            json.dump(predictions_dict, f, indent=2, sort_keys=True)
        
        print(f"✓ Predictions saved to {args.output}")
        print(f"  - Total samples: {len(predictions_dict)}")
        print(f"  - Format: OCA-SENTINEL")
        print(f"  - Classes: 0=20s, 1=30s, 2=40s, 3=50s, 4=60s, 5=70s")
        print(f"  - Ensemble method: {args.method}")
        print(f"  - Models used: {len(models)}")
        print()
        
        # Print sample predictions
        print(f"Sample predictions (first 10):")
        for i in range(min(10, len(predictions_dict))):
            print(f"  Sample {i}: Class {predictions_dict[i]}")
        
        print()
        print("=" * 80)
        print("✓ Submission generation completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
