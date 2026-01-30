#!/usr/bin/env python3
"""
Generate DARPA Challenge Submission Predictions

This script generates predictions on the evaluation dataset in the required
JSON format for DARPA challenge submission.

Usage:
    python generate_submission.py --model MODEL_PATH --aorta-data AORTA_CSV --brach-data BRACH_CSV --output OUTPUT_JSON

Example:
    python generate_submission.py \
        --model best_model.pth \
        --aorta-data aortaP_test_data.csv \
        --brach-data brachP_test_data.csv \
        --output OCA-SENTINEL_output.json \
        --device auto \
        --team-name OCA-SENTINEL
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
import pandas as pd
import numpy as np
import warnings

# This suppresses all warnings
warnings.filterwarnings("ignore")


# Device utilities
def get_device(device_str: str) -> torch.device:
    """Get and validate the device.
    
    Args:
        device_str: Device string ('cuda', 'cpu', or 'auto')
        
    Returns:
        torch.device: Validated device
    """
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    # Validate device
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            print("⚠️  CUDA requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
        else:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA available: {device_count} GPU(s) - {device_name}")
    else:
        print("✓ Using CPU for inference")
    
    return device

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.age_group_predictor import AgeGroupPredictor
from data.dataset import MultimodalAgeDataset
from utils.config import load_config


def load_model(model_path: str, config: dict = None, device: torch.device = None) -> AgeGroupPredictor:
    """Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint (.pth file)
        config: Optional config dict for model architecture
        device: Device to load model onto (default: cpu)
        
    Returns:
        AgeGroupPredictor: Loaded model in eval mode
    """
    if device is None:
        device = torch.device('cpu')
        
    print(f"Loading model from {model_path}...")
    
    # Load checkpoint - always load to CPU first, then move to target device
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get config from checkpoint if not provided
    if config is None:
        config = checkpoint.get('config', {})
    
    # Create model
    model = AgeGroupPredictor(
        d_model=config.get('d_model', 128),
        nhead=config.get('nhead', 8),
        num_layers=config.get('num_layers', 4),
        num_classes=config.get('num_classes', 6),
        dropout=config.get('dropout', 0.1)
    )
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel checkpoints (remove 'module.' prefix)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    
    # Move to target device
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded successfully on {device}")
    return model


def load_evaluation_data(aorta_path: str, brach_path: str, device: torch.device) -> torch.utils.data.DataLoader:
    """Load and preprocess evaluation dataset.
    
    Args:
        aorta_path: Path to AortaP CSV
        brach_path: Path to BrachP CSV
        device: Device for inference
        
    Returns:
        DataLoader: Evaluation data loader
    """
    print(f"Loading evaluation data...")
    print(f"  AortaP: {aorta_path}")
    print(f"  BrachP: {brach_path}")
    
    # Create dataset with preprocessing
    dataset = MultimodalAgeDataset(
        aorta_path=aorta_path,
        brach_path=brach_path,
        normalize=True,
        fit_scaler=None,  # No pre-fitted scaler for test data
        apply_filter=True
    )
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✓ Loaded {len(dataset)} samples")
    return dataloader


def preprocess_signals(aorta_signal: np.ndarray, brach_signal: np.ndarray) -> tuple:
    """Preprocess signals for model input.
    
    Args:
        aorta_signal: AortaP signal array
        brach_signal: BrachP signal array
        
    Returns:
        tuple: (normalized_aorta, normalized_brach)
    """
    # Signals are already preprocessed by MultimodalAgeDataset
    aorta_norm = aorta_signal / (np.std(aorta_signal[~np.isnan(aorta_signal)]) + 1e-8)
    brach_norm = brach_signal / (np.std(brach_signal[~np.isnan(brach_signal)]) + 1e-8)
    return aorta_norm, brach_norm


@torch.no_grad()
def generate_predictions(
    model: AgeGroupPredictor,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[int, int]:
    """Generate predictions for all samples.
    
    Args:
        model: Trained model
        dataloader: DataLoader with evaluation data
        device: torch.device to run inference on
        
    Returns:
        dict: {subject_index: predicted_class}
    """
    print(f"Generating predictions on {device}...")
    model.eval()
    
    predictions = {}
    sample_idx = 0
    total_batches = len(dataloader)
    
    try:
        # Process batches
        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch
            aorta, brach, aorta_mask, brach_mask = batch
            
            # Move to device
            aorta = aorta.to(device)
            brach = brach.to(device)
            aorta_mask = aorta_mask.to(device)
            brach_mask = brach_mask.to(device)
            
            # Get predictions
            logits = model(aorta, brach, aorta_mask, brach_mask)
            predicted_classes = logits.argmax(dim=1).cpu().numpy()
            
            # Store predictions
            for pred_class in predicted_classes:
                predictions[int(sample_idx)] = int(pred_class)
                sample_idx += 1
            
            # Print progress
            progress_pct = ((batch_idx + 1) / total_batches) * 100
            print(f"  Progress: {batch_idx + 1}/{total_batches} batches ({progress_pct:.1f}%) - {sample_idx} samples processed", end='\r')
    
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        raise
    
    print(f"\n✓ Generated {len(predictions)} predictions")
    return predictions


def validate_predictions(predictions: Dict[int, int]) -> bool:
    """Validate predictions meet submission requirements.
    
    Args:
        predictions: Dictionary of predictions
        
    Returns:
        bool: True if valid
    """
    print("Validating predictions...")
    
    # Check count
    if len(predictions) != 875:
        print(f"❌ Wrong count: expected 875, got {len(predictions)}")
        return False
    
    # Check indices
    expected_indices = set(range(875))
    actual_indices = set(predictions.keys())
    
    if expected_indices != actual_indices:
        missing = expected_indices - actual_indices
        extra = actual_indices - expected_indices
        if missing:
            print(f"❌ Missing indices: {sorted(missing)[:10]}")
        if extra:
            print(f"❌ Extra indices: {sorted(extra)[:10]}")
        return False
    
    # Check values
    valid_classes = {0, 1, 2, 3, 4, 5}
    invalid_values = []
    
    for idx, value in predictions.items():
        if not isinstance(value, int):
            invalid_values.append(f"Index {idx}: not int ({type(value).__name__})")
        elif value not in valid_classes:
            invalid_values.append(f"Index {idx}: value {value} not in {{0,1,2,3,4,5}}")
    
    if invalid_values:
        print(f"❌ Invalid values found:")
        for error in invalid_values[:10]:
            print(f"  - {error}")
        return False
    
    print("✓ Predictions are valid")
    return True


def save_predictions(predictions: Dict[int, int], output_file: str) -> None:
    """Save predictions to JSON file in submission format.
    
    Args:
        predictions: Dictionary of predictions
        output_file: Output JSON file path
    """
    print(f"Saving predictions to {output_file}...")
    
    # Ensure all keys and values are integers
    formatted_predictions = {int(k): int(v) for k, v in predictions.items()}
    
    # Save with nice formatting
    with open(output_file, 'w') as f:
        json.dump(formatted_predictions, f, indent=2, sort_keys=True)
    
    file_size = Path(output_file).stat().st_size
    print(f"✓ Saved {len(formatted_predictions)} predictions ({file_size:,} bytes)")


def print_statistics(predictions: Dict[int, int]) -> None:
    """Print statistics about predictions.
    
    Args:
        predictions: Dictionary of predictions
    """
    print("\n📊 Prediction Statistics:")
    print("=" * 50)
    
    # Class distribution
    class_counts = {i: 0 for i in range(6)}
    for value in predictions.values():
        if value in class_counts:
            class_counts[value] += 1
    
    print("Class Distribution:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / len(predictions) * 100) if len(predictions) > 0 else 0
        print(f"  Class {class_id}: {count:4d} ({percentage:5.1f}%)")
    
    print("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate DARPA challenge submission predictions"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pth)'
    )
    parser.add_argument(
        '--aorta-data',
        type=str,
        required=True,
        help='Path to AortaP test data CSV'
    )
    parser.add_argument(
        '--brach-data',
        type=str,
        required=True,
        help='Path to BrachP test data CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file path (e.g., myteam_output.json)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file (default: config/config.yaml)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Device to use for inference: auto (default), cuda, or cpu'
    )
    parser.add_argument(
        '--team-name',
        type=str,
        help='Team name (for filename validation)'
    )
    
    args = parser.parse_args()
    
    print("🚀 DARPA Challenge Prediction Generator")
    print("=" * 70)
    
    # Get and validate device
    device = get_device(args.device)
    print()
    
    # Validate output filename
    if args.team_name:
        expected_name = f"{args.team_name}_output.json"
        if not args.output.endswith(expected_name):
            print(f"⚠️  Warning: Output file should be named '{expected_name}'")
            print(f"   Current: {args.output}\n")
    
    # Load config
    config = load_config(args.config) if Path(args.config).exists() else {}
    
    # Load model
    model = load_model(args.model, config, device)
    
    # Validate data file paths
    if not Path(args.aorta_data).exists():
        print(f"❌ Error: AortaP file not found: {args.aorta_data}")
        sys.exit(1)
    
    if not Path(args.brach_data).exists():
        print(f"❌ Error: BrachP file not found: {args.brach_data}")
        sys.exit(1)
    
    # Load data and create dataloader
    dataloader = load_evaluation_data(args.aorta_data, args.brach_data, device)
    
    # Generate predictions
    predictions = generate_predictions(model, dataloader, device)
    
    # Validate predictions
    if not validate_predictions(predictions):
        print("❌ Prediction validation failed!")
        sys.exit(1)
    
    # Print statistics
    print_statistics(predictions)
    
    # Save predictions
    save_predictions(predictions, args.output)
    
    print("\n✅ Prediction generation complete!")

if __name__ == "__main__":
    main()
