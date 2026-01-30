"""
Utility to load configuration from YAML file
"""

import yaml
import os

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file and convert to dict format"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert nested structure to flat config dict expected by train_model
    preprocessing = config_dict.get('preprocessing', {})
    config = {
        'aorta_train_path': config_dict['data']['aorta_train_path'],
        'brach_train_path': config_dict['data']['brach_train_path'],
        'aorta_test_path': config_dict['data'].get('aorta_test_path'),
        'brach_test_path': config_dict['data'].get('brach_test_path'),
        'model_config': {
            'd_model': config_dict['model']['d_model'],
            'nhead': config_dict['model']['nhead'],
            'num_layers': config_dict['model']['num_layers'],
            'dim_feedforward': config_dict['model']['dim_feedforward'],
            'fusion_dim': config_dict['model']['fusion_dim'],
            'num_classes': config_dict['model']['num_classes'],
            'dropout': config_dict['model']['dropout'],
            'max_len': config_dict['model']['max_len'],
            'pooling': config_dict['model']['pooling']
        },
        'batch_size': config_dict['training']['batch_size'],
        'learning_rate': config_dict['training']['learning_rate'],
        'weight_decay': config_dict['training']['weight_decay'],
        'num_epochs': config_dict['training']['num_epochs'],
        'max_samples': config_dict['training'].get('max_samples'),
        'val_split': config_dict['training'].get('val_split', 0.2),
        'split_seed': config_dict['training'].get('split_seed', 42),
        'train_ratio': config_dict['training'].get('train_ratio', 0.8),
        'val_ratio': config_dict['training'].get('val_ratio', 0.1),
        'test_ratio': config_dict['training'].get('test_ratio', 0.1),
        'device_ids': config_dict['gpu']['device_ids'],
        'num_workers': config_dict['gpu']['num_workers'],
        'checkpoint_dir': config_dict['output']['checkpoint_dir'],
        'log_dir': config_dict['output']['log_dir'],
        # Preprocessing/filtering parameters
        'apply_filter': preprocessing.get('apply_filter', True),
        'filter_fs_hz': preprocessing.get('filter_fs_hz', 500),
        'filter_lowpass_hz': preprocessing.get('filter_lowpass_hz', 25),
        'filter_order': preprocessing.get('filter_order', 4)
    }
    
    return config
