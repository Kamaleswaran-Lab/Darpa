"""
Main package initialization for DARPA Age Prediction
"""

__version__ = "1.0.0"
__author__ = "DARPA Challenge Team"

from .models import AgeGroupPredictor, create_model
from .data import MultimodalAgeDataset, create_dataloaders
from .utils import load_config

__all__ = [
    "AgeGroupPredictor",
    "create_model",
    "MultimodalAgeDataset",
    "create_dataloaders",
    "load_config",
]
