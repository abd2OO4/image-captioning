"""Image Captioning Module

This package contains utilities for image captioning including:
- Feature extraction from images
- Dataset loading and preprocessing
- Model building and training
- Caption generation
"""

from .dataset import load_dataset
from .feature_extractor import create_feature_extractor, extract_features
from .model import build_model
from .generator import CaptionGenerator
from .preprocessing import clean_caption

__all__ = [
    'load_dataset',
    'create_feature_extractor',
    'extract_features',
    'build_model',
    'CaptionGenerator',
    'clean_caption',
]
