"""
Synthetic data generation for SfM pipeline evaluation.

This module provides tools to generate synthetic scenes with known ground truth
for systematic evaluation of the SfM pipeline components.
"""

from .ground_truth import GroundTruthData, SceneGroundTruth, CameraGroundTruth, save_ground_truth, load_ground_truth
from .scene_generator import SceneGenerator
from .camera_generator import CameraGenerator
from .renderer import SyntheticRenderer
from .evaluation import SyntheticEvaluator

__all__ = [
    'GroundTruthData',
    'SceneGroundTruth',
    'CameraGroundTruth',
    'save_ground_truth',
    'load_ground_truth',
    'SceneGenerator',
    'CameraGenerator',
    'SyntheticRenderer',
    'SyntheticEvaluator'
]