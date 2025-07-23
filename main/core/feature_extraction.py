"""
Feature extraction using LightGlue SuperPoint.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from tqdm import tqdm

try:
    from lightglue import SuperPoint
    from lightglue.utils import load_image
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False
    raise ImportError("LightGlue not available. Install with: pip install lightglue")

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Feature extraction using LightGlue SuperPoint.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.max_keypoints = config.feature_extractor.max_keypoints
        self.detection_threshold = config.feature_extractor.detection_threshold
        
        # Initialize SuperPoint extractor
        self.extractor = SuperPoint(max_num_keypoints=self.max_keypoints).eval().to(self.device)
        
        logger.info(f"FeatureExtractor initialized with SuperPoint on {self.device}")
    
    def _preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Preprocess a single image for feature extraction.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor [C, H, W]
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image using LightGlue utilities
        image = load_image(image_path)
        
        # Convert RGB to grayscale if needed (SuperPoint expects 1 channel)
        if image.shape[0] == 3:  # RGB image
            # Convert RGB to grayscale using standard luminance weights
            image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            image = image.unsqueeze(0)  # Add channel dimension back
        
        # Resize if needed
        max_size = self.config.io.max_image_size
        if max(image.shape[-2:]) > max_size:
            scale = max_size / max(image.shape[-2:])
            new_size = (int(image.shape[-2] * scale), int(image.shape[-1] * scale))
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=new_size, mode='bilinear', align_corners=False
            ).squeeze(0)
        
        return image
    
    def extract_features_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict]:
        """
        Extract features from multiple images using batch processing.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of feature dictionaries
        """
        if not image_paths:
            return []
        
        # Preprocess all images
        images = []
        valid_paths = []
        
        for image_path in image_paths:
            try:
                image = self._preprocess_image(image_path)
                images.append(image)
                valid_paths.append(Path(image_path))
            except Exception as e:
                logger.error(f"Failed to preprocess {image_path}: {e}")
                continue
        
        if not images:
            logger.warning("No valid images to process")
            return []
        
        # Stack images into batch tensor [B, C, H, W]
        batch_tensor = torch.stack(images).to(self.device)
        
        # Extract features for the entire batch
        with torch.no_grad():
            batch_features = self.extractor({'image': batch_tensor})
        
        # Process results for each image in the batch
        features_list = []
        for i, image_path in enumerate(valid_paths):
            keypoints = batch_features['keypoints'][i].cpu().numpy()
            descriptors = batch_features['descriptors'][i].cpu().numpy()
            scores = batch_features['keypoint_scores'][i].cpu().numpy()
            
            features_list.append({
                'keypoints': keypoints,
                'descriptors': descriptors,
                'scores': scores,
                'image_path': str(image_path),
                'image_size': images[i].shape[-2:],
                'num_features': len(keypoints),
                'extractor_type': 'superpoint'
            })
        
        logger.info(f"Successfully extracted features from {len(features_list)}/{len(image_paths)} images")
        return features_list
    
    def extract_features_single(self, image_path: Union[str, Path]) -> Dict:
        """
        Extract features from a single image using SuperPoint.
        This is a convenience wrapper around extract_features_batch.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing keypoints, descriptors, and metadata
        """
        batch_results = self.extract_features_batch([image_path])
        if not batch_results:
            raise RuntimeError(f"Failed to extract features from {image_path}")
        return batch_results[0]
    
    def visualize_features(self, image_path: Union[str, Path], features: Dict, 
                          output_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Visualize extracted features on the image.
        
        Args:
            image_path: Path to the original image
            features: Features dictionary from extract_features_single
            output_path: Optional path to save the visualization
            
        Returns:
            Image with features drawn
        """
        # Load image at original size
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Calculate scale factor used during preprocessing
        max_size = self.config.io.max_image_size
        original_max_dim = max(image.shape[:2])
        scale_factor = 1.0
        
        if original_max_dim > max_size:
            scale_factor = original_max_dim / max_size
        
        # Scale up keypoints to original image coordinates
        keypoints = features['keypoints'] * scale_factor
        scores = features.get('scores', np.ones(len(keypoints)))
        
        for i, (kp, score) in enumerate(zip(keypoints, scores)):
            x, y = int(kp[0]), int(kp[1])
            # Color based on score (red = high, blue = low)
            color = (int(255 * (1 - score)), 0, int(255 * score))
            cv2.circle(image, (x, y), 3, color, -1)
        
        # Add text info
        info_text = f"Features: {features['num_features']} | SuperPoint"
        cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save if requested
        if output_path:
            cv2.imwrite(str(output_path), image)
            logger.info(f"Feature visualization saved to {output_path}")
        
        return image


def load_images_from_directory(directory: Union[str, Path], 
                             extensions: Optional[List[str]] = None) -> List[Path]:
    """
    Load all image files from a directory.
    
    Args:
        directory: Path to directory containing images
        extensions: List of file extensions to include
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(directory.glob(f"*{ext}"))
        image_paths.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(image_paths)
