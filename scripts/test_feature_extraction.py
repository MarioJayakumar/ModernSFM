"""
Test script for feature extraction functionality.
"""

import sys
from pathlib import Path
import logging
import os


import hydra
from omegaconf import DictConfig
from main.core.feature_extraction import FeatureExtractor, load_images_from_directory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="base_config")
def test_feature_extraction(cfg: DictConfig) -> None:
    """Test feature extraction on sample images."""
    
    # Initialize feature extractor
    try:
        extractor = FeatureExtractor(cfg)
        logger.info("Feature extractor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize feature extractor: {e}")
        return
    
    # Look for test images
    data_dir = Path("data/test_data")
    if not data_dir.exists():
        logger.warning(f"Test data directory not found: {data_dir}")
        logger.info("Please add some test images to data/test_data/")
        return
    
    # Load images
    image_paths = load_images_from_directory(data_dir, cfg.io.image_extensions)
    
    if not image_paths:
        logger.warning("No images found in data/test_data/")
        logger.info("Please add some test images (.jpg, .png, etc.) to data/test_data/")
        return
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Test single image extraction
    test_image = image_paths[0]
    logger.info(f"Testing feature extraction on: {test_image}")
    
    try:
        features = extractor.extract_features_single(test_image)
        logger.info(f"Successfully extracted {features['num_features']} features")
        logger.info(f"Keypoints shape: {features['keypoints'].shape}")
        logger.info(f"Descriptors shape: {features['descriptors'].shape}")
        
        # Visualize features
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        vis_path = output_dir / f"features_{test_image.stem}.jpg"
        extractor.visualize_features(test_image, features, vis_path)
        
        logger.info(f"Feature visualization saved to: {vis_path}")
        
    except Exception as e:
        logger.exception(f"Feature extraction failed: {e}")
        return
    
    # Test batch extraction if multiple images
    if len(image_paths) > 1:
        logger.info(f"Testing batch extraction on {min(3, len(image_paths))} images")
        
        try:
            batch_features = extractor.extract_features_batch(image_paths[:3])
            logger.info(f"Batch extraction successful for {len(batch_features)} images")
            
            for i, feat in enumerate(batch_features):
                logger.info(f"Image {i+1}: {feat['num_features']} features")
                
        except Exception as e:
            logger.error(f"Batch extraction failed: {e}")
    
    logger.info("Feature extraction test completed!")


if __name__ == "__main__":
    test_feature_extraction()
