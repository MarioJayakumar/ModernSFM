"""
Test script for feature matching functionality.
"""

import sys
from pathlib import Path
import logging
import os

import hydra
from omegaconf import DictConfig
from main.core.feature_extraction import FeatureExtractor, load_images_from_directory
from main.core.feature_matching import FeatureMatcher

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="base_config")
def test_feature_matching(cfg: DictConfig) -> None:
    """Test feature matching on sample images."""
    
    # Initialize feature extractor and matcher
    try:
        extractor = FeatureExtractor(cfg)
        matcher = FeatureMatcher(cfg)
        logger.info("Feature extractor and matcher initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return
    
    # Look for test images
    data_dir = Path("data/test_data")
    if not data_dir.exists():
        logger.warning(f"Test data directory not found: {data_dir}")
        logger.info("Please add some test images to data/test_data/")
        return
    
    # Load images
    image_paths = load_images_from_directory(data_dir, cfg.io.image_extensions)
    
    if len(image_paths) < 2:
        logger.warning("Need at least 2 images for matching tests")
        logger.info("Please add more test images (.jpg, .png, etc.) to data/test_data/")
        return
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Extract features from first few images
    test_images = image_paths[:min(3, len(image_paths))]
    logger.info(f"Extracting features from {len(test_images)} images")
    
    try:
        all_features = extractor.extract_features_batch(test_images)
        logger.info(f"Successfully extracted features from {len(all_features)} images")
        
        # Create output directory
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Visualize extracted features for each image
        logger.info("Visualizing extracted features for each image")
        for i, (feat, image_path) in enumerate(zip(all_features, test_images)):
            logger.info(f"Image {i}: {feat['num_features']} features ({feat['extractor_type']})")
            
            # Visualize features for this image
            feature_vis_path = output_dir / f"features_{i}.jpg"
            extractor.visualize_features(image_path, feat, feature_vis_path)
            logger.info(f"Feature visualization {i} saved to: {feature_vis_path}")
            
    except Exception as e:
        logger.exception(f"Feature extraction failed: {e}")
        return
    
    # Test pairwise matching
    if len(all_features) >= 2:
        logger.info("Testing pairwise feature matching")
        
        try:
            matches = matcher.match_pairs(all_features[0], all_features[1])
            logger.info(f"Found {matches['num_matches']} matches between images 0 and 1")
            logger.info(f"Match confidence range: {matches['confidence'].min():.3f} - {matches['confidence'].max():.3f}")
            
            # Create output directory
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            
            # Visualize matches
            vis_path = output_dir / f"matches_{test_images[0].stem}_{test_images[1].stem}.jpg"
            matcher.visualize_matches(test_images[0], test_images[1], matches, vis_path)
            logger.info(f"Match visualization saved to: {vis_path}")
            
            # Test geometric filtering
            logger.info("Testing geometric filtering")
            
            # Test fundamental matrix filtering
            filtered_matches = matcher.filter_matches(matches, method="fundamental")
            logger.info(f"Fundamental matrix filtering: {matches['num_matches']} → {filtered_matches['num_matches']} matches")
            logger.info(f"Inlier ratio: {filtered_matches.get('inlier_ratio', 0):.3f}")
            
            # Apply spatial distribution to filtered matches
            logger.info("Applying spatial distribution to matches")
            spatially_distributed_matches = matcher.apply_spatial_distribution_to_matches(
                filtered_matches, grid_size=8, matches_per_cell=12
            )
            logger.info(f"Spatial distribution: {filtered_matches['num_matches']} → {spatially_distributed_matches['num_matches']} matches")
            
            # Visualize filtered matches with enhanced debugging view (show more keypoints)
            vis_filtered_path = output_dir / f"matches_filtered_{test_images[0].stem}_{test_images[1].stem}.jpg"
            matcher.visualize_matches_with_all_keypoints(
                test_images[0], test_images[1], 
                all_features[0], all_features[1], 
                filtered_matches, vis_filtered_path,
                max_matches=200, max_keypoints_per_image=2000  # Show many more keypoints
            )
            logger.info(f"Enhanced filtered match visualization saved to: {vis_filtered_path}")
            
            # Visualize spatially distributed matches with enhanced debugging view (show more keypoints)
            vis_spatial_path = output_dir / f"matches_spatial_{test_images[0].stem}_{test_images[1].stem}.jpg"
            matcher.visualize_matches_with_all_keypoints(
                test_images[0], test_images[1], 
                all_features[0], all_features[1], 
                spatially_distributed_matches, vis_spatial_path,
                max_matches=200, max_keypoints_per_image=2000  # Show many more keypoints
            )
            logger.info(f"Enhanced spatially distributed match visualization saved to: {vis_spatial_path}")
            
            # Also create standard visualizations for comparison
            vis_filtered_standard_path = output_dir / f"matches_filtered_standard_{test_images[0].stem}_{test_images[1].stem}.jpg"
            matcher.visualize_matches(test_images[0], test_images[1], filtered_matches, vis_filtered_standard_path)
            logger.info(f"Standard filtered match visualization saved to: {vis_filtered_standard_path}")
            
            # Test homography filtering
            homography_matches = matcher.filter_matches(matches, method="homography")
            logger.info(f"Homography filtering: {matches['num_matches']} → {homography_matches['num_matches']} matches")
            logger.info(f"Homography inlier ratio: {homography_matches.get('inlier_ratio', 0):.3f}")
            
        except Exception as e:
            logger.exception(f"Pairwise matching failed: {e}")
            return
    
    # Test all-pairs matching if we have multiple images
    if len(all_features) >= 3:
        logger.info("Testing all-pairs matching")
        
        try:
            all_matches = matcher.match_all_pairs(all_features)
            
            total_matches = sum(m['num_matches'] for m in all_matches.values())
            logger.info(f"All-pairs matching completed: {len(all_matches)} pairs, {total_matches} total matches")
            
            # Show match statistics
            for (i, j), match_result in all_matches.items():
                if 'error' not in match_result:
                    logger.info(f"Pair ({i}, {j}): {match_result['num_matches']} matches")
                else:
                    logger.warning(f"Pair ({i}, {j}): Failed - {match_result['error']}")
            
            # Visualize a few pairs
            vis_count = 0
            for (i, j), match_result in all_matches.items():
                if vis_count >= 2:  # Limit visualizations
                    break
                
                if match_result['num_matches'] > 0:
                    vis_path = output_dir / f"matches_pair_{i}_{j}.jpg"
                    matcher.visualize_matches(
                        test_images[i], test_images[j], 
                        match_result, vis_path
                    )
                    logger.info(f"Pair ({i}, {j}) visualization saved to: {vis_path}")
                    vis_count += 1
                    
        except Exception as e:
            logger.exception(f"All-pairs matching failed: {e}")
    
    # Performance summary
    logger.info("=== Feature Matching Test Summary ===")
    logger.info(f"Matcher type: {cfg.feature_matching.matcher_type}")
    logger.info(f"Match threshold: {cfg.feature_matching.match_threshold}")
    logger.info(f"Max matches: {cfg.feature_matching.max_matches}")
    logger.info(f"Images processed: {len(all_features)}")
    
    if len(all_features) >= 2:
        logger.info(f"Sample match count: {matches['num_matches']}")
        if 'inlier_ratio' in filtered_matches:
            logger.info(f"Geometric filtering inlier ratio: {filtered_matches['inlier_ratio']:.3f}")
    
    logger.info("Feature matching test completed!")


if __name__ == "__main__":
    test_feature_matching()
