"""
Centralized visualization coordinator for the ModernSFM pipeline.

This module provides a unified interface for generating all intermediate visualizations
during the reconstruction pipeline, avoiding code duplication across scripts.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import numpy as np

logger = logging.getLogger(__name__)


class PipelineVisualizer:
    """
    Centralized visualization coordinator that delegates to existing component visualizers.
    """
    
    def __init__(self, feature_extractor=None, feature_matcher=None, visualizer=None):
        """
        Initialize with references to component visualizers.
        
        Args:
            feature_extractor: FeatureExtractor instance with visualize_features method
            feature_matcher: FeatureMatcher instance with visualize_matches method  
            visualizer: SfMVisualizer instance for 3D visualizations
        """
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.visualizer = visualizer
        
    def visualize_feature_extraction(self, 
                                   images: List[Path], 
                                   features_list: List[Dict],
                                   output_dir: str = "outputs",
                                   max_images: int = 3,
                                   max_keypoints: int = 2000) -> List[Path]:
        """
        Generate feature extraction visualizations using the component's method.
        
        Args:
            images: List of image paths
            features_list: List of feature dictionaries
            output_dir: Output directory for visualizations
            max_images: Maximum number of images to visualize
            max_keypoints: Maximum keypoints to display per image
            
        Returns:
            List of paths to generated visualization files
        """
        if not self.feature_extractor:
            logger.warning("No feature extractor provided for visualization")
            return []
            
        output_paths = []
        
        try:
            logger.info("Generating feature extraction visualizations...")
            
            # Limit number of images to avoid too many files
            num_to_visualize = min(max_images, len(images))
            
            for i in range(num_to_visualize):
                image_path = str(images[i])
                features = features_list[i]
                
                if features and 'keypoints' in features:
                    # Use the feature extractor's existing method
                    output_path = f"{output_dir}/features_image_{i+1}.jpg"
                    
                    self.feature_extractor.visualize_features(
                        image_path, features, output_path
                    )
                    
                    output_paths.append(Path(output_path))
                    logger.info(f"Generated feature visualization: {output_path}")
                    
        except Exception as e:
            logger.warning(f"Feature extraction visualization failed: {e}")
            
        return output_paths
    
    def visualize_feature_matching(self,
                                 images: List[Path],
                                 features_list: List[Dict], 
                                 matches_dict: Dict[Tuple[int, int], Dict],
                                 output_dir: str = "outputs",
                                 max_pairs: int = 5) -> List[Path]:
        """
        Generate feature matching visualizations using the component's method.
        
        Args:
            images: List of image paths
            features_list: List of feature dictionaries  
            matches_dict: Dictionary of pairwise matches
            output_dir: Output directory for visualizations
            max_pairs: Maximum number of pairs to visualize
            
        Returns:
            List of paths to generated visualization files
        """
        if not self.feature_matcher:
            logger.warning("No feature matcher provided for visualization")
            return []
            
        output_paths = []
        
        try:
            logger.info("Generating feature matching visualizations...")
            
            # Select representative pairs to visualize
            pairs_to_visualize = []
            pair_count = 0
            
            for pair, match_data in matches_dict.items():
                if pair_count >= max_pairs:
                    break
                    
                i, j = pair
                if (i < len(images) and j < len(images) and 
                    match_data and 'matches' in match_data and 
                    len(match_data['matches']) > 0):
                    pairs_to_visualize.append((i, j, match_data))
                    pair_count += 1
            
            # Generate visualizations for selected pairs
            for i, j, match_data in pairs_to_visualize:
                try:
                    # Use the feature matcher's existing method
                    output_path = f"{output_dir}/matches_pair_{i+1}_{j+1}.jpg"
                    
                    self.feature_matcher.visualize_matches(
                        str(images[i]), str(images[j]),
                        match_data, output_path
                    )
                    
                    output_paths.append(Path(output_path))
                    logger.info(f"Generated match visualization: {output_path}")
                    
                except Exception as e:
                    logger.warning(f"Failed to visualize matches for pair ({i+1}, {j+1}): {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Feature matching visualization failed: {e}")
            
        return output_paths
    
    def visualize_poses_and_points(self,
                                 points_3d: np.ndarray,
                                 camera_poses: List[Dict],
                                 title: str = "3D Reconstruction",
                                 output_dir: str = "outputs/visualizations") -> List[Path]:
        """
        Generate 3D visualization of poses and points using the SfM visualizer.
        
        Args:
            points_3d: 3D points array [N, 3]
            camera_poses: List of camera pose dictionaries
            title: Title for the visualization
            output_dir: Output directory for visualizations
            
        Returns:
            List of paths to generated visualization files
        """
        if not self.visualizer:
            logger.warning("No 3D visualizer provided for visualization")
            return []
            
        output_paths = []
        
        try:
            logger.info(f"Generating 3D visualization: {title}")
            
            # Convert poses to format expected by visualizer
            camera_poses_dict = {}
            for i, pose in enumerate(camera_poses):
                if pose is not None and 'R' in pose and 't' in pose:
                    # Combine R and t into [3, 4] matrix format
                    camera_poses_dict[i] = np.hstack([pose['R'], pose['t']])
            
            # Use the SfM visualizer's existing method
            self.visualizer.visualize_point_cloud(
                points_3d,
                colors=None,
                camera_poses=camera_poses_dict,
                title=title,
                method="export"
            )
            
            # The visualizer saves to its default location - record expected paths
            # (Note: the actual paths depend on the visualizer's internal naming)
            expected_files = [
                f"outputs/visualizations/{title.lower().replace(' ', '_')}.ply",
                f"outputs/visualizations/{title.lower().replace(' ', '_')}.obj", 
                f"outputs/visualizations/{title.lower().replace(' ', '_')}_viewer.html"
            ]
            
            for file_path in expected_files:
                if Path(file_path).exists():
                    output_paths.append(Path(file_path))
                    
            logger.info(f"Generated 3D visualization files: {len(output_paths)} files")
            
        except Exception as e:
            logger.warning(f"3D visualization failed: {e}")
            
        return output_paths
    
    def visualize_pipeline_stage(self, 
                               stage: str,
                               images: List[Path] = None,
                               features_list: List[Dict] = None,
                               matches_dict: Dict = None,
                               points_3d: np.ndarray = None,
                               camera_poses: List[Dict] = None,
                               output_dir: str = "outputs",
                               **kwargs) -> List[Path]:
        """
        Unified interface to generate visualizations for any pipeline stage.
        
        Args:
            stage: Stage name ('features', 'matches', 'triangulation', 'bundle_adjustment')
            images: Image paths (required for features/matches)
            features_list: Feature data (required for features/matches) 
            matches_dict: Match data (required for matches)
            points_3d: 3D points (required for triangulation/bundle_adjustment)
            camera_poses: Camera poses (required for triangulation/bundle_adjustment)
            output_dir: Output directory
            **kwargs: Additional arguments passed to specific visualizers
            
        Returns:
            List of paths to generated visualization files
        """
        output_paths = []
        
        try:
            if stage == "features" and images and features_list:
                output_paths = self.visualize_feature_extraction(
                    images, features_list, output_dir, **kwargs
                )
                
            elif stage == "matches" and images and features_list and matches_dict:
                output_paths = self.visualize_feature_matching(
                    images, features_list, matches_dict, output_dir, **kwargs
                )
                
            elif stage in ["triangulation", "bundle_adjustment"] and points_3d is not None and camera_poses:
                title = "Triangulated Points" if stage == "triangulation" else "Bundle Adjusted Reconstruction"
                output_paths = self.visualize_poses_and_points(
                    points_3d, camera_poses, title, **kwargs
                )
                
            else:
                logger.warning(f"Cannot visualize stage '{stage}' - missing required data")
                
        except Exception as e:
            logger.error(f"Pipeline visualization failed for stage '{stage}': {e}")
            
        return output_paths