"""
Full end-to-end SfM reconstruction pipeline integrating all ModernSFM components.

This module provides a complete Structure-from-Motion pipeline that connects:
1. Feature extraction (SuperPoint + multi-scale + spatial distribution)
2. Feature matching (LightGlue + robust filtering)
3. Pose estimation (COLMAP/OpenCV fallback)
4. Triangulation (multi-view with quality filtering)  
5. Bundle adjustment (GPU-accelerated PyTorch optimization)

The pipeline handles the complete data flow from input images to final 3D reconstruction
with comprehensive error handling, progress tracking, and quality assessment.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Sequence
import logging
from tqdm import tqdm
import json
import time
from dataclasses import dataclass
import os

from ..core.feature_extraction import FeatureExtractor
from ..core.feature_matching import FeatureMatcher
from ..core.pose_estimation import PoseEstimator
from ..core.triangulation import Triangulator
from ..core.bundle_adjustment import BundleAdjuster, BundleAdjustmentResult
from ..utils.visualization import SfMVisualizer
from ..utils.pipeline_visualizer import PipelineVisualizer

logger = logging.getLogger(__name__)

@dataclass
class ReconstructionResult:
    """Complete reconstruction results from the full pipeline."""
    success: bool
    error_message: Optional[str]
    
    # Pipeline stage results
    num_images: int
    num_features_extracted: Dict[int, int]
    num_matches: Dict[Tuple[int, int], int]
    num_poses_estimated: int
    num_points_triangulated: int
    
    # Final reconstruction data
    points_3d: Optional[np.ndarray]
    camera_poses: Optional[List[Dict[str, np.ndarray]]]
    camera_intrinsics: Optional[np.ndarray]
    bundle_adjustment_result: Optional[BundleAdjustmentResult]
    
    # Quality metrics
    mean_reprojection_error: Optional[float]
    inlier_ratio: Optional[float]
    reconstruction_scale: Optional[float]
    
    # Performance metrics
    total_time: float
    stage_times: Dict[str, float]
    
    # Output paths
    output_directory: Optional[Path]
    visualization_files: Optional[List[Path]]


class FullReconstructionPipeline:
    """
    Complete end-to-end Structure-from-Motion reconstruction pipeline.
    
    Integrates all ModernSFM components into a unified workflow:
    - Loads images from input directory
    - Extracts features using SuperPoint with multi-scale and spatial distribution
    - Matches features using LightGlue with robust geometric filtering
    - Estimates camera poses using COLMAP/OpenCV methods
    - Triangulates 3D points from multi-view correspondences
    - Optimizes reconstruction using GPU-accelerated bundle adjustment
    - Generates comprehensive visualizations and quality reports
    """
    
    def __init__(self, config):
        """
        Initialize the reconstruction pipeline.
        
        Args:
            config: Hydra configuration object with all pipeline parameters
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Initialize all pipeline components
        self.feature_extractor = FeatureExtractor(config)
        self.feature_matcher = FeatureMatcher(config)
        self.pose_estimator = PoseEstimator(config)
        self.triangulator = Triangulator(config)
        self.bundle_adjuster = BundleAdjuster(config)
        self.visualizer = SfMVisualizer(config)
        
        # Initialize centralized pipeline visualizer
        self.pipeline_visualizer = PipelineVisualizer(
            feature_extractor=self.feature_extractor,
            feature_matcher=self.feature_matcher,
            visualizer=self.visualizer
        )
        
        # Pipeline state
        self.images = []
        self.features = []
        self.matches = {}
        self.poses = []
        self.points_3d = None
        self.intrinsics = None
        
        # Directory structure for organized output
        self.intermediate_dir = None
        
        logger.info(f"FullReconstructionPipeline initialized on {self.device}")
    
    def load_images(self, image_dir: Union[str, Path], max_images: Optional[int]) -> List[Path]:
        """
        Load images from directory with validation.
        
        Args:
            image_dir: Path to directory containing images
            
        Returns:
            List of valid image paths
            
        Raises:
            ValueError: If directory doesn't exist or contains no valid images
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise ValueError(f"Image directory does not exist: {image_dir}")
        
        # Get valid image extensions from config
        valid_extensions = self.config.io.image_extensions
        
        # Find all valid image files
        image_paths = []
        for ext in valid_extensions:
            image_paths.extend(image_dir.glob(f"*{ext}"))
            image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
        
        image_paths = sorted(image_paths)[:max_images]
        
        if len(image_paths) == 0:
            raise ValueError(f"No valid images found in {image_dir}. Supported extensions: {valid_extensions}")
        
        if len(image_paths) < 2:
            raise ValueError(f"At least 2 images required for reconstruction, found {len(image_paths)}")
        
        logger.info(f"Loaded {len(image_paths)} images from {image_dir}")
        self.images = image_paths
        return image_paths
    
    def extract_features(self) -> Dict[int, Dict]:
        """
        Extract features from all loaded images.
        
        Returns:
            Dictionary mapping image indices to feature dictionaries
            
        Raises:
            RuntimeError: If feature extraction fails
        """
        logger.info("Starting feature extraction...")
        start_time = time.time()
        
        try:
            # Extract features from all images
            self.features = self.feature_extractor.extract_features_batch(self.images)
            
            # Validate results
            num_features = {}
            for i, features in enumerate(self.features):
                if features is None or 'keypoints' not in features:
                    raise RuntimeError(f"Feature extraction failed for image {i}: {self.images[i]}")
                num_features[i] = len(features['keypoints'])
            
            extraction_time = time.time() - start_time
            total_features = sum(num_features.values())
            
            logger.info(f"Feature extraction completed in {extraction_time:.2f}s")
            logger.info(f"Extracted {total_features} total features across {len(self.images)} images")
            logger.info(f"Features per image: {num_features}")
            
            # Generate intermediate visualizations only if intermediate_dir is provided
            if self.intermediate_dir:
                self.intermediate_dir.mkdir(exist_ok=True)
                features_dir = self.intermediate_dir / "features"
                features_dir.mkdir(exist_ok=True)
                self.pipeline_visualizer.visualize_pipeline_stage(
                    "features", 
                    images=self.images, 
                    features_list=self.features,
                    max_images=3,
                    max_keypoints=2000
                )
            
            return {i: features for i, features in enumerate(self.features)}
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise RuntimeError(f"Feature extraction failed: {e}")
    
    def match_features(self) -> Dict[Tuple[int, int], Dict]:
        """
        Match features between all image pairs.
        
        Returns:
            Dictionary mapping image pair indices to match dictionaries
            
        Raises:
            RuntimeError: If feature matching fails
        """
        logger.info("Starting feature matching...")
        start_time = time.time()
        
        try:
            # Match all pairs using the FeatureMatcher
            self.matches = self.feature_matcher.match_all_pairs(self.features)
            
            # Validate and log results
            num_matches = {}
            total_matches = 0
            valid_pairs = 0
            
            for pair, match_data in self.matches.items():
                if match_data is None or 'matches' not in match_data:
                    logger.warning(f"No matches found for pair {pair}")
                    num_matches[pair] = 0
                    continue
                    
                num_matches[pair] = len(match_data['matches'])
                total_matches += num_matches[pair]
                if num_matches[pair] > 0:
                    valid_pairs += 1
            
            matching_time = time.time() - start_time
            
            logger.info(f"Feature matching completed in {matching_time:.2f}s")
            logger.info(f"Found {total_matches} total matches across {len(self.matches)} pairs")
            logger.info(f"Valid pairs with matches: {valid_pairs}/{len(self.matches)}")
            
            if valid_pairs < len(self.images) - 1:
                logger.warning(f"Insufficient connectivity: only {valid_pairs} valid pairs from {len(self.images)} images")
            
            # Generate intermediate visualizations only if intermediate_dir is provided
            if self.intermediate_dir:
                matches_dir = self.intermediate_dir / "matches"
                matches_dir.mkdir(exist_ok=True)
                self.pipeline_visualizer.visualize_pipeline_stage(
                    "matches",
                    images=self.images,
                    features_list=self.features, 
                    matches_dict=self.matches,
                    max_pairs=5
                )
            
            return self.matches
            
        except Exception as e:
            logger.error(f"Feature matching failed: {e}")
            raise RuntimeError(f"Feature matching failed: {e}")
    
    def estimate_poses(self) -> List[Dict[str, np.ndarray]]:
        """
        Estimate camera poses from matched features.
        
        Returns:
            List of camera pose dictionaries with 'R' and 't' keys
            
        Raises:
            RuntimeError: If pose estimation fails
        """
        logger.info("Starting pose estimation...")
        start_time = time.time()
        
        try:
            # Build pose estimation graph from matches
            pose_pairs = []
            for pair, match_data in self.matches.items():
                if match_data is None or 'matches' not in match_data or len(match_data['matches']) == 0:
                    continue
                
                i, j = pair
                matches = match_data['matches']
                
                # Get matched keypoints
                kpts1 = self.features[i]['keypoints'][matches[:, 0]]
                kpts2 = self.features[j]['keypoints'][matches[:, 1]]
                
                pose_pairs.append((i, j, kpts1, kpts2))
            
            if len(pose_pairs) == 0:
                raise RuntimeError("No valid matches found for pose estimation")
            
            # Initialize poses list
            num_cameras = len(self.images)
            self.poses = [None] * num_cameras
            
            # Set first camera as reference (identity pose)
            self.poses[0] = {
                'R': np.eye(3),
                't': np.zeros((3, 1))
            }
            
            # Estimate poses incrementally
            estimated_cameras = {0}  # Track which cameras have poses
            
            # Process pose pairs to build incremental reconstruction
            max_iterations = len(pose_pairs) * 2  # Prevent infinite loops
            iteration = 0
            
            while len(estimated_cameras) < num_cameras and iteration < max_iterations:
                progress_made = False
                iteration += 1
                
                for i, j, kpts1, kpts2 in pose_pairs:
                    # Skip if both cameras already estimated
                    if i in estimated_cameras and j in estimated_cameras:
                        continue
                    
                    # Estimate relative pose
                    try:
                        intrinsics = self.get_default_intrinsics()
                        result = self.pose_estimator.estimate_two_view_geometry(
                            kpts1, kpts2, intrinsics, intrinsics
                        )
                        
                        if not result['success']:
                            continue
                        
                        R_rel = result['R']
                        t_rel = result['t']
                        
                        # Add pose to global reconstruction
                        if i in estimated_cameras and j not in estimated_cameras:
                            # Camera i is known, compute j
                            R_i = self.poses[i]['R']
                            t_i = self.poses[i]['t']
                            
                            # Global pose for camera j
                            R_j = R_rel @ R_i
                            t_j = R_rel @ t_i + t_rel
                            
                            self.poses[j] = {'R': R_j, 't': t_j}
                            estimated_cameras.add(j)
                            progress_made = True
                            
                        elif j in estimated_cameras and i not in estimated_cameras:
                            # Camera j is known, compute i  
                            R_j = self.poses[j]['R']
                            t_j = self.poses[j]['t']
                            
                            # Global pose for camera i (inverse relative pose)
                            R_i = R_rel.T @ R_j
                            t_i = R_rel.T @ (t_j - t_rel)
                            
                            self.poses[i] = {'R': R_i, 't': t_i}
                            estimated_cameras.add(i)
                            progress_made = True
                    
                    except Exception as e:
                        logger.warning(f"Pose estimation failed for pair ({i}, {j}): {e}")
                        continue
                
                if not progress_made:
                    logger.warning(f"No progress in pose estimation iteration {iteration}")
                    break
            
            # Fill remaining poses with identity (fallback)
            for i in range(num_cameras):
                if self.poses[i] is None:
                    logger.warning(f"Could not estimate pose for camera {i}, using identity")
                    self.poses[i] = {
                        'R': np.eye(3),
                        't': np.zeros((3, 1))
                    }
            
            pose_time = time.time() - start_time
            logger.info(f"Pose estimation completed in {pose_time:.2f}s")
            logger.info(f"Estimated poses for {len(estimated_cameras)}/{num_cameras} cameras")
            
            return self.poses
            
        except Exception as e:
            logger.error(f"Pose estimation failed: {e}")
            raise RuntimeError(f"Pose estimation failed: {e}")
    
    def triangulate_points(self) -> np.ndarray:
        """
        Triangulate 3D points from multi-view correspondences.
        
        Returns:
            3D points array of shape [N, 3]
            
        Raises:
            RuntimeError: If triangulation fails
        """
        logger.info("Starting point triangulation...")
        start_time = time.time()
        
        try:
            # Get default intrinsics
            self.intrinsics = self.get_default_intrinsics()
            
            # Build observations dictionary for multi-view triangulation
            # Track correspondences across views
            tracks = {}  # track_id -> {camera_id: keypoint_idx}
            track_id = 0
            
            # Use multi-view triangulation for improved success rate
            logger.info("Building multi-view tracks for improved triangulation")
            
            # Build comprehensive tracks from all pairwise matches
            tracks = self.triangulator.create_tracks(
                {pair: data['matches'] for pair, data in self.matches.items() if data and 'matches' in data}, 
                len(self.images)
            )
            
            logger.info(f"Built {len(tracks)} multi-view tracks from pairwise matches")
            
            # Log track length distribution for debugging
            track_lengths = [len(track) for track in tracks.values()]
            if track_lengths:
                logger.info(f"Track length distribution: min={min(track_lengths)}, "
                          f"max={max(track_lengths)}, mean={np.mean(track_lengths):.1f}")
            
            # If we have too few tracks, fall back to two-view approach
            if len(tracks) < 100:
                logger.info("Too few multi-view tracks, falling back to two-view triangulation")
                
                # Find the pair with the most matches
                best_pair = None
                max_matches = 0
                
                for pair, match_data in self.matches.items():
                    if match_data and 'matches' in match_data:
                        num_matches = len(match_data['matches'])
                        if num_matches > max_matches:
                            max_matches = num_matches
                            best_pair = pair
                
                if best_pair is None:
                    raise RuntimeError("No valid matches found for triangulation")
                
                i, j = best_pair
                match_data = self.matches[best_pair]
                matches = match_data['matches']
                
                logger.info(f"Using best pair ({i}, {j}) with {len(matches)} matches for triangulation")
                
                # Create simple tracks for just these two views
                tracks = {}
                for idx, match in enumerate(matches):
                    kpt_i, kpt_j = match[0], match[1]
                    tracks[idx] = {i: kpt_i, j: kpt_j}
            
            logger.info(f"Using {len(tracks)} tracks for triangulation")
            
            # Determine if we should use multi-view or two-view triangulation
            track_lengths = [len(track) for track in tracks.values()]
            multi_view_tracks = [track for track in tracks.values() if len(track) >= 3]
            
            logger.info(f"Found {len(multi_view_tracks)} tracks with 3+ views, {len(tracks) - len(multi_view_tracks)} with 2 views")
            
            # Prepare camera matrices for all cameras
            camera_matrices = {}
            for cam_id in range(len(self.images)):
                if self.poses[cam_id] is not None:
                    R = self.poses[cam_id]['R']
                    t = self.poses[cam_id]['t']
                    if t.shape == (3, 1):
                        t = t.flatten()
                    camera_matrices[cam_id] = self.intrinsics @ np.hstack([R, t.reshape(3, 1)])
            
            # Use multi-view triangulation when possible
            if len(multi_view_tracks) > len(tracks) // 4:  # If >25% are multi-view, use multi-view approach
                logger.info("Using multi-view triangulation approach")
                result = self._triangulate_multi_view_tracks(tracks, camera_matrices)
            else:
                logger.info("Using two-view triangulation approach")
                result = self._triangulate_two_view_tracks(tracks, camera_matrices)
            
            # Extract results
            if result and result.get('success', False):
                triangulated_points = result['triangulated_points']
                self.track_observations = result['track_observations']
                
                logger.info(f"Successfully triangulated {len(triangulated_points)}/{len(tracks)} points")
                logger.info(f"Created {len(self.track_observations)} observations for bundle adjustment")
            else:
                logger.warning("Triangulation failed")
                triangulated_points = []
                self.track_observations = []
            
            if len(triangulated_points) == 0:
                raise RuntimeError("No points could be triangulated")
            
            # Combine all triangulated points
            self.points_3d = np.array(triangulated_points) if triangulated_points else np.array([])
            
            triangulation_time = time.time() - start_time
            logger.info(f"Point triangulation completed in {triangulation_time:.2f}s")
            logger.info(f"Triangulated {len(self.points_3d)} 3D points from tracks")
            
            # Generate intermediate visualizations only if intermediate_dir is provided
            if self.intermediate_dir:
                self.pipeline_visualizer.visualize_pipeline_stage(
                    "triangulation",
                    points_3d=self.points_3d,
                    camera_poses=self.poses
                )
            
            return self.points_3d
            
        except Exception as e:
            logger.error(f"Point triangulation failed: {e}")
            raise RuntimeError(f"Point triangulation failed: {e}")
    
    def run_bundle_adjustment(self) -> BundleAdjustmentResult:
        """
        Run bundle adjustment to optimize the reconstruction.
        
        Returns:
            BundleAdjustmentResult with optimized parameters
            
        Raises:
            RuntimeError: If bundle adjustment fails
        """
        logger.info("Starting bundle adjustment...")
        start_time = time.time()
        
        try:
            # Use the real observations from triangulation (not synthetic reprojections)
            if not hasattr(self, 'track_observations') or len(self.track_observations) == 0:
                raise RuntimeError("No observations available for bundle adjustment. Run triangulation first.")
            
            logger.info(f"Using {len(self.track_observations)} real observations from triangulation")
            
            # Run bundle adjustment
            intrinsics = self.get_default_intrinsics()
            ba_result = self.bundle_adjuster.optimize_reconstruction(
                np.array(self.points_3d), self.poses, intrinsics, self.track_observations
            )
            
            # Update reconstruction with optimized results
            if ba_result.final_rmse < float('inf'):  # Check if optimization worked
                self.points_3d = ba_result.optimized_points_3d
                self.poses = ba_result.optimized_poses
                if ba_result.optimized_intrinsics is not None:
                    self.intrinsics = ba_result.optimized_intrinsics
            
            ba_time = time.time() - start_time
            logger.info(f"Bundle adjustment completed in {ba_time:.2f}s")
            logger.info(f"RMSE: {ba_result.initial_rmse:.3f} -> {ba_result.final_rmse:.3f}")
            logger.info(f"Inlier ratio: {ba_result.inlier_ratio:.3f}")
            
            return ba_result
            
        except Exception as e:
            logger.error(f"Bundle adjustment failed: {e}")
            # Create failed result
            return BundleAdjustmentResult(
                optimized_points_3d=self.points_3d,
                optimized_poses=self.poses,
                optimized_intrinsics=self.intrinsics,
                initial_rmse=float('inf'),
                final_rmse=float('inf'),
                num_iterations=0,
                convergence_reason=f"Failed: {e}",
                inlier_ratio=0.0,
                optimization_time=0.0
            )
    
    def save_results(self, output_dir: Union[str, Path], 
                    name: str = "reconstruction",
                    data_dir: Optional[Path] = None,
                    viz_dir: Optional[Path] = None) -> List[Path]:
        """
        Save reconstruction results in organized directory structure.
        
        Args:
            output_dir: Main output directory
            name: Base name for output files
            data_dir: Directory for core reconstruction data
            viz_dir: Directory for visualizations
            
        Returns:
            List of paths to generated files
        """
        # Use organized directories if provided, otherwise fall back to output_dir
        data_output_dir = data_dir if data_dir else Path(output_dir)
        data_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = []
        
        try:
            # Save 3D points in multiple formats in data directory
            if self.points_3d is not None:
                # NumPy format
                points_npy = data_output_dir / "points_3d.npy"
                np.save(points_npy, self.points_3d)
                output_files.append(points_npy)
                
                # Text format
                points_txt = data_output_dir / "points_3d.txt"
                np.savetxt(points_txt, self.points_3d, fmt='%.6f')
                output_files.append(points_txt)
            
            # Save camera poses in data directory
            if self.poses:
                poses_json = data_output_dir / "camera_poses.json"
                poses_data = []
                for i, pose in enumerate(self.poses):
                    poses_data.append({
                        'camera_id': i,
                        'R': pose['R'].tolist(),
                        't': pose['t'].flatten().tolist()
                    })
                
                with open(poses_json, 'w') as f:
                    json.dump(poses_data, f, indent=2)
                output_files.append(poses_json)
            
            # Save intrinsics in data directory
            if self.intrinsics is not None:
                intrinsics_json = data_output_dir / "camera_intrinsics.json"
                with open(intrinsics_json, 'w') as f:
                    json.dump({
                        'intrinsics': self.intrinsics.tolist(),
                        'image_size': [640, 480]  # Default - should get from actual images
                    }, f, indent=2)
                output_files.append(intrinsics_json)
            
            logger.info(f"Saved reconstruction results to {output_dir}")
            return output_files
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return output_files
    
    def create_visualizations(self, output_dir: Union[str, Path],
                            name: str = "reconstruction",
                            viz_dir: Optional[Path] = None) -> List[Path]:
        """
        Create comprehensive visualizations of the reconstruction.
        
        Args:
            output_dir: Main output directory
            name: Base name for visualization files
            viz_dir: Directory for visualizations (if None, creates visualizations subdir)
            
        Returns:
            List of paths to generated visualization files
        """
        # Use organized viz directory if provided, otherwise create visualizations subdir
        if viz_dir:
            vis_output_dir = viz_dir
        else:
            vis_output_dir = Path(output_dir) / "visualizations"
        
        vis_output_dir.mkdir(parents=True, exist_ok=True)
        
        vis_files = []
        
        try:
            if self.points_3d is not None and self.poses:
                # Create triangulation_result dict in the format expected by visualizer
                triangulation_result = {
                    'points_3d': self.points_3d,
                    'valid_mask': np.ones(len(self.points_3d), dtype=bool),  # All points are valid
                    'method_used': 'pipeline_triangulation'
                }
                
                # Convert poses list to camera_poses dict format
                camera_poses_dict = {}
                for i, pose in enumerate(self.poses):
                    if pose is not None and 'R' in pose and 't' in pose:
                        # Combine R and t into [3, 4] matrix format expected by visualizer
                        camera_poses_dict[i] = np.hstack([pose['R'], pose['t']])
                
                # Create visualization using the correct format  
                vis_files = self.visualizer.visualize_reconstruction(
                    triangulation_result,
                    camera_poses_dict,
                    title="reconstruction"  # Use consistent naming
                )
                
                logger.info(f"Created {len(vis_files)} visualization files")
            
            return vis_files
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
            return vis_files
    
    def get_default_intrinsics(self) -> np.ndarray:
        """Get default camera intrinsics for reconstruction."""
        # Default intrinsics for typical images
        # In practice, this should be calibrated or estimated
        fx = fy = 800.0  # Focal length
        cx = cy = 320.0  # Principal point (assuming 640x480 images)
        
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    
    def reconstruct(self, image_dir: Union[str, Path], 
                   output_dir: Union[str, Path],
                   name: str = "reconstruction",
                   max_images: Optional[int] = -1,
                   data_dir: Optional[Path] = None,
                   viz_dir: Optional[Path] = None,
                   intermediate_dir: Optional[Path] = None) -> ReconstructionResult:
        """
        Run the complete reconstruction pipeline.
        
        Args:
            image_dir: Directory containing input images
            output_dir: Main output directory
            name: Base name for output files
            max_images: Maximum number of images to process
            data_dir: Directory for core reconstruction data (points, poses, etc.)
            viz_dir: Directory for final visualizations
            intermediate_dir: Directory for intermediate/debug files (optional)
            
        Returns:
            ReconstructionResult with complete pipeline results
        """
        logger.info("="*50)
        logger.info("STARTING FULL RECONSTRUCTION PIPELINE")
        logger.info("="*50)
        
        pipeline_start = time.time()
        stage_times = {}
        
        # Store intermediate directory for conditional use
        self.intermediate_dir = intermediate_dir
        
        try:
            # Stage 1: Load images
            logger.info("Stage 1/6: Loading images...")
            stage_start = time.time()
            images = self.load_images(image_dir, max_images=max_images)
            stage_times['image_loading'] = time.time() - stage_start
            
            # Stage 2: Extract features
            logger.info("Stage 2/6: Extracting features...")
            stage_start = time.time()
            features = self.extract_features()
            stage_times['feature_extraction'] = time.time() - stage_start
            
            # Stage 3: Match features
            logger.info("Stage 3/6: Matching features...")
            stage_start = time.time()
            matches = self.match_features()
            stage_times['feature_matching'] = time.time() - stage_start
            
            # Stage 4: Estimate poses
            logger.info("Stage 4/6: Estimating camera poses...")
            stage_start = time.time()
            poses = self.estimate_poses()
            stage_times['pose_estimation'] = time.time() - stage_start
            
            # Stage 5: Triangulate points
            logger.info("Stage 5/6: Triangulating 3D points...")
            stage_start = time.time()
            points_3d = self.triangulate_points()
            stage_times['triangulation'] = time.time() - stage_start
            
            # Stage 6: Bundle adjustment
            logger.info("Stage 6/6: Running bundle adjustment...")
            stage_start = time.time()
            ba_result = self.run_bundle_adjustment()
            stage_times['bundle_adjustment'] = time.time() - stage_start
            
            # Save results with organized directory structure
            logger.info("Saving reconstruction results...")
            stage_start = time.time()
            output_files = self.save_results(output_dir, name, data_dir, viz_dir)
            vis_files = self.create_visualizations(output_dir, name, viz_dir)
            stage_times['save_results'] = time.time() - stage_start
            
            total_time = time.time() - pipeline_start
            
            # Compute quality metrics
            mean_error = ba_result.final_rmse if ba_result else None
            inlier_ratio = ba_result.inlier_ratio if ba_result else None
            
            # Compute reconstruction scale (approximate)
            scale = None
            if self.points_3d is not None and len(self.points_3d) > 1:
                distances = np.linalg.norm(
                    self.points_3d[1:] - self.points_3d[:-1], axis=1
                )
                scale = np.median(distances)
            
            # Create success result
            result = ReconstructionResult(
                success=True,
                error_message=None,
                num_images=len(images),
                num_features_extracted={i: len(f['keypoints']) for i, f in enumerate(self.features)},
                num_matches={pair: len(m.get('matches', [])) for pair, m in self.matches.items()},
                num_poses_estimated=len([p for p in self.poses if p is not None]),
                num_points_triangulated=len(self.points_3d) if self.points_3d is not None else 0,
                points_3d=self.points_3d,
                camera_poses=self.poses,
                camera_intrinsics=self.intrinsics,
                bundle_adjustment_result=ba_result,
                mean_reprojection_error=mean_error,
                inlier_ratio=inlier_ratio,
                reconstruction_scale=scale,
                total_time=total_time,
                stage_times=stage_times,
                output_directory=Path(output_dir),
                visualization_files=vis_files
            )
            
            logger.info("="*50)
            logger.info("RECONSTRUCTION PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Total time: {total_time:.2f}s")
            logger.info(f"Reconstructed {len(self.points_3d)} 3D points from {len(images)} images")
            logger.info(f"Results saved to: {output_dir}")
            logger.info("="*50)
            
            return result
            
        except Exception as e:
            total_time = time.time() - pipeline_start
            error_msg = f"Pipeline failed: {e}"
            logger.error(error_msg)
            
            # Create failure result
            return ReconstructionResult(
                success=False,
                error_message=error_msg,
                num_images=len(self.images) if self.images else 0,
                num_features_extracted={},
                num_matches={},
                num_poses_estimated=0,
                num_points_triangulated=0,
                points_3d=None,
                camera_poses=None,
                camera_intrinsics=None,
                bundle_adjustment_result=None,
                mean_reprojection_error=None,
                inlier_ratio=None,
                reconstruction_scale=None,
                total_time=total_time,
                stage_times=stage_times,
                output_directory=None,
                visualization_files=None
            )
    
    def _triangulate_multi_view_tracks(self, tracks: Dict, camera_matrices: Dict) -> Dict:
        """Triangulate using multi-view approach for better accuracy."""
        triangulated_points = []
        track_observations = []
        successful_tracks = 0
        
        for track_id, track in tracks.items():
            if len(track) < 2:
                continue
                
            # Prepare observations for this track
            observations = {}
            for cam_id, kpt_idx in track.items():
                if cam_id in camera_matrices:
                    point_2d = self.features[cam_id]['keypoints'][kpt_idx]
                    observations[cam_id] = point_2d
            
            if len(observations) < 2:
                continue
            
            # Filter camera matrices for this track
            track_cameras = {cam_id: camera_matrices[cam_id] for cam_id in observations.keys()}
            
            # Triangulate this point using multi-view
            point_result = self.triangulator._triangulate_point_multi_view(
                observations, track_cameras, method='dlt'
            )
            
            if point_result.get('success', False) and point_result.get('valid', False):
                point_idx = len(triangulated_points)
                triangulated_points.append(point_result['point_3d'])
                
                # Add observations for bundle adjustment
                for cam_id, point_2d in observations.items():
                    track_observations.append({
                        'camera_id': cam_id,
                        'point_id': point_idx,
                        'point_2d': point_2d
                    })
                
                successful_tracks += 1
        
        logger.info(f"Multi-view triangulation: {successful_tracks}/{len(tracks)} tracks successful")
        
        return {
            'success': True,
            'triangulated_points': triangulated_points,
            'track_observations': track_observations
        }
    
    def _triangulate_two_view_tracks(self, tracks: Dict, camera_matrices: Dict) -> Dict:
        """Triangulate using two-view approach with batch processing."""
        # Find the best camera pair (most tracks)
        cam_pair_counts = {}
        for track in tracks.values():
            cam_ids = list(track.keys())
            if len(cam_ids) >= 2:
                for i in range(len(cam_ids)):
                    for j in range(i + 1, len(cam_ids)):
                        pair = tuple(sorted([cam_ids[i], cam_ids[j]]))
                        if pair[0] in camera_matrices and pair[1] in camera_matrices:
                            cam_pair_counts[pair] = cam_pair_counts.get(pair, 0) + 1
        
        if not cam_pair_counts:
            return {'success': False}
        
        # Use the pair with most tracks
        best_pair = max(cam_pair_counts.items(), key=lambda x: x[1])[0]
        cam1_id, cam2_id = best_pair
        
        logger.info(f"Using camera pair ({cam1_id}, {cam2_id}) with {cam_pair_counts[best_pair]} tracks")
        
        # Collect point pairs for batch triangulation
        all_points1 = []
        all_points2 = []
        valid_tracks = []
        
        for track_id, track in tracks.items():
            if cam1_id in track and cam2_id in track:
                point1 = self.features[cam1_id]['keypoints'][track[cam1_id]]
                point2 = self.features[cam2_id]['keypoints'][track[cam2_id]]
                
                all_points1.append(point1)
                all_points2.append(point2)
                valid_tracks.append((track_id, track))
        
        if len(all_points1) == 0:
            return {'success': False}
        
        # Batch triangulation
        points1_array = np.array(all_points1)
        points2_array = np.array(all_points2)
        P1 = camera_matrices[cam1_id]
        P2 = camera_matrices[cam2_id]
        
        result = self.triangulator.triangulate_two_view(
            points1_array, points2_array, P1, P2
        )
        
        if (result['points_3d'] is None or result['valid_mask'] is None):
            return {'success': False}
        
        # Extract successful triangulations
        triangulated_points = []
        track_observations = []
        
        valid_indices = np.where(result['valid_mask'])[0]
        for point_idx, track_idx in enumerate(valid_indices):
            # Add the 3D point
            triangulated_points.append(result['points_3d'][track_idx])
            
            # Add observations from ALL cameras in the track (not just the two used for triangulation)
            track_id, track = valid_tracks[track_idx]
            for cam_id, kpt_idx in track.items():
                if cam_id in camera_matrices:  # Only include cameras with valid poses
                    point_2d = self.features[cam_id]['keypoints'][kpt_idx]
                    track_observations.append({
                        'camera_id': cam_id,
                        'point_id': point_idx,
                        'point_2d': point_2d
                    })
        
        logger.info(f"Two-view triangulation: {len(triangulated_points)}/{len(valid_tracks)} tracks successful")
        
        return {
            'success': True,
            'triangulated_points': triangulated_points,
            'track_observations': track_observations
        }