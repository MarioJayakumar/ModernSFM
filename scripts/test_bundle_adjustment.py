"""
Test script for bundle adjustment module.
Tests both global and local bundle adjustment with synthetic and real data.
"""

import hydra
from omegaconf import DictConfig
import numpy as np
import logging
import sys
import os
from pathlib import Path
import time
import matplotlib.pyplot as plt

# Add the main directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from main.core.bundle_adjustment import BundleAdjuster, BundleAdjustmentResult
from main.core.feature_extraction import FeatureExtractor
from main.core.feature_matching import FeatureMatcher
from main.core.pose_estimation import PoseEstimator
from main.core.triangulation import Triangulator
from main.utils.visualization import SfMVisualizer

logger = logging.getLogger(__name__)

def create_synthetic_reconstruction(num_cameras: int = 5, num_points: int = 100, 
                                  noise_level: float = 0.5) -> tuple:
    """
    Create synthetic reconstruction data for testing bundle adjustment.
    
    Args:
        num_cameras: Number of cameras
        num_points: Number of 3D points
        noise_level: Noise level in pixels for 2D observations
        
    Returns:
        Tuple of (points_3d, poses, intrinsics, observations)
    """
    logger.info(f"Creating synthetic reconstruction with {num_cameras} cameras and {num_points} points")
    
    # Create synthetic 3D points in a cube
    np.random.seed(42)  # For reproducible results
    points_3d = np.random.uniform(-5, 5, (num_points, 3))
    points_3d[:, 2] += 10  # Move points away from cameras
    
    # Create synthetic camera poses in a circle looking at center
    poses = []
    radius = 6.0
    for i in range(num_cameras):
        angle = 2 * np.pi * i / num_cameras
        
        # Camera position in world coordinates
        camera_pos = np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
        
        # Simple rotation: camera looks toward center [0,0,10]
        # For simplicity, use identity rotation initially and just translate camera
        if i == 0:  # First camera at identity
            R = np.eye(3)
            t = np.array([0, 0, 0])  
        else:
            # Simple rotation around Z axis  
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            R = np.array([
                [cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0], 
                [0,      0,     1]
            ])
            # Translation to move camera
            t = np.array([-radius * cos_a, -radius * sin_a, 0])
            
        poses.append({
            'R': R,
            't': t.reshape(3, 1)
        })
    
    # Create synthetic intrinsics
    intrinsics = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Create observations with noise
    observations = []
    for cam_id, pose in enumerate(poses):
        R = pose['R']
        t = pose['t'].flatten()
        
        for point_id, point_3d in enumerate(points_3d):
            # Transform to camera coordinates: X_cam = R * X_world + t
            point_cam = R @ point_3d + t.flatten()
            
            # Only include points in front of camera with reasonable depth
            if point_cam[2] > 2.0:  # Must be at least 2 meters in front
                # Project to image: P = K * X_cam
                point_2d_hom = intrinsics @ point_cam
                point_2d = point_2d_hom[:2] / point_2d_hom[2]
                
                # Add noise
                point_2d += np.random.normal(0, noise_level, 2)
                
                # Only include points within image bounds (640x480 image)
                if 50 <= point_2d[0] <= 590 and 50 <= point_2d[1] <= 430:  # Leave border margin
                    observations.append({
                        'camera_id': cam_id,
                        'point_id': point_id,
                        'point_2d': point_2d.astype(np.float32)
                    })
    
    logger.info(f"Created {len(observations)} observations")
    return points_3d, poses, intrinsics, observations

def add_noise_to_reconstruction(points_3d: np.ndarray, poses: list, 
                               point_noise: float = 0.1, pose_noise: float = 0.01) -> tuple:
    """
    Add noise to reconstruction for testing optimization.
    
    Args:
        points_3d: Original 3D points
        poses: Original camera poses
        point_noise: Noise level for 3D points
        pose_noise: Noise level for camera poses
        
    Returns:
        Tuple of (noisy_points_3d, noisy_poses)
    """
    logger.info(f"Adding noise to reconstruction (point: {point_noise}, pose: {pose_noise})")
    
    # Add noise to 3D points
    noisy_points_3d = points_3d + np.random.normal(0, point_noise, points_3d.shape)
    
    # Add noise to camera poses
    noisy_poses = []
    for pose in poses:
        R = pose['R'].copy()
        t = pose['t'].copy()
        
        # Add small rotation noise (axis-angle)
        axis_angle_noise = np.random.normal(0, pose_noise, 3)
        angle = np.linalg.norm(axis_angle_noise)
        if angle > 0:
            axis = axis_angle_noise / angle
            # Rodrigues formula for small rotations
            K = np.array([[0, -axis[2], axis[1]],
                         [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]])
            R_noise = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
            R = R_noise @ R
        
        # Add translation noise
        t = t.astype(np.float64) + np.random.normal(0, pose_noise, t.shape)
        
        noisy_poses.append({
            'R': R,
            't': t
        })
    
    return noisy_points_3d, noisy_poses

def compute_reconstruction_error(points_3d_gt: np.ndarray, poses_gt: list,
                               points_3d_est: np.ndarray, poses_est: list) -> dict:
    """
    Compute reconstruction error metrics.
    
    Args:
        points_3d_gt: Ground truth 3D points
        poses_gt: Ground truth poses
        points_3d_est: Estimated 3D points
        poses_est: Estimated poses
        
    Returns:
        Dictionary of error metrics
    """
    # Align reconstructions (simple translation alignment)
    points_offset = np.mean(points_3d_gt, axis=0) - np.mean(points_3d_est, axis=0)
    points_3d_est_aligned = points_3d_est + points_offset
    
    # Compute point errors
    point_errors = np.linalg.norm(points_3d_gt - points_3d_est_aligned, axis=1)
    
    # Compute pose errors
    rotation_errors = []
    translation_errors = []
    
    for pose_gt, pose_est in zip(poses_gt, poses_est):
        # Rotation error (Frobenius norm of difference)
        R_diff = pose_gt['R'] - pose_est['R']
        rot_error = np.linalg.norm(R_diff, 'fro')
        rotation_errors.append(rot_error)
        
        # Translation error
        t_diff = pose_gt['t'].flatten() - pose_est['t'].flatten()
        trans_error = np.linalg.norm(t_diff)
        translation_errors.append(trans_error)
    
    return {
        'point_rmse': np.sqrt(np.mean(point_errors ** 2)),
        'point_mean_error': np.mean(point_errors),
        'point_max_error': np.max(point_errors),
        'rotation_rmse': np.sqrt(np.mean(np.array(rotation_errors) ** 2)),
        'translation_rmse': np.sqrt(np.mean(np.array(translation_errors) ** 2))
    }

def test_synthetic_bundle_adjustment(cfg: DictConfig):
    """Test bundle adjustment with synthetic data."""
    logger.info("=== Testing Bundle Adjustment with Synthetic Data ===")
    
    # Create synthetic reconstruction
    points_3d_gt, poses_gt, intrinsics, observations = create_synthetic_reconstruction(
        num_cameras=5, num_points=50, noise_level=0.5
    )
    
    # Add noise to create initial estimates
    points_3d_noisy, poses_noisy = add_noise_to_reconstruction(
        points_3d_gt, poses_gt, point_noise=0.2, pose_noise=0.02
    )
    
    # Compute initial error
    initial_error = compute_reconstruction_error(points_3d_gt, poses_gt, points_3d_noisy, poses_noisy)
    logger.info(f"Initial reconstruction error:")
    logger.info(f"  Point RMSE: {initial_error['point_rmse']:.4f}")
    logger.info(f"  Rotation RMSE: {initial_error['rotation_rmse']:.4f}")
    logger.info(f"  Translation RMSE: {initial_error['translation_rmse']:.4f}")
    
    # Initialize bundle adjuster
    bundle_adjuster = BundleAdjuster(cfg)
    
    # Run bundle adjustment
    logger.info("Running global bundle adjustment...")
    result = bundle_adjuster.optimize_reconstruction(
        points_3d_noisy, poses_noisy, intrinsics, observations
    )
    
    # Compute final error
    final_error = compute_reconstruction_error(
        points_3d_gt, poses_gt, result.optimized_points_3d, result.optimized_poses
    )
    logger.info(f"Final reconstruction error:")
    logger.info(f"  Point RMSE: {final_error['point_rmse']:.4f}")
    logger.info(f"  Rotation RMSE: {final_error['rotation_rmse']:.4f}")
    logger.info(f"  Translation RMSE: {final_error['translation_rmse']:.4f}")
    
    # Print optimization results
    logger.info(f"Bundle adjustment results:")
    logger.info(f"  Initial RMSE: {result.initial_rmse:.4f} pixels")
    logger.info(f"  Final RMSE: {result.final_rmse:.4f} pixels")
    logger.info(f"  Improvement: {result.initial_rmse - result.final_rmse:.4f} pixels")
    logger.info(f"  Iterations: {result.num_iterations}")
    logger.info(f"  Convergence: {result.convergence_reason}")
    logger.info(f"  Inlier ratio: {result.inlier_ratio:.3f}")
    logger.info(f"  Optimization time: {result.optimization_time:.2f}s")
    
    # Visualize results
    visualizer = SfMVisualizer(cfg)
    
    # Save initial reconstruction
    logger.info("Saving initial (noisy) reconstruction...")
    # Create camera poses dict for visualization
    camera_poses_dict = {i: np.hstack([pose['R'], pose['t']]) for i, pose in enumerate(poses_noisy)}
    visualizer.visualize_point_cloud(
        points_3d_noisy, 
        colors=None,
        camera_poses=camera_poses_dict,
        title="Bundle Adjustment Initial",
        method="export"
    )
    
    # Save optimized reconstruction
    logger.info("Saving optimized reconstruction...")
    camera_poses_opt_dict = {i: np.hstack([pose['R'], pose['t']]) for i, pose in enumerate(result.optimized_poses)}
    visualizer.visualize_point_cloud(
        result.optimized_points_3d,
        colors=None,
        camera_poses=camera_poses_opt_dict,
        title="Bundle Adjustment Optimized",
        method="export"
    )
    
    # Save ground truth for comparison
    logger.info("Saving ground truth reconstruction...")
    camera_poses_gt_dict = {i: np.hstack([pose['R'], pose['t']]) for i, pose in enumerate(poses_gt)}
    visualizer.visualize_point_cloud(
        points_3d_gt,
        colors=None,
        camera_poses=camera_poses_gt_dict,
        title="Bundle Adjustment Ground Truth",
        method="export"
    )
    
    return result

def test_local_bundle_adjustment(cfg: DictConfig):
    """Test local bundle adjustment with larger synthetic data."""
    logger.info("=== Testing Local Bundle Adjustment ===")
    
    # Create larger synthetic reconstruction
    points_3d_gt, poses_gt, intrinsics, observations = create_synthetic_reconstruction(
        num_cameras=10, num_points=100, noise_level=0.3
    )
    
    # Add noise
    points_3d_noisy, poses_noisy = add_noise_to_reconstruction(
        points_3d_gt, poses_gt, point_noise=0.15, pose_noise=0.015
    )
    
    # Initialize bundle adjuster
    bundle_adjuster = BundleAdjuster(cfg)
    
    # Run local bundle adjustment
    logger.info("Running local bundle adjustment...")
    result = bundle_adjuster.local_bundle_adjustment(
        points_3d_noisy, poses_noisy, intrinsics, observations, window_size=5
    )
    
    # Compute final error
    final_error = compute_reconstruction_error(
        points_3d_gt, poses_gt, result.optimized_points_3d, result.optimized_poses
    )
    
    logger.info(f"Local bundle adjustment results:")
    logger.info(f"  Initial RMSE: {result.initial_rmse:.4f} pixels")
    logger.info(f"  Final RMSE: {result.final_rmse:.4f} pixels")
    logger.info(f"  Improvement: {result.initial_rmse - result.final_rmse:.4f} pixels")
    logger.info(f"  Final point RMSE: {final_error['point_rmse']:.4f}")
    logger.info(f"  Optimization time: {result.optimization_time:.2f}s")
    
    return result

def test_real_data_bundle_adjustment(cfg: DictConfig):
    """Test bundle adjustment with real image data."""
    logger.info("=== Testing Bundle Adjustment with Real Data ===")
    
    # Check if test images exist
    image_dir = Path("data/test_data")
    if not image_dir.exists():
        logger.warning("Test data directory not found, skipping real data test")
        return None
    
    image_paths = list(image_dir.glob("*.jpg"))
    if len(image_paths) < 3:
        logger.warning("Need at least 3 images for bundle adjustment test")
        return None
    
    # Use first 3 images
    image_paths = sorted(image_paths)[:3]
    logger.info(f"Using images: {[p.name for p in image_paths]}")
    
    try:
        # Extract features
        feature_extractor = FeatureExtractor(cfg)
        logger.info("Extracting features...")
        
        all_features = []
        for img_path in image_paths:
            features = feature_extractor.extract_features_single(str(img_path))
            all_features.append(features)
        
        # Match features
        feature_matcher = FeatureMatcher(cfg)
        logger.info("Matching features...")
        
        matches_dict = feature_matcher.match_all_pairs(all_features)
        
        # Estimate poses
        pose_estimator = PoseEstimator(cfg)
        logger.info("Estimating poses...")
        
        # Create simple intrinsics estimate
        intrinsics = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Estimate poses between first two images
        key = (0, 1)
        if key in matches_dict:
            matches = matches_dict[key]
            # Extract matched points for pose estimation with bounds checking
            match_indices = matches['matches']
            kpts1 = matches['keypoints1']
            kpts2 = matches['keypoints2']
            
            # Filter matches to ensure indices are within bounds
            valid_matches = []
            for match in match_indices:
                if match[0] < len(kpts1) and match[1] < len(kpts2):
                    valid_matches.append(match)
            
            if len(valid_matches) < 8:
                logger.warning(f"Insufficient valid matches: {len(valid_matches)} < 8")
                return None
            
            valid_matches = np.array(valid_matches)
            points1 = kpts1[valid_matches[:, 0]]
            points2 = kpts2[valid_matches[:, 1]]
            
            logger.info(f"Using {len(valid_matches)} valid matches out of {len(match_indices)} total matches")
            pose_result = pose_estimator.estimate_two_view_geometry(points1, points2, intrinsics, intrinsics)
            
            if pose_result['success']:
                # Create pose list (first camera at origin)
                poses = [
                    {'R': np.eye(3), 't': np.zeros((3, 1))},
                    {'R': pose_result['R'], 't': pose_result['t']}
                ]
                
                # Triangulate points
                triangulator = Triangulator(cfg)
                logger.info("Triangulating points...")
                
                # Create projection matrices from poses and intrinsics
                P1 = intrinsics @ np.hstack([poses[0]['R'], poses[0]['t']])
                P2 = intrinsics @ np.hstack([poses[1]['R'], poses[1]['t']])
                
                # Use the valid matches for triangulation
                triangulation_result = triangulator.triangulate_two_view(
                    points1, points2, P1, P2, method='opencv'
                )
                
                # Convert to expected format for bundle adjustment
                if triangulation_result['points_3d'] is not None and triangulation_result['valid_mask'] is not None:
                    valid_points = triangulation_result['points_3d'][triangulation_result['valid_mask']]
                    valid_indices = np.where(triangulation_result['valid_mask'])[0]
                    
                    # Create observations for bundle adjustment
                    observations = []
                    for i, point_idx in enumerate(valid_indices):
                        # Add observation for camera 0
                        observations.append({
                            'camera_id': 0,
                            'point_id': i,
                            'point_2d': points1[point_idx]
                        })
                        # Add observation for camera 1
                        observations.append({
                            'camera_id': 1,
                            'point_id': i,
                            'point_2d': points2[point_idx]
                        })
                    
                    triangulation_result = {
                        'success': True,
                        'points_3d': valid_points,
                        'observations': observations
                    }
                else:
                    triangulation_result = {'success': False}
                
                if triangulation_result['success'] and len(triangulation_result['points_3d']) > 10:
                    points_3d = triangulation_result['points_3d']
                    observations = triangulation_result['observations']
                    
                    logger.info(f"Triangulated {len(points_3d)} points with {len(observations)} observations")
                    
                    # Run bundle adjustment
                    bundle_adjuster = BundleAdjuster(cfg)
                    logger.info("Running bundle adjustment on real data...")
                    
                    result = bundle_adjuster.optimize_reconstruction(
                        points_3d, poses, intrinsics, observations
                    )
                    
                    logger.info(f"Real data bundle adjustment results:")
                    logger.info(f"  Initial RMSE: {result.initial_rmse:.4f} pixels")
                    logger.info(f"  Final RMSE: {result.final_rmse:.4f} pixels")
                    logger.info(f"  Improvement: {result.initial_rmse - result.final_rmse:.4f} pixels")
                    logger.info(f"  Inlier ratio: {result.inlier_ratio:.3f}")
                    logger.info(f"  Optimization time: {result.optimization_time:.2f}s")
                    
                    # Visualize results
                    visualizer = SfMVisualizer(cfg)
                    
                    # Save initial reconstruction
                    camera_poses_initial = {i: np.hstack([pose['R'], pose['t']]) for i, pose in enumerate(poses)}
                    visualizer.visualize_point_cloud(
                        points_3d,
                        colors=None,
                        camera_poses=camera_poses_initial,
                        title="Bundle Adjustment Real Initial",
                        method="export"
                    )
                    
                    # Save optimized reconstruction
                    camera_poses_optimized = {i: np.hstack([pose['R'], pose['t']]) for i, pose in enumerate(result.optimized_poses)}
                    visualizer.visualize_point_cloud(
                        result.optimized_points_3d,
                        colors=None,
                        camera_poses=camera_poses_optimized,
                        title="Bundle Adjustment Real Optimized",
                        method="export"
                    )
                    
                    return result
                else:
                    logger.warning("Triangulation failed or insufficient points")
            else:
                logger.warning("Pose estimation failed")
        else:
            logger.warning("No matches found between first two images")
    
    except Exception as e:
        logger.error(f"Error in real data test: {e}")
        return None
    
    return None

@hydra.main(version_base=None, config_path="../config", config_name="base_config")
def main(cfg: DictConfig) -> None:
    """Main test function."""
    logger.info("Starting Bundle Adjustment Tests")
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Test synthetic data
    synthetic_result = test_synthetic_bundle_adjustment(cfg)
    
    # Test local bundle adjustment
    local_result = test_local_bundle_adjustment(cfg)
    
    # Test real data (if available)
    real_result = test_real_data_bundle_adjustment(cfg)
    
    logger.info("=== Bundle Adjustment Tests Complete ===")
    
    # Summary
    logger.info("Test Summary:")
    if synthetic_result:
        logger.info(f"  Synthetic data: RMSE improved by {synthetic_result.initial_rmse - synthetic_result.final_rmse:.4f} pixels")
    if local_result:
        logger.info(f"  Local BA: RMSE improved by {local_result.initial_rmse - local_result.final_rmse:.4f} pixels")
    if real_result:
        logger.info(f"  Real data: RMSE improved by {real_result.initial_rmse - real_result.final_rmse:.4f} pixels")
    
    logger.info("Check outputs/ directory for visualization files")

if __name__ == "__main__":
    main()
