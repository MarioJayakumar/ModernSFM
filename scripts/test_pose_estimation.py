"""
Test script for pose estimation functionality.
"""

import sys
from pathlib import Path
import logging
import os

import hydra
from omegaconf import DictConfig
import numpy as np
import cv2
import matplotlib.pyplot as plt

from main.core.feature_extraction import FeatureExtractor, load_images_from_directory
from main.core.feature_matching import FeatureMatcher
from main.core.pose_estimation import create_pose_estimator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_images(cfg):
    """Load test images for pose estimation."""
    data_dir = Path("data/test_data")
    if not data_dir.exists():
        logger.warning(f"Test data directory not found: {data_dir}")
        logger.info("Please add some test images to data/test_data/")
        return []
    
    # Load images using the same function as other test scripts
    image_paths = load_images_from_directory(data_dir, cfg.io.image_extensions)
    
    if len(image_paths) < 2:
        logger.warning("Need at least 2 test images for pose estimation")
        logger.info("Please add more test images (.jpg, .png, etc.) to data/test_data/")
        return []
    
    images = []
    for img_file in image_paths[:3]:  # Use first 3 images
        img = cv2.imread(str(img_file))
        if img is None:
            logger.warning(f"Could not load image: {img_file}")
            continue
        images.append(img)
        logger.info(f"Loaded image: {img_file.name} ({img.shape})")
    
    return images


def create_dummy_intrinsics(image_shape):
    """Create dummy camera intrinsics for testing."""
    h, w = image_shape[:2]
    
    # Assume reasonable focal length and principal point
    fx = fy = max(w, h)  # Focal length
    cx, cy = w / 2, h / 2  # Principal point
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    return K


def test_two_view_pose_estimation(cfg):
    """Test pose estimation between two views."""
    logger.info("=== Testing Two-View Pose Estimation ===")
    
    # Load components
    try:
        feature_extractor = FeatureExtractor(cfg)
        feature_matcher = FeatureMatcher(cfg)
        pose_estimator = create_pose_estimator(cfg)
        logger.info("Components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return False
    
    # Load test images
    images = load_test_images(cfg)
    if len(images) < 2:
        return False
    
    img1, img2 = images[0], images[1]
    
    # Create camera intrinsics
    K1 = create_dummy_intrinsics(img1.shape)
    K2 = create_dummy_intrinsics(img2.shape)
    
    logging.info(f"Camera intrinsics:\n{K1}")
    
    # Extract features
    logger.info("Extracting features...")
    try:
        features1 = feature_extractor.extract_features_single(Path("data/test_data") / "1.jpg")
        features2 = feature_extractor.extract_features_single(Path("data/test_data") / "2.jpg")
        logger.info(f"Extracted {features1['num_features']} and {features2['num_features']} features")
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return False
    
    # Match features
    logger.info("Matching features...")
    try:
        matches = feature_matcher.match_pairs(features1, features2)
        
        if matches['matches'] is None or len(matches['matches']) < 8:
            logger.error("Insufficient matches for pose estimation")
            return False
        
        logger.info(f"Found {matches['num_matches']} matches")
    except Exception as e:
        logger.error(f"Feature matching failed: {e}")
        return False
    
    # Get corresponding points
    kp1 = features1['keypoints']
    kp2 = features2['keypoints']
    
    points1 = np.array([kp1[m[0]] for m in matches['matches']])
    points2 = np.array([kp2[m[1]] for m in matches['matches']])
    
    # Test different pose estimation methods
    methods = ['opencv']
    if hasattr(pose_estimator, 'preferred_method'):
        methods.insert(0, pose_estimator.preferred_method)
    
    results = {}
    
    for method in methods:
        logging.info(f"\nTesting pose estimation with method: {method}")
        
        try:
            result = pose_estimator.estimate_two_view_geometry(
                points1, points2, K1, K2, method=method
            )
            
            if result['success']:
                logger.info(f"✓ {method} pose estimation successful!")
                logger.info(f"  Inliers: {result['num_inliers']}/{len(points1)} ({result['inlier_ratio']:.3f})")
                logger.info(f"  Method used: {result['method_used']}")
                
                # Validate pose
                is_valid = pose_estimator.validate_pose(result, points1, points2)
                logger.info(f"  Pose validation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
                
                results[method] = result
                
                # Print pose information
                R = result['R']
                t = result['t']
                logger.info(f"  Rotation matrix determinant: {np.linalg.det(R):.6f}")
                logger.info(f"  Translation norm: {np.linalg.norm(t):.6f}")
                
            else:
                logger.warning(f"✗ {method} pose estimation failed")
                
        except Exception as e:
            logger.error(f"✗ {method} pose estimation error: {e}")
    
    return len(results) > 0


def test_pnp_pose_estimation(cfg):
    """Test PnP pose estimation with synthetic 3D points."""
    logger.info("=== Testing PnP Pose Estimation ===")
    
    try:
        pose_estimator = create_pose_estimator(cfg)
    except Exception as e:
        logger.error(f"Failed to create pose estimator: {e}")
        return False
    
    # Create synthetic 3D points
    np.random.seed(42)
    points_3d = np.random.randn(20, 3) * 5  # 20 random 3D points
    points_3d[:, 2] += 10  # Move points in front of camera
    
    # Create synthetic camera
    K = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Create synthetic pose
    R_true = cv2.Rodrigues(np.array([0.1, 0.2, 0.3]))[0]
    t_true = np.array([[1], [2], [3]], dtype=np.float64)
    
    # Project 3D points to 2D
    points_3d_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    P = K @ np.hstack([R_true, t_true])
    points_2d_h = (P @ points_3d_h.T).T
    points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:3]
    
    # Add some noise
    points_2d += np.random.randn(*points_2d.shape) * 0.5
    
    logger.info(f"Created {len(points_3d)} synthetic 3D-2D correspondences")
    
    # Test PnP estimation
    try:
        result = pose_estimator.estimate_pose_pnp(points_3d, points_2d, K)
        
        if result['success']:
            logger.info("✓ PnP pose estimation successful!")
            logger.info(f"  Inliers: {result['num_inliers']}/{len(points_3d)} ({result['inlier_ratio']:.3f})")
            
            # Compare with ground truth
            R_est = result['R']
            t_est = result['t']
            
            R_error = np.linalg.norm(R_true - R_est, 'fro')
            t_error = np.linalg.norm(t_true - t_est)
            
            logger.info(f"  Rotation error: {R_error:.6f}")
            logger.info(f"  Translation error: {t_error:.6f}")
            
            return R_error < 0.1 and t_error < 0.5
        else:
            logger.warning("✗ PnP pose estimation failed")
            return False
    except Exception as e:
        logger.error(f"PnP pose estimation error: {e}")
        return False


def test_pose_validation(cfg):
    """Test pose validation functionality."""
    logger.info("=== Testing Pose Validation ===")
    
    try:
        pose_estimator = create_pose_estimator(cfg)
    except Exception as e:
        logger.error(f"Failed to create pose estimator: {e}")
        return False
    
    # Create valid pose result
    valid_result = {
        'success': True,
        'R': np.eye(3),  # Identity rotation
        't': np.array([[1], [0], [0]]),  # Unit translation
        'inlier_ratio': 0.8,
        'inlier_mask': np.ones(100, dtype=bool),  # All points are inliers
        'num_inliers': 80,
        'fundamental_matrix': np.array([
            [0, 0, 0.1],
            [0, 0, -0.1],
            [-0.1, 0.1, 0]
        ])
    }
    
    # Create dummy points for validation
    points1 = np.random.randn(100, 2)
    points2 = np.random.randn(100, 2)
    
    try:
        is_valid = pose_estimator.validate_pose(valid_result, points1, points2)
        logger.info(f"Valid pose validation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
        
        # Create invalid pose result (low inlier ratio)
        invalid_result = valid_result.copy()
        invalid_result['inlier_ratio'] = 0.05
        invalid_result['num_inliers'] = 5
        invalid_result['inlier_mask'] = np.zeros(100, dtype=bool)
        invalid_result['inlier_mask'][:5] = True  # Only first 5 points are inliers
        
        is_invalid = not pose_estimator.validate_pose(invalid_result, points1, points2)
        logger.info(f"Invalid pose rejection: {'✓ PASSED' if is_invalid else '✗ FAILED'}")
    except Exception as e:
        logger.error(f"Pose validation error: {e}")
        return False
    
    return is_valid and is_invalid


def visualize_pose_estimation_results(cfg):
    """Create visualization of pose estimation results."""
    logger.info("=== Creating Pose Estimation Visualization ===")
    
    try:
        # Load components
        feature_extractor = FeatureExtractor(cfg)
        feature_matcher = FeatureMatcher(cfg)
        pose_estimator = create_pose_estimator(cfg)
        
        # Load test images
        images = load_test_images(cfg)
        if len(images) < 2:
            logger.warning("Insufficient images for visualization")
            return
        
        img1, img2 = images[0], images[1]
        
        # Create camera intrinsics
        K1 = create_dummy_intrinsics(img1.shape)
        K2 = create_dummy_intrinsics(img2.shape)
        
        # Extract and match features
        features1 = feature_extractor.extract_features_single(Path("data/test_data") / "1.jpg")
        features2 = feature_extractor.extract_features_single(Path("data/test_data") / "2.jpg")
        matches = feature_matcher.match_pairs(features1, features2)
        
        if matches['matches'] is None or len(matches['matches']) < 8:
            logging.warning("Insufficient matches for visualization")
            return
        
        # Get corresponding points
        kp1 = features1['keypoints']
        kp2 = features2['keypoints']
        
        points1 = np.array([kp1[m[0]] for m in matches['matches']])
        points2 = np.array([kp2[m[1]] for m in matches['matches']])
        
        # Estimate pose
        result = pose_estimator.estimate_two_view_geometry(points1, points2, K1, K2)
        
        if not result['success']:
            logging.warning("Pose estimation failed for visualization")
            return
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Pose Estimation Results', fontsize=16)
        
        # Original images
        axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Image 1')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Image 2')
        axes[0, 1].axis('off')
        
        # Feature matches
        inliers = result['inlier_mask']
        
        # Plot all matches
        axes[1, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        axes[1, 0].scatter(points1[:, 0], points1[:, 1], c='red', s=20, alpha=0.7, label='All matches')
        if inliers is not None:
            axes[1, 0].scatter(points1[inliers, 0], points1[inliers, 1], c='green', s=20, label='Inliers')
        axes[1, 0].set_title(f'Matches on Image 1 ({result["num_inliers"]}/{len(points1)} inliers)')
        axes[1, 0].legend()
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        axes[1, 1].scatter(points2[:, 0], points2[:, 1], c='red', s=20, alpha=0.7, label='All matches')
        if inliers is not None:
            axes[1, 1].scatter(points2[inliers, 0], points2[inliers, 1], c='green', s=20, label='Inliers')
        axes[1, 1].set_title(f'Matches on Image 2 (Method: {result["method_used"]})')
        axes[1, 1].legend()
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = Path("outputs/pose_estimation_test.png")
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved pose estimation visualization to {output_path}")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")


@hydra.main(version_base=None, config_path="../config", config_name="base_config")
def test_pose_estimation(cfg: DictConfig) -> None:
    """Test pose estimation functionality."""
    
    logger.info("Starting pose estimation tests")
    
    # Run tests
    test_results = []
    
    try:
        # Test two-view pose estimation
        result1 = test_two_view_pose_estimation(cfg)
        test_results.append(("Two-view pose estimation", result1))
        
        # Test PnP pose estimation
        result2 = test_pnp_pose_estimation(cfg)
        test_results.append(("PnP pose estimation", result2))
        
        # Test pose validation
        result3 = test_pose_validation(cfg)
        test_results.append(("Pose validation", result3))
        
        # Create visualization
        visualize_pose_estimation_results(cfg)
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        test_results.append(("Test execution", False))
    
    # Print summary
    logger.info("=" * 50)
    logger.info("POSE ESTIMATION TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    for test_name, result in test_results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        logger.info("🎉 All pose estimation tests passed!")
    else:
        logger.warning("⚠️  Some pose estimation tests failed")
    
    logger.info("Pose estimation test completed!")


if __name__ == "__main__":
    test_pose_estimation()
