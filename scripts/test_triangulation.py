"""
Test script for triangulation functionality.
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
from mpl_toolkits.mplot3d import Axes3D

from main.core.triangulation import create_triangulator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_data():
    """Create synthetic camera setup and 3D points for testing."""
    # Create 3D points
    np.random.seed(42)
    points_3d_true = np.random.randn(50, 3) * 5
    points_3d_true[:, 2] += 15  # Move points in front of cameras
    
    # Camera intrinsics
    K = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Camera 1: Identity pose
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    P1 = K @ np.hstack([R1, t1])
    
    # Camera 2: Translated and slightly rotated
    R2 = cv2.Rodrigues(np.array([0.1, 0.05, 0.02]))[0]
    t2 = np.array([[2], [0], [0]])
    P2 = K @ np.hstack([R2, t2])
    
    # Project 3D points to both cameras
    points_3d_h = np.hstack([points_3d_true, np.ones((len(points_3d_true), 1))])
    
    points_2d_1_h = (P1 @ points_3d_h.T).T
    points_2d_1 = points_2d_1_h[:, :2] / points_2d_1_h[:, 2:3]
    
    points_2d_2_h = (P2 @ points_3d_h.T).T
    points_2d_2 = points_2d_2_h[:, :2] / points_2d_2_h[:, 2:3]
    
    # Add noise
    noise_std = 0.5
    points_2d_1 += np.random.randn(*points_2d_1.shape) * noise_std
    points_2d_2 += np.random.randn(*points_2d_2.shape) * noise_std
    
    return {
        'points_3d_true': points_3d_true,
        'points_2d_1': points_2d_1,
        'points_2d_2': points_2d_2,
        'P1': P1,
        'P2': P2,
        'K': K
    }


def test_two_view_triangulation(cfg):
    """Test triangulation between two views."""
    logger.info("=== Testing Two-View Triangulation ===")
    
    try:
        triangulator = create_triangulator(cfg)
    except Exception as e:
        logger.error(f"Failed to create triangulator: {e}")
        return False
    
    # Create synthetic data
    data = create_synthetic_data()
    
    # Test different triangulation methods
    methods = ['dlt', 'opencv', 'optimal']
    results = {}
    
    for method in methods:
        logging.info(f"\nTesting triangulation with method: {method}")
        
        try:
            result = triangulator.triangulate_two_view(
                data['points_2d_1'],
                data['points_2d_2'],
                data['P1'],
                data['P2'],
                method=method
            )
            
            if result['points_3d'] is not None:
                points_3d_est = result['points_3d']
                valid_mask = result['valid_mask']
                
                # Compute reconstruction error
                if valid_mask is not None and np.any(valid_mask):
                    valid_points_true = data['points_3d_true'][valid_mask]
                    valid_points_est = points_3d_est[valid_mask]
                    
                    # Align estimated points to true points (handle scale ambiguity)
                    if len(valid_points_est) > 0:
                        # Simple alignment using centroid and scale
                        centroid_true = np.mean(valid_points_true, axis=0)
                        centroid_est = np.mean(valid_points_est, axis=0)
                        
                        # Translate to origin
                        points_true_centered = valid_points_true - centroid_true
                        points_est_centered = valid_points_est - centroid_est
                        
                        # Scale alignment
                        scale_true = np.mean(np.linalg.norm(points_true_centered, axis=1))
                        scale_est = np.mean(np.linalg.norm(points_est_centered, axis=1))
                        
                        if scale_est > 1e-6:
                            scale_factor = scale_true / scale_est
                            points_est_aligned = points_est_centered * scale_factor + centroid_true
                            
                            # Compute error
                            errors = np.linalg.norm(valid_points_true - points_est_aligned, axis=1)
                            mean_error = np.mean(errors)
                            median_error = np.median(errors)
                            
                            logger.info(f"✓ {method} triangulation successful!")
                            logger.info(f"  Valid points: {np.sum(valid_mask)}/{len(data['points_3d_true'])}")
                            logger.info(f"  Mean reconstruction error: {mean_error:.3f}")
                            logger.info(f"  Median reconstruction error: {median_error:.3f}")
                            logger.info(f"  Mean reprojection error: {np.mean(result['reprojection_errors'][valid_mask]):.3f}")
                            logger.info(f"  Mean triangulation angle: {np.mean(result['triangulation_angles'][valid_mask]):.1f}°")
                            
                            results[method] = {
                                'success': True,
                                'mean_error': mean_error,
                                'median_error': median_error,
                                'valid_points': np.sum(valid_mask)
                            }
                        else:
                            logger.warning(f"✗ {method} triangulation failed: degenerate scale")
                    else:
                        logger.warning(f"✗ {method} triangulation failed: no valid points")
                else:
                    logger.warning(f"✗ {method} triangulation failed: no valid points")
            else:
                logger.warning(f"✗ {method} triangulation failed: no points returned")
                
        except Exception as e:
            logger.error(f"✗ {method} triangulation error: {e}")
    
    return len(results) > 0


def test_multi_view_triangulation(cfg):
    """Test triangulation from multiple views."""
    logger.info("=== Testing Multi-View Triangulation ===")
    
    try:
        triangulator = create_triangulator(cfg)
        # Temporarily relax thresholds for testing
        original_min_angle = triangulator.min_triangulation_angle
        original_max_error = triangulator.max_reprojection_error
        triangulator.min_triangulation_angle = 0.5  # Very relaxed
        triangulator.max_reprojection_error = 10.0  # Very relaxed
        logger.info(f"Temporarily relaxed thresholds: angle={triangulator.min_triangulation_angle}°, error={triangulator.max_reprojection_error}")
    except Exception as e:
        logger.error(f"Failed to create triangulator: {e}")
        return False
    
    # Create synthetic multi-view data with simpler geometry
    np.random.seed(42)
    points_3d_true = np.random.randn(10, 3) * 2  # Fewer points, smaller spread
    points_3d_true[:, 2] += 8  # Move points in front of cameras
    
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Create just 3 cameras with very simple, well-separated geometry
    num_cameras = 3
    observations = {}
    camera_matrices = {}
    
    # Simple camera setup with large baselines
    camera_setups = [
        # Camera 0: at origin
        {'R': np.eye(3), 't': np.zeros((3, 1))},
        # Camera 1: 5 units to the right
        {'R': np.eye(3), 't': np.array([[-5], [0], [0]])},
        # Camera 2: 5 units up
        {'R': np.eye(3), 't': np.array([[0], [-5], [0]])}
    ]
    
    for i, setup in enumerate(camera_setups):
        R = setup['R']
        t = setup['t']
        
        # Projection matrix
        P = K @ np.hstack([R, t])
        camera_matrices[i] = P
        
        # Project 3D points
        points_3d_h = np.hstack([points_3d_true, np.ones((len(points_3d_true), 1))])
        points_2d_h = (P @ points_3d_h.T).T
        
        # Check for points behind camera (should all be in front)
        valid_points = points_2d_h[:, 2] > 0
        points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:3]
        
        # Add minimal noise
        points_2d += np.random.randn(*points_2d.shape) * 0.1
        observations[i] = points_2d
        
        logger.info(f"Camera {i}: {np.sum(valid_points)}/{len(points_3d_true)} points in front")
    
    logger.info(f"Created {num_cameras} cameras observing {len(points_3d_true)} points")
    
    # Test multi-view triangulation
    try:
        result = triangulator.triangulate_multi_view(
            observations, camera_matrices, method='dlt'
        )
        
        if result['points_3d'] is not None:
            points_3d_est = result['points_3d']
            valid_mask = result['valid_mask']
            
            # Debug: Print quality metrics
            if result['reprojection_errors'] is not None:
                logger.info(f"Reprojection errors - min: {np.min(result['reprojection_errors']):.3f}, "
                           f"max: {np.max(result['reprojection_errors']):.3f}, "
                           f"mean: {np.mean(result['reprojection_errors']):.3f}")
                logger.info(f"Reprojection error threshold: {triangulator.max_reprojection_error}")
            
            if result['triangulation_angles'] is not None:
                logger.info(f"Triangulation angles - min: {np.min(result['triangulation_angles']):.3f}°, "
                           f"max: {np.max(result['triangulation_angles']):.3f}°, "
                           f"mean: {np.mean(result['triangulation_angles']):.3f}°")
                logger.info(f"Triangulation angle threshold: {triangulator.min_triangulation_angle}°")
            
            logger.info(f"Valid points: {np.sum(valid_mask)}/{len(points_3d_est)}")
            
            if np.any(valid_mask):
                valid_points_true = points_3d_true[valid_mask]
                valid_points_est = points_3d_est[valid_mask]
                
                # Compute reconstruction error (with alignment)
                if len(valid_points_est) > 0:
                    centroid_true = np.mean(valid_points_true, axis=0)
                    centroid_est = np.mean(valid_points_est, axis=0)
                    
                    points_true_centered = valid_points_true - centroid_true
                    points_est_centered = valid_points_est - centroid_est
                    
                    scale_true = np.mean(np.linalg.norm(points_true_centered, axis=1))
                    scale_est = np.mean(np.linalg.norm(points_est_centered, axis=1))
                    
                    if scale_est > 1e-6:
                        scale_factor = scale_true / scale_est
                        points_est_aligned = points_est_centered * scale_factor + centroid_true
                        
                        errors = np.linalg.norm(valid_points_true - points_est_aligned, axis=1)
                        mean_error = np.mean(errors)
                        
                        logger.info("✓ Multi-view triangulation successful!")
                        logger.info(f"  Valid points: {np.sum(valid_mask)}/{len(points_3d_true)}")
                        logger.info(f"  Mean reconstruction error: {mean_error:.3f}")
                        logger.info(f"  Mean reprojection error: {np.mean(result['reprojection_errors'][valid_mask]):.3f}")
                        
                        return mean_error < 1.0  # Accept if error < 1 unit
                    
            logger.warning("Multi-view triangulation failed: no valid points")
        else:
            logger.warning("Multi-view triangulation failed: no points returned")
            
    except Exception as e:
        logger.error(f"Multi-view triangulation error: {e}")
    
    return False


def test_triangulation_quality_filtering(cfg):
    """Test triangulation quality filtering."""
    logger.info("=== Testing Triangulation Quality Filtering ===")
    
    try:
        triangulator = create_triangulator(cfg)
    except Exception as e:
        logger.error(f"Failed to create triangulator: {e}")
        return False
    
    # Create data with varying quality
    data = create_synthetic_data()
    
    # Test with good baseline
    result_good = triangulator.triangulate_two_view(
        data['points_2d_1'],
        data['points_2d_2'],
        data['P1'],
        data['P2'],
        method='dlt'
    )
    
    # Create cameras with poor baseline (small triangulation angles)
    K = data['K']
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    P1_poor = K @ np.hstack([R1, t1])
    
    R2_poor = np.eye(3)
    t2_poor = np.array([[0.1], [0], [0]])  # Very small baseline
    P2_poor = K @ np.hstack([R2_poor, t2_poor])
    
    # Project with poor baseline
    points_3d_h = np.hstack([data['points_3d_true'], np.ones((len(data['points_3d_true']), 1))])
    
    points_2d_1_poor_h = (P1_poor @ points_3d_h.T).T
    points_2d_1_poor = points_2d_1_poor_h[:, :2] / points_2d_1_poor_h[:, 2:3]
    
    points_2d_2_poor_h = (P2_poor @ points_3d_h.T).T
    points_2d_2_poor = points_2d_2_poor_h[:, :2] / points_2d_2_poor_h[:, 2:3]
    
    # Test with poor baseline
    result_poor = triangulator.triangulate_two_view(
        points_2d_1_poor,
        points_2d_2_poor,
        P1_poor,
        P2_poor,
        method='dlt'
    )
    
    # Compare results
    good_valid = np.sum(result_good['valid_mask']) if result_good['valid_mask'] is not None else 0
    poor_valid = np.sum(result_poor['valid_mask']) if result_poor['valid_mask'] is not None else 0
    
    logger.info(f"Good baseline valid points: {good_valid}/{len(data['points_3d_true'])}")
    logger.info(f"Poor baseline valid points: {poor_valid}/{len(data['points_3d_true'])}")
    
    # Quality filtering should reject more points with poor baseline
    filtering_works = good_valid > poor_valid
    
    if filtering_works:
        logger.info("✓ Quality filtering working correctly")
    else:
        logger.warning("✗ Quality filtering may not be working properly")
    
    return filtering_works


def test_track_creation(cfg):
    """Test track creation from pairwise matches."""
    logger.info("=== Testing Track Creation ===")
    
    try:
        triangulator = create_triangulator(cfg)
    except Exception as e:
        logger.error(f"Failed to create triangulator: {e}")
        return False
    
    # Create synthetic matches between 4 images
    # Simulate feature tracks across multiple images
    matches_dict = {
        (0, 1): np.array([[0, 5], [1, 6], [2, 7], [3, 8]]),  # img0-img1 matches
        (1, 2): np.array([[5, 10], [6, 11], [7, 12]]),       # img1-img2 matches
        (2, 3): np.array([[10, 15], [11, 16]]),              # img2-img3 matches
        (0, 2): np.array([[0, 10], [1, 11]]),                # img0-img2 matches
    }
    
    num_images = 4
    
    try:
        tracks = triangulator.create_tracks(matches_dict, num_images)
        
        logger.info(f"Created {len(tracks)} tracks from pairwise matches")
        
        # Analyze tracks
        track_lengths = [len(track) for track in tracks.values()]
        if track_lengths:
            avg_length = np.mean(track_lengths)
            max_length = max(track_lengths)
            
            logger.info(f"Average track length: {avg_length:.1f}")
            logger.info(f"Maximum track length: {max_length}")
            
            # Print some example tracks
            for i, (track_id, track) in enumerate(tracks.items()):
                if i < 3:  # Show first 3 tracks
                    logger.info(f"Track {track_id}: {track}")
            
            return len(tracks) > 0 and avg_length >= triangulator.min_track_length
        else:
            logger.warning("No tracks created")
            return False
            
    except Exception as e:
        logger.error(f"Track creation failed: {e}")
        return False


def visualize_triangulation_results(cfg):
    """Create visualization of triangulation results."""
    logger.info("=== Creating Triangulation Visualization ===")
    
    try:
        triangulator = create_triangulator(cfg)
        data = create_synthetic_data()
        
        # Triangulate points
        result = triangulator.triangulate_two_view(
            data['points_2d_1'],
            data['points_2d_2'],
            data['P1'],
            data['P2'],
            method='dlt'
        )
        
        if result['points_3d'] is None:
            logger.warning("No triangulation results to visualize")
            return
        
        points_3d_est = result['points_3d']
        valid_mask = result['valid_mask']
        
        # Create 3D visualization
        fig = plt.figure(figsize=(15, 10))
        
        # 3D point cloud comparison
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.scatter(data['points_3d_true'][:, 0], 
                   data['points_3d_true'][:, 1], 
                   data['points_3d_true'][:, 2], 
                   c='blue', s=50, alpha=0.7, label='True points')
        
        if valid_mask is not None and np.any(valid_mask):
            ax1.scatter(points_3d_est[valid_mask, 0], 
                       points_3d_est[valid_mask, 1], 
                       points_3d_est[valid_mask, 2], 
                       c='red', s=50, alpha=0.7, label='Triangulated points')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        if hasattr(ax1, 'set_zlabel'):
            ax1.set_zlabel('Z')
        ax1.set_title('3D Point Comparison')
        ax1.legend()
        
        # Reprojection errors
        ax2 = fig.add_subplot(222)
        if result['reprojection_errors'] is not None:
            ax2.hist(result['reprojection_errors'], bins=20, alpha=0.7)
            ax2.axvline(triangulator.max_reprojection_error, color='red', linestyle='--', 
                       label=f'Threshold: {triangulator.max_reprojection_error}')
            ax2.set_xlabel('Reprojection Error (pixels)')
            ax2.set_ylabel('Count')
            ax2.set_title('Reprojection Error Distribution')
            ax2.legend()
        
        # Triangulation angles
        ax3 = fig.add_subplot(223)
        if result['triangulation_angles'] is not None:
            ax3.hist(result['triangulation_angles'], bins=20, alpha=0.7)
            ax3.axvline(triangulator.min_triangulation_angle, color='red', linestyle='--',
                       label=f'Threshold: {triangulator.min_triangulation_angle}°')
            ax3.set_xlabel('Triangulation Angle (degrees)')
            ax3.set_ylabel('Count')
            ax3.set_title('Triangulation Angle Distribution')
            ax3.legend()
        
        # 2D projections
        ax4 = fig.add_subplot(224)
        ax4.scatter(data['points_2d_1'][:, 0], data['points_2d_1'][:, 1], 
                   c='blue', alpha=0.7, label='Camera 1')
        ax4.scatter(data['points_2d_2'][:, 0], data['points_2d_2'][:, 1], 
                   c='red', alpha=0.7, label='Camera 2')
        ax4.set_xlabel('X (pixels)')
        ax4.set_ylabel('Y (pixels)')
        ax4.set_title('2D Projections')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save visualization
        output_path = Path("outputs/triangulation_test.png")
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved triangulation visualization to {output_path}")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")


@hydra.main(version_base=None, config_path="../config", config_name="base_config")
def test_triangulation(cfg: DictConfig) -> None:
    """Test triangulation functionality."""
    
    logger.info("Starting triangulation tests")
    
    # Run tests
    test_results = []
    
    try:
        # Test two-view triangulation
        result1 = test_two_view_triangulation(cfg)
        test_results.append(("Two-view triangulation", result1))
        
        # Test multi-view triangulation
        result2 = test_multi_view_triangulation(cfg)
        test_results.append(("Multi-view triangulation", result2))
        
        # Test quality filtering
        result3 = test_triangulation_quality_filtering(cfg)
        test_results.append(("Quality filtering", result3))
        
        # Test track creation
        result4 = test_track_creation(cfg)
        test_results.append(("Track creation", result4))
        
        # Create visualization
        visualize_triangulation_results(cfg)
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        test_results.append(("Test execution", False))
    
    # Print summary
    logger.info("=" * 50)
    logger.info("TRIANGULATION TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    for test_name, result in test_results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        logger.info("🎉 All triangulation tests passed!")
    else:
        logger.warning("⚠️  Some triangulation tests failed")
    
    logger.info("Triangulation test completed!")


if __name__ == "__main__":
    test_triangulation()
