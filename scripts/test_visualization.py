"""
Test script for 3D visualization functionality.

This script demonstrates the various visualization options available for
viewing 3D point cloud data over SSH connections.
"""

import sys
from pathlib import Path
import logging
import os

import hydra
from omegaconf import DictConfig
import numpy as np
import cv2

from main.core.triangulation import create_triangulator
from main.utils.visualization import create_visualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_reconstruction_data():
    """Create synthetic reconstruction data for testing visualization."""
    # Create 3D points in a simple scene
    np.random.seed(42)
    
    # Create a cube of points
    n_points = 100
    points_3d = np.random.randn(n_points, 3) * 2
    points_3d[:, 2] += 10  # Move points in front of cameras
    
    # Add some structure - create a simple building-like shape
    # Ground plane
    ground_points = np.random.uniform(-5, 5, (20, 2))
    ground_z = np.zeros((20, 1))
    ground_3d = np.hstack([ground_points, ground_z + 8])
    
    # Walls
    wall_points = []
    for i in range(4):
        angle = i * np.pi / 2
        x = np.cos(angle) * 3 + np.random.normal(0, 0.1, 10)
        y = np.sin(angle) * 3 + np.random.normal(0, 0.1, 10)
        z = np.random.uniform(8, 12, 10)
        wall_3d = np.column_stack([x, y, z])
        wall_points.append(wall_3d)
    
    # Combine all points
    all_points = np.vstack([points_3d, ground_3d] + wall_points)
    
    # Create synthetic reprojection errors (for coloring)
    errors = np.random.exponential(1.0, len(all_points))
    
    # Create camera poses
    camera_poses = {}
    
    # Camera 0: at origin looking forward
    R0 = np.eye(3)
    t0 = np.zeros((3, 1))
    camera_poses[0] = np.hstack([R0, t0])
    
    # Camera 1: moved to the right
    R1 = np.eye(3)
    t1 = np.array([[-3], [0], [0]])
    camera_poses[1] = np.hstack([R1, t1])
    
    # Camera 2: moved up and rotated slightly
    R2 = cv2.Rodrigues(np.array([0.1, 0.0, 0.0]))[0]
    t2 = np.array([[0], [-2], [-1]])
    camera_poses[2] = np.hstack([R2, t2])
    
    # Create triangulation result format
    triangulation_result = {
        'points_3d': all_points,
        'reprojection_errors': errors,
        'triangulation_angles': np.random.uniform(5, 45, len(all_points)),
        'valid_mask': errors < 3.0,  # Mark points with low error as valid
        'method_used': 'synthetic'
    }
    
    return triangulation_result, camera_poses


def test_visualization_methods(cfg):
    """Test all visualization methods."""
    logger.info("=== Testing 3D Visualization Methods ===")
    
    # Create visualizer
    try:
        visualizer = create_visualizer(cfg)
    except Exception as e:
        logger.error(f"Failed to create visualizer: {e}")
        return False
    
    # Create synthetic data
    triangulation_result, camera_poses = create_synthetic_reconstruction_data()
    
    logger.info(f"Created synthetic scene with {len(triangulation_result['points_3d'])} points")
    logger.info(f"Valid points: {np.sum(triangulation_result['valid_mask'])}")
    logger.info(f"Camera poses: {len(camera_poses)}")
    
    # Test different visualization methods
    methods_to_test = ['terminal', 'export', 'matplotlib', 'web']
    
    results = {}
    
    for method in methods_to_test:
        logger.info(f"\n--- Testing {method.upper()} visualization ---")
        
        try:
            result_path = visualizer.visualize_reconstruction(
                triangulation_result,
                camera_poses,
                title=f"Test Reconstruction ({method})",
                method=method
            )
            
            if result_path:
                logger.info(f"✓ {method} visualization successful!")
                logger.info(f"  Output: {result_path}")
                results[method] = True
            else:
                logger.warning(f"✗ {method} visualization returned no result")
                results[method] = False
                
        except Exception as e:
            logger.error(f"✗ {method} visualization failed: {e}")
            results[method] = False
    
    return results


def test_direct_point_cloud_visualization(cfg):
    """Test direct point cloud visualization without triangulation results."""
    logger.info("=== Testing Direct Point Cloud Visualization ===")
    
    try:
        visualizer = create_visualizer(cfg)
    except Exception as e:
        logger.error(f"Failed to create visualizer: {e}")
        return False
    
    # Create simple point cloud
    np.random.seed(123)
    points_3d = np.random.randn(50, 3) * 3
    points_3d[:, 2] += 8  # Move in front
    
    # Create colors (rainbow effect based on height)
    z_normalized = (points_3d[:, 2] - np.min(points_3d[:, 2])) / (np.max(points_3d[:, 2]) - np.min(points_3d[:, 2]))
    colors = np.zeros((len(points_3d), 3))
    colors[:, 0] = z_normalized  # Red increases with height
    colors[:, 1] = 1 - z_normalized  # Green decreases with height
    colors[:, 2] = 0.5  # Constant blue
    
    # Test auto-detection
    try:
        result_path = visualizer.visualize_point_cloud(
            points_3d,
            colors,
            title="Direct Point Cloud Test",
            method="auto"
        )
        
        if result_path:
            logger.info("✓ Direct point cloud visualization successful!")
            logger.info(f"  Output: {result_path}")
            return True
        else:
            logger.warning("✗ Direct point cloud visualization failed")
            return False
            
    except Exception as e:
        logger.error(f"Direct point cloud visualization error: {e}")
        return False


def test_environment_detection(cfg):
    """Test environment detection capabilities."""
    logger.info("=== Testing Environment Detection ===")
    
    try:
        visualizer = create_visualizer(cfg)
    except Exception as e:
        logger.error(f"Failed to create visualizer: {e}")
        return False
    
    # Test environment detection methods
    is_ssh = visualizer._is_ssh_session()
    has_x11 = visualizer._has_x11_forwarding()
    best_method = visualizer._detect_best_method()
    
    logger.info(f"SSH session detected: {is_ssh}")
    logger.info(f"X11 forwarding available: {has_x11}")
    logger.info(f"Best visualization method: {best_method}")
    
    # Print environment variables for debugging
    ssh_vars = ['SSH_CLIENT', 'SSH_TTY', 'SSH_CONNECTION', 'DISPLAY']
    logger.info("Environment variables:")
    for var in ssh_vars:
        value = os.environ.get(var, 'Not set')
        logger.info(f"  {var}: {value}")
    
    return True


def demonstrate_usage_examples(cfg):
    """Demonstrate practical usage examples."""
    logger.info("=== Usage Examples ===")
    
    logger.info("\n1. BASIC USAGE:")
    logger.info("   from main.utils.visualization import create_visualizer")
    logger.info("   visualizer = create_visualizer(config)")
    logger.info("   visualizer.visualize_reconstruction(triangulation_result)")
    
    logger.info("\n2. SPECIFIC METHOD:")
    logger.info("   # For SSH without X11 forwarding")
    logger.info("   visualizer.visualize_point_cloud(points_3d, method='export')")
    logger.info("   # Creates PLY, OBJ, and HTML files you can download")
    
    logger.info("\n3. SSH WITH X11 FORWARDING:")
    logger.info("   # Connect with: ssh -X your_server")
    logger.info("   visualizer.visualize_point_cloud(points_3d, method='matplotlib')")
    logger.info("   # Opens interactive 3D plot")
    
    logger.info("\n4. WEB VIEWER WITH SSH TUNNEL:")
    logger.info("   # Local: ssh -L 8080:localhost:8080 your_server")
    logger.info("   visualizer.visualize_point_cloud(points_3d, method='web')")
    logger.info("   # Server: cd outputs && python -m http.server 8080")
    logger.info("   # Browser: http://localhost:8080/visualizations/file.html")
    
    logger.info("\n5. QUICK TERMINAL PREVIEW:")
    logger.info("   visualizer.visualize_point_cloud(points_3d, method='terminal')")
    logger.info("   # Shows ASCII art and statistics in terminal")


def integration_example_with_triangulation(cfg):
    """Show how to integrate with existing triangulation workflow."""
    logger.info("=== Integration Example ===")
    
    try:
        # Create triangulator (your existing workflow)
        triangulator = create_triangulator(cfg)
        
        # Create visualizer
        visualizer = create_visualizer(cfg)
        
        # Simulate your existing triangulation workflow
        triangulation_result, camera_poses = create_synthetic_reconstruction_data()
        
        logger.info("Example integration with your existing workflow:")
        logger.info("")
        logger.info("# After running triangulation:")
        logger.info("result = triangulator.triangulate_two_view(points1, points2, P1, P2)")
        logger.info("")
        logger.info("# Visualize the results:")
        logger.info("visualizer.visualize_reconstruction(result, camera_poses)")
        logger.info("")
        logger.info("# Or just the point cloud:")
        logger.info("if result['points_3d'] is not None:")
        logger.info("    valid_points = result['points_3d'][result['valid_mask']]")
        logger.info("    visualizer.visualize_point_cloud(valid_points)")
        
        # Actually run the visualization
        result_path = visualizer.visualize_reconstruction(
            triangulation_result,
            camera_poses,
            title="Integration Example",
            method="auto"
        )
        
        if result_path:
            logger.info(f"\n✓ Integration example completed successfully!")
            logger.info(f"  Check output: {result_path}")
            return True
        else:
            logger.warning("✗ Integration example failed")
            return False
            
    except Exception as e:
        logger.error(f"Integration example error: {e}")
        return False


@hydra.main(version_base=None, config_path="../config", config_name="base_config")
def test_visualization(cfg: DictConfig) -> None:
    """Test 3D visualization functionality."""
    
    logger.info("Starting 3D visualization tests")
    logger.info("=" * 60)
    
    # Run tests
    test_results = []
    
    try:
        # Test environment detection
        result1 = test_environment_detection(cfg)
        test_results.append(("Environment detection", result1))
        
        # Test visualization methods
        method_results = test_visualization_methods(cfg)
        if isinstance(method_results, dict):
            for method, success in method_results.items():
                test_results.append((f"{method} visualization", success))
        else:
            test_results.append(("Visualization methods", method_results))
        
        # Test direct point cloud visualization
        result2 = test_direct_point_cloud_visualization(cfg)
        test_results.append(("Direct point cloud", result2))
        
        # Show usage examples
        demonstrate_usage_examples(cfg)
        
        # Integration example
        result3 = integration_example_with_triangulation(cfg)
        test_results.append(("Integration example", result3))
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        test_results.append(("Test execution", False))
    
    # Print summary
    logger.info("=" * 60)
    logger.info("3D VISUALIZATION TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    for test_name, result in test_results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        logger.info("🎉 All visualization tests passed!")
    else:
        logger.warning("⚠️  Some visualization tests failed")
    
    # Final recommendations
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATIONS FOR YOUR SSH SETUP")
    logger.info("=" * 60)
    
    if os.environ.get('SSH_CONNECTION'):
        logger.info("✓ SSH session detected")
        if os.environ.get('DISPLAY'):
            logger.info("✓ X11 forwarding available - use method='matplotlib' for interactive plots")
        else:
            logger.info("⚠️  No X11 forwarding - use method='export' to create files you can download")
            logger.info("   Or use method='web' with SSH tunneling for interactive viewing")
    else:
        logger.info("ℹ️  Local session - all methods should work")
    
    logger.info("\nVisualization test completed!")


if __name__ == "__main__":
    test_visualization()
