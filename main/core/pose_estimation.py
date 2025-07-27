"""
Classical Pose Estimation Module for Modern SfM Pipeline

This module implements robust camera pose estimation using classical geometric
methods with COLMAP and OpenCV integration for maximum reliability.

Author: ModernSFM Pipeline
Date: July 25th, 2025
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

try:
    import pycolmap
    COLMAP_AVAILABLE = True
except ImportError:
    COLMAP_AVAILABLE = False
    logging.warning("PyColmap not available. Using OpenCV-only pose estimation.")


class PoseEstimator:
    """
    Robust pose estimation using classical geometric methods.
    
    Supports both COLMAP and OpenCV solvers with automatic fallback
    for maximum reliability across different scenarios.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.pose_config = config.get('pose_estimation', {})
        
        # RANSAC parameters
        self.ransac_config = self.pose_config.get('ransac', {})
        self.ransac_threshold = self.ransac_config.get('threshold', 1.0)
        self.ransac_confidence = self.ransac_config.get('confidence', 0.9999)
        self.ransac_max_iters = self.ransac_config.get('max_iterations', 10000)
        
        # Method preference
        self.preferred_method = self.pose_config.get('method', 'colmap')
        
        logging.info(f"Initialized PoseEstimator with method: {self.preferred_method}")
        logging.info(f"COLMAP available: {COLMAP_AVAILABLE}")
    
    def estimate_two_view_geometry(self,
                                 points1: np.ndarray,
                                 points2: np.ndarray,
                                 K1: np.ndarray,
                                 K2: np.ndarray,
                                 method: Optional[str] = None) -> Dict:
        """
        Estimate relative pose between two views.
        
        Args:
            points1, points2: [N, 2] corresponding points in pixel coordinates
            K1, K2: [3, 3] camera intrinsic matrices
            method: 'colmap', 'opencv', or None (use config default)
            
        Returns:
            Dictionary containing:
                - R: [3, 3] rotation matrix from camera 1 to camera 2
                - t: [3, 1] translation vector from camera 1 to camera 2
                - essential_matrix: [3, 3] essential matrix
                - fundamental_matrix: [3, 3] fundamental matrix
                - inlier_mask: [N] boolean mask of inliers
                - num_inliers: int number of inliers
                - inlier_ratio: float ratio of inliers
                - method_used: str method that was successful
                - success: bool whether estimation succeeded
        """
        result = {
            'R': None,
            't': None,
            'essential_matrix': None,
            'fundamental_matrix': None,
            'inlier_mask': None,
            'num_inliers': 0,
            'inlier_ratio': 0.0,
            'method_used': 'none',
            'success': False
        }
        
        if len(points1) < 8:
            logging.warning(f"Insufficient correspondences: {len(points1)} < 8")
            return result
        
        # Determine method to use
        use_method = method or self.preferred_method
        
        # Try COLMAP first if available and requested
        if use_method == 'colmap' and COLMAP_AVAILABLE:
            try:
                colmap_result = self._estimate_pose_colmap(points1, points2, K1, K2)
                if colmap_result['success']:
                    result.update(colmap_result)
                    result['method_used'] = 'colmap'
                    logging.info(f"COLMAP pose estimation successful: {result['num_inliers']}/{len(points1)} inliers")
                    return result
                else:
                    logging.warning("COLMAP pose estimation failed, falling back to OpenCV")
            except Exception as e:
                logging.warning(f"COLMAP pose estimation error: {e}, falling back to OpenCV")
        
        # Try OpenCV method
        try:
            opencv_result = self._estimate_pose_opencv(points1, points2, K1, K2)
            if opencv_result['success']:
                result.update(opencv_result)
                result['method_used'] = 'opencv'
                logging.info(f"OpenCV pose estimation successful: {result['num_inliers']}/{len(points1)} inliers")
                return result
        except Exception as e:
            logging.error(f"OpenCV pose estimation failed: {e}")
        
        logging.error("All pose estimation methods failed")
        return result
    
    def _estimate_pose_colmap(self,
                            points1: np.ndarray,
                            points2: np.ndarray,
                            K1: np.ndarray,
                            K2: np.ndarray) -> Dict:
        """Estimate pose using COLMAP's robust solvers."""
        # Normalize points to camera coordinates
        points1_norm = self._normalize_points(points1, K1)
        points2_norm = self._normalize_points(points2, K2)
        
        # COLMAP essential matrix estimation
        ransac_options = {
            'max_error': self.ransac_threshold / max(K1[0, 0], K1[1, 1]),  # Normalize threshold
            'confidence': self.ransac_confidence,
            'max_num_trials': self.ransac_max_iters,
            'min_inlier_ratio': 0.1
        }
        
        # Estimate essential matrix
        result = pycolmap.essential_matrix_estimation(
            points1_norm, points2_norm, ransac_options
        )
        
        if result is None or 'E' not in result:
            return {'success': False}
        
        E = result['E']
        inlier_mask = result.get('inliers', np.ones(len(points1), dtype=bool))
        
        # Recover pose from essential matrix
        R, t, pose_inliers = self._recover_pose_from_essential(
            E, points1_norm[inlier_mask], points2_norm[inlier_mask]
        )
        
        # Combine inlier masks
        final_inliers = np.zeros(len(points1), dtype=bool)
        inlier_indices = np.where(inlier_mask)[0]
        if len(pose_inliers) > 0 and np.any(pose_inliers):
            pose_inlier_indices = inlier_indices[pose_inliers.astype(bool)]
            final_inliers[pose_inlier_indices] = True
        
        # Compute fundamental matrix
        F = self._essential_to_fundamental(E, K1, K2)
        
        return {
            'R': R,
            't': t,
            'essential_matrix': E,
            'fundamental_matrix': F,
            'inlier_mask': final_inliers,
            'num_inliers': np.sum(final_inliers),
            'inlier_ratio': np.sum(final_inliers) / len(points1),
            'success': True
        }
    
    def _estimate_pose_opencv(self,
                            points1: np.ndarray,
                            points2: np.ndarray,
                            K1: np.ndarray,
                            K2: np.ndarray) -> Dict:
        """Estimate pose using OpenCV's robust solvers."""
        # Estimate essential matrix
        E, inlier_mask = cv2.findEssentialMat(
            points1, points2, K1,
            method=cv2.RANSAC,
            prob=self.ransac_confidence,
            threshold=self.ransac_threshold,
            maxIters=self.ransac_max_iters
        )
        
        if E is None:
            return {'success': False}
        
        inlier_mask = inlier_mask.ravel().astype(bool)
        
        # Recover pose
        _, R, t, pose_mask = cv2.recoverPose(
            E, points1[inlier_mask], points2[inlier_mask], K1
        )
        
        # Combine masks
        final_inliers = np.zeros(len(points1), dtype=bool)
        inlier_indices = np.where(inlier_mask)[0]
        pose_inliers_bool = pose_mask.ravel().astype(bool)
        if len(pose_inliers_bool) > 0:
            valid_indices = inlier_indices[pose_inliers_bool]
            final_inliers[valid_indices] = True
        
        # Compute fundamental matrix
        F = self._essential_to_fundamental(E, K1, K2)
        
        return {
            'R': R,
            't': t,
            'essential_matrix': E,
            'fundamental_matrix': F,
            'inlier_mask': final_inliers,
            'num_inliers': np.sum(final_inliers),
            'inlier_ratio': np.sum(final_inliers) / len(points1),
            'success': True
        }
    
    def estimate_pose_pnp(self,
                         points_3d: np.ndarray,
                         points_2d: np.ndarray,
                         K: np.ndarray,
                         dist_coeffs: Optional[np.ndarray] = None) -> Dict:
        """
        Estimate camera pose using PnP (Perspective-n-Point) solver.
        
        Args:
            points_3d: [N, 3] 3D points in world coordinates
            points_2d: [N, 2] corresponding 2D points in image coordinates
            K: [3, 3] camera intrinsic matrix
            dist_coeffs: [4/5/8] distortion coefficients (optional)
            
        Returns:
            Dictionary with pose estimation results
        """
        result = {
            'R': None,
            't': None,
            'rvec': None,
            'tvec': None,
            'inlier_mask': None,
            'num_inliers': 0,
            'inlier_ratio': 0.0,
            'success': False
        }
        
        if len(points_3d) < 4:
            logging.warning(f"Insufficient 3D-2D correspondences: {len(points_3d)} < 4")
            return result
        
        if dist_coeffs is None:
            dist_coeffs = np.zeros(4)
        
        try:
            # Use RANSAC-based PnP solver
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d.astype(np.float32),
                points_2d.astype(np.float32),
                K.astype(np.float32),
                dist_coeffs.astype(np.float32),
                confidence=self.ransac_confidence,
                reprojectionError=self.ransac_threshold,
                iterationsCount=self.ransac_max_iters
            )
            
            if success and inliers is not None:
                # Convert rotation vector to rotation matrix
                R, _ = cv2.Rodrigues(rvec)
                
                # Create inlier mask
                inlier_mask = np.zeros(len(points_3d), dtype=bool)
                inlier_mask[inliers.ravel()] = True
                
                result.update({
                    'R': R,
                    't': tvec,
                    'rvec': rvec,
                    'tvec': tvec,
                    'inlier_mask': inlier_mask,
                    'num_inliers': len(inliers),
                    'inlier_ratio': len(inliers) / len(points_3d),
                    'success': True
                })
                
                logging.info(f"PnP estimation successful: {len(inliers)}/{len(points_3d)} inliers")
            
        except Exception as e:
            logging.error(f"PnP estimation failed: {e}")
        
        return result
    
    def _normalize_points(self, points: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Convert pixel coordinates to normalized camera coordinates."""
        points_h = np.hstack([points, np.ones((len(points), 1))])
        K_inv = np.linalg.inv(K)
        points_norm = (K_inv @ points_h.T).T
        return points_norm[:, :2]
    
    def _recover_pose_from_essential(self,
                                   E: np.ndarray,
                                   points1_norm: np.ndarray,
                                   points2_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Recover pose from essential matrix using normalized coordinates."""
        # Decompose essential matrix
        U, S, Vt = np.linalg.svd(E)
        
        # Ensure proper rotation
        if np.linalg.det(U) < 0:
            U[:, -1] *= -1
        if np.linalg.det(Vt) < 0:
            Vt[-1, :] *= -1
        
        # Two possible rotations
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        R1 = U @ W @ Vt
        R2 = U @ W.T @ Vt
        
        # Translation (up to scale)
        t = U[:, 2].reshape(3, 1)
        
        # Test all four combinations
        poses = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
        best_pose = None
        max_positive = 0
        
        for R, t_test in poses:
            # Triangulate points to test cheirality
            positive_count = self._count_positive_depth(
                R, t_test, points1_norm, points2_norm
            )
            
            if positive_count > max_positive:
                max_positive = positive_count
                best_pose = (R, t_test)
        
        if best_pose is None:
            # Fallback to first pose
            best_pose = poses[0]
        
        R, t = best_pose
        
        # Create inlier mask based on positive depth
        inlier_mask = self._get_positive_depth_mask(R, t, points1_norm, points2_norm)
        
        return R, t, inlier_mask
    
    def _count_positive_depth(self,
                            R: np.ndarray,
                            t: np.ndarray,
                            points1_norm: np.ndarray,
                            points2_norm: np.ndarray) -> int:
        """Count points with positive depth in both cameras."""
        # Simple triangulation to check depth
        count = 0
        
        # Camera matrices (normalized coordinates)
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = np.hstack([R, t])
        
        for p1, p2 in zip(points1_norm, points2_norm):
            # Triangulate point
            A = np.array([
                p1[0] * P1[2] - P1[0],
                p1[1] * P1[2] - P1[1],
                p2[0] * P2[2] - P2[0],
                p2[1] * P2[2] - P2[1]
            ])
            
            try:
                _, _, Vt = np.linalg.svd(A)
                X = Vt[-1]
                X = X / X[3]  # Homogeneous to 3D
                
                # Check depth in both cameras
                depth1 = X[2]
                depth2 = (R @ X[:3].reshape(3, 1) + t)[2, 0]
                
                if depth1 > 0 and depth2 > 0:
                    count += 1
                    
            except:
                continue
        
        return count
    
    def _get_positive_depth_mask(self,
                               R: np.ndarray,
                               t: np.ndarray,
                               points1_norm: np.ndarray,
                               points2_norm: np.ndarray) -> np.ndarray:
        """Get mask of points with positive depth in both cameras."""
        mask = np.zeros(len(points1_norm), dtype=bool)
        
        # Camera matrices
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = np.hstack([R, t])
        
        for i, (p1, p2) in enumerate(zip(points1_norm, points2_norm)):
            try:
                # Triangulate point
                A = np.array([
                    p1[0] * P1[2] - P1[0],
                    p1[1] * P1[2] - P1[1],
                    p2[0] * P2[2] - P2[0],
                    p2[1] * P2[2] - P2[1]
                ])
                
                _, _, Vt = np.linalg.svd(A)
                X = Vt[-1]
                X = X / X[3]
                
                # Check depth
                depth1 = X[2]
                depth2 = (R @ X[:3].reshape(3, 1) + t)[2, 0]
                
                if depth1 > 0 and depth2 > 0:
                    mask[i] = True
                    
            except:
                continue
        
        return mask
    
    def _essential_to_fundamental(self,
                                E: np.ndarray,
                                K1: np.ndarray,
                                K2: np.ndarray) -> np.ndarray:
        """Convert essential matrix to fundamental matrix."""
        K1_inv = np.linalg.inv(K1)
        K2_inv = np.linalg.inv(K2)
        F = K2_inv.T @ E @ K1_inv
        return F
    
    def validate_pose(self,
                     pose_result: Dict,
                     points1: np.ndarray,
                     points2: np.ndarray) -> bool:
        """
        Validate pose estimation result using geometric constraints.
        
        Args:
            pose_result: Result from pose estimation
            points1, points2: Original point correspondences
            
        Returns:
            True if pose passes validation checks
        """
        if not pose_result['success']:
            return False
        
        try:
            # Check minimum inlier ratio
            min_inlier_ratio = 0.1
            if pose_result['inlier_ratio'] < min_inlier_ratio:
                logging.warning(f"Low inlier ratio: {pose_result['inlier_ratio']:.3f}")
                return False
            
            # Check rotation matrix validity
            R = pose_result['R']
            if not self._is_valid_rotation_matrix(R):
                logging.warning("Invalid rotation matrix")
                return False
            
            # Check translation vector
            t = pose_result['t']
            if np.linalg.norm(t) < 1e-6:
                logging.warning("Translation vector too small")
                return False
            
            # Check epipolar constraint
            if 'fundamental_matrix' in pose_result and pose_result['fundamental_matrix'] is not None:
                F = pose_result['fundamental_matrix']
                inliers = pose_result['inlier_mask']
                
                epipolar_errors = self._compute_epipolar_errors(
                    points1[inliers], points2[inliers], F
                )
                median_error = np.median(epipolar_errors)
                
                if median_error > self.ransac_threshold * 2:
                    logging.warning(f"High epipolar error: {median_error:.3f}")
                    return False
            
            return True
            
        except Exception as e:
            logging.warning(f"Pose validation failed: {e}")
            return False
    
    def _is_valid_rotation_matrix(self, R: np.ndarray) -> bool:
        """Check if matrix is a valid rotation matrix."""
        if R.shape != (3, 3):
            return False
        
        # Check orthogonality
        should_be_identity = R @ R.T
        identity = np.eye(3)
        
        # Check determinant
        det = np.linalg.det(R)
        
        return (np.allclose(should_be_identity, identity, atol=1e-2) and 
                np.allclose(det, 1.0, atol=1e-2))
    
    def _compute_epipolar_errors(self,
                               points1: np.ndarray,
                               points2: np.ndarray,
                               F: np.ndarray) -> np.ndarray:
        """Compute symmetric epipolar errors."""
        # Convert to homogeneous coordinates
        points1_h = np.hstack([points1, np.ones((len(points1), 1))])
        points2_h = np.hstack([points2, np.ones((len(points2), 1))])
        
        # Compute epipolar lines
        lines1 = F.T @ points2_h.T  # Lines in image 1
        lines2 = F @ points1_h.T    # Lines in image 2
        
        # Point-to-line distances
        errors1 = np.abs(np.sum(points1_h * lines1.T, axis=1)) / np.sqrt(lines1[0]**2 + lines1[1]**2)
        errors2 = np.abs(np.sum(points2_h * lines2.T, axis=1)) / np.sqrt(lines2[0]**2 + lines2[1]**2)
        
        # Symmetric error
        return (errors1 + errors2) / 2


def create_pose_estimator(config: Dict) -> PoseEstimator:
    """Factory function to create pose estimator."""
    return PoseEstimator(config)


if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Load configuration
    with open("../../config/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create pose estimator
    pose_estimator = create_pose_estimator(config)
    
    print("Pose Estimator initialized successfully!")
    print(f"Preferred method: {pose_estimator.preferred_method}")
    print(f"COLMAP available: {COLMAP_AVAILABLE}")
